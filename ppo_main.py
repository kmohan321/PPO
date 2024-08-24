from gymnasium.wrappers import AtariPreprocessing,FrameStack
import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv
import torch
from ppo_model import ppo_net
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

config_dict = {
    'env_id':'BreakoutNoFrameskip-v4',
    'n_envs' : 8,
    'exp_name': 'ppo',
    'n_steps': 128,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'timesteps': 10000000,
    'batch_size':8*128,
    'decay_lr':True,
    'lr':2.5e-4,
    'gamma':0.99,
    'gae_lambda':0.95,
    'update_epoch':4,
    'entropy_weight':0.01,
    'value_weight':0.5,
    'max_grad_norm':0.5,
    'eps':0.1,
    'n_mini_batch':4
}
config_dict['mini_batch_size'] = int(config_dict['batch_size']//config_dict['n_mini_batch'])
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

config = Config(config_dict)

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.
    Args:
        env (gym.Env): The environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)
    
    def reward(self, reward: float) -> float:
        return np.sign(reward)

class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    Args:
        env (gym.Env): The environment to wrap
    """
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info


# atari preprocessing

def AtariWrappers(env):
    env = AtariPreprocessing(
        env,
        noop_max=30,                   
        frame_skip=4,  
        screen_size=84,                
        terminal_on_life_loss=True,    
        grayscale_obs=True,            
        scale_obs=True,                
    )
    return env

def make_env(gym_id, idx, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = AtariWrappers(env)                           
        if 'FIRE' in env.unwrapped.get_action_meanings():  
            env = FireResetEnv(env)                        
        env = ClipRewardEnv(env)                           
        env = FrameStack(env, 4)              
        return env
    return thunk

envs = SyncVectorEnv([
    make_env(config.env_id,
              i, 
              config.exp_name) for i in range(config.n_envs)
    ])
# print(envs.reset()[0].shape)

# memory

states = torch.zeros((config.n_steps,config.n_envs) + envs.single_observation_space.shape).to(config.device)
actions = torch.zeros((config.n_steps,config.n_envs)).to(config.device)
values = torch.zeros((config.n_steps,config.n_envs)).to(config.device)
log_probs = torch.zeros((config.n_steps,config.n_envs)).to(config.device)
rewards = torch.zeros((config.n_steps,config.n_envs)).to(config.device)
dones = torch.zeros((config.n_steps,config.n_envs)).to(config.device)

# train loop
total_steps = 0
episodes = config.timesteps//config.batch_size

obs_space_channel = envs.single_observation_space.shape
action_space_out = envs.single_action_space.n
# print(obs_space_channel)
ppo_network = ppo_net(obs_space_channel,action_space_out).to(config.device)
optimizer = optim.Adam(ppo_network.parameters(),lr=config.lr,eps=1e-5)
loss_fn = nn.MSELoss()

state = torch.Tensor(envs.reset()[0]).to(config.device)
done = torch.zeros(config.n_envs).to(config.device)

reward_episode = np.zeros([config.n_envs])
for i in range(1,episodes+1):
      
      if config.decay_lr:
        fraction = 1.0 - ((i - 1.0) / episodes)
        lr_current = fraction * config.lr
        optimizer.param_groups[0]['lr'] = lr_current

      for j in range(0,config.n_steps):
        total_steps += 1 * config.n_envs
        states[j] = state
        dones[j] = done

        with torch.no_grad():
          action, log_prob, _, value = ppo_network(state)
          values[j] = value.flatten()
        log_probs[j] = log_prob
        actions[j] = action

        state ,reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated,truncated)
        rewards[j] = torch.Tensor(reward).to(config.device).view(-1)
        state = torch.Tensor(state).to(config.device)
        done = torch.Tensor(done).to(config.device)

        reward_episode += reward

        if 'final_observation' in info.keys():
          for k , done_flag in enumerate(info['_final_observation']):
            if done_flag:
              writer.add_scalar('rewards_per episode',reward_episode[k],total_steps)
              print(f" timesteps {total_steps} | reward_per_episode {reward_episode[k]} ")
              reward_episode[k] = 0 


      with torch.no_grad():
            next_value = ppo_network.forward_value(state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(config.device)
            lastgaelam = 0
            for t in reversed(range(config.n_steps)):
                if t == config.n_steps - 1:
                    next_non_terminal = 1.0 - done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + config.gamma * next_values * next_non_terminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * next_non_terminal * lastgaelam
            returns = advantages + values

      b_states = states.reshape((-1,) + envs.single_observation_space.shape)
      b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
      b_logprobs= log_probs.reshape(-1)
      b_advantages = advantages.reshape(-1)
      b_returns = returns.reshape(-1)
      b_values = values.reshape(-1)

      # batch_indicies
      batch_indicies = np.arange(config.batch_size)

      for update in range(config.update_epoch):
          np.random.shuffle(batch_indicies)
          # batch splitting into minibatches

          for min_batch_id in range(0,config.batch_size,config.mini_batch_size):

            end_id = min_batch_id + config.mini_batch_size
            mini_batch_indices = batch_indicies[min_batch_id:end_id]

            _, new_log_prob,entropy,new_values = ppo_network(b_states[mini_batch_indices],b_actions.long()[mini_batch_indices])

            ratio = torch.exp(new_log_prob-b_logprobs[mini_batch_indices])

            mini_advantages = (b_advantages[mini_batch_indices]- b_advantages[mini_batch_indices].mean())/(b_advantages[mini_batch_indices].std() + 1e-8)

            clipped = torch.clip(ratio,1-config.eps,1+config.eps)

            loss_policy  = -torch.min(ratio*mini_advantages,clipped*mini_advantages).mean()

            new_values= new_values.view(-1)

            loss_value = loss_fn(new_values,b_values[mini_batch_indices])

            overall_loss = loss_policy - config.entropy_weight*entropy.mean() + config.value_weight*loss_value

            optimizer.zero_grad()
            overall_loss.backward()
            nn.utils.clip_grad_norm_(ppo_network.parameters(), config.max_grad_norm)
            optimizer.step()

      writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], total_steps)
      writer.add_scalar('losses/value_loss', loss_value.item(), total_steps)
      writer.add_scalar('losses/policy_loss', loss_policy.item(), total_steps)
      if (i%4)==0:
        torch.save(ppo_network.state_dict(),'ppo_net1.pth')
        print(f" saved the model at {i}")













