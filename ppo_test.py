from gymnasium.wrappers import AtariPreprocessing,FrameStack
import gymnasium as gym
import numpy as np
import torch
from ppo_model import ppo_net


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
episodes = 10
env_id = 'BreakoutNoFrameskip-v4'
env = gym.make(env_id, render_mode='human')
envs = AtariPreprocessing(env,scale_obs=True)
envs = FireResetEnv(envs)
envs = FrameStack(envs,4)

obs_space_channel = envs.observation_space.shape
action_space_out = envs.action_space.n
ppo_network = ppo_net(obs_space_channel,action_space_out).to(device)
ppo_network.load_state_dict(torch.load('ppo_net1.pth'))

re = []
for episode in range(1,episodes+1):
  reward_episode=0
  state = envs.reset()[0]
  state = np.array(state)
  state = torch.Tensor(state).to(device)
  while True:
    action = ppo_network.select_action(state.unsqueeze(0))
    newstate,reward,terminated,truncated,_ = envs.step(action)
    newstate = np.array(newstate)
    newstate = torch.Tensor(newstate).to(device)
    state = newstate
    reward_episode += reward
    if terminated:
        break
  print(f"episode {episode} | reward_per_episode {reward_episode}")
  re.append(reward_episode)
print(f"average reward over the episodes {sum(re)/episodes}")
    
 
