#this application displays a simple policy where we accelerate left when the pole is leaning to the left
#and accelerates right when the pole is leaning to the right
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def basic_policy(obs):
    print(obs)
    angle = obs[2] #angle that the pole is leaning
    
    return 0 if angle < 0 else 1 #pole fell?

def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array") #initialize environment
    totals = []
    for ep in range(500):
        ep_rewards = 0
        obs, info = env.reset(seed=ep)
        for step in range(200):
            action = basic_policy(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_rewards += reward
            if done or truncated:
                break
        totals.append(ep_rewards)

    print(np.mean(totals), np.std(totals), min(totals), max(totals))
    return

if __name__ == "__main__":
    main()