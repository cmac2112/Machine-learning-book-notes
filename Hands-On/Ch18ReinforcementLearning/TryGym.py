import gymnasium as gym
import matplotlib.pyplot as plt
def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array") #initialize environment
    obs, info = env.reset(seed=42)
    print(obs) #print first observation [cartposition, cart velocity, pole angle, pole angular velocity]

    img = env.render()
    print(img.shape) #shape of the rendered frame

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(img)

    for _ in range(1000): #random action loop 1000steps
        if not plt.fignum_exists(fig.number): #stops early if window closes
            break
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) #takes random action (push left or right)
        im.set_data(env.render()) #draws the new frame
        plt.pause(0.02) #pauses for 0.2s
        if terminated or truncated: #resets environment when episode ends (either pole fell or cart left the screen, or hit 1000 step limit)
            obs, info = env.reset()

    env.close()
    return

if __name__ == "__main__":
    main()
