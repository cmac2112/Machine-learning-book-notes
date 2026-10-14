import gymnasium as gym
import matplotlib.pyplot as plt
def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print(obs)

    img = env.render()
    print(img.shape)

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(img)
    for _ in range(1000):
        if not plt.fignum_exists(fig.number):
            break
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        im.set_data(env.render())
        plt.pause(0.02)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    return

if __name__ == "__main__":
    main()
