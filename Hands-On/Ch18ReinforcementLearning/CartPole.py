import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis]) #call the model giving it a single observation. Outputs the probability of going left

        #same random float btwn 0 - 1, check if its > left_proba
        action = (tf.random.uniform([1,1]) > left_proba)

        #define the target probability of going left: its 1 minus the action (Cast to a float)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        #compute the loss
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))

    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, truncated, info = env.step(int(action[0, 0]))
    return obs, reward, done, truncated, grads

def play_mulitple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, truncated, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done or truncated:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards,discount_factor) 
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

def render_policy(model, n_episodes=3, n_max_steps=500):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset()

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(env.render())
    for episode in range(n_episodes):
        obs, info = env.reset()
        for step in range(n_max_steps):
            if not plt.fignum_exists(fig.number):
                env.close()
                return
            left_proba = model(obs[np.newaxis])[0, 0].numpy()
            #act greedily when watching: go left if the model thinks left is more likely
            action = 0 if left_proba > 0.5 else 1
            obs, reward, done, truncated, info = env.step(action)
            im.set_data(env.render())
            ax.set_title(f"Episode {episode + 1}/{n_episodes} | step {step + 1}")
            plt.pause(0.02)
            if done or truncated:
                break
        print(f"Watched episode {episode + 1}: {step + 1} steps")
    env.close()

def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array") #initialize environment
    obs, info = env.reset(seed=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    loss_fn = tf.keras.losses.binary_crossentropy
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)

    n_iterations = 100
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_factor = 0.95
    for iteration in range(n_iterations):
        all_rewards, all_grads = play_mulitple_episodes(env, n_episodes_per_update, n_max_steps, model,
                                                        loss_fn)
        #episode length == total reward, since cartpole gives +1 per step
        episode_lengths = [len(rewards) for rewards in all_rewards]
        print(f"Iteration {iteration + 1:3d}/{n_iterations} | "
              f"mean: {np.mean(episode_lengths):6.1f} | "
              f"min: {min(episode_lengths):3d} | max: {max(episode_lengths):3d}")
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
        all_mean_grads =[]
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean([final_reward * all_grads[episode_index][step][var_index]
                                         for episode_index, final_rewards in enumerate(all_final_rewards)
                                         for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    model.save("cartpole.keras")
    render_policy(model)




    return

if __name__ == "__main__":
    main()