## Reinforcement Learning

Reinforcement Learning (RL) has produced many different applications over the years, paticularly in games and in machine control.

Deepmind demonstrated a system that could play Atari games outperforming most humans only using raw pixels as inputs without any prior knowledge of the game.

### Learning to Optimize rewards
In reinforcement learning, a software agent makes observations and takes actions within an environment, in return it recieves rewards from the environment. Its main objective is to learn to act in a way that will maximize its expected rewards over time.

a. Agent can control a robot where the environment is the real world. It observes through sensors, and its actions consist of sending signals to activate motors. It may get rewards when moving to a destination 
b. Aget can be controlling Pacman, in this case the environment is a simulation of the Atari game., the actions are the possible joystick positions, the observations are screenshots, and the rewards are just the game points
c. the agent can play board games such as go and recieve rewards when it makes good moves or wins
d. The agent oes not have to control a physically moving thing. It could be a thermostat getting positive rewards whenever it is close to the target temperature and saves energy. And negative rewards when humans need to tweak the temperature so it learns to anticipate human needs.
e. Agent can observe stock market prices and decide how much to buy or sell every second. Rewards are monetary gain or losses

### Policy Search
The algorithm a software agent uses to determine its actions is called its policy. 

The policy could be a neural network taking observations as inputs and outputting the action to take.

Policies can be any algorithm, it does not have to be deterministic.

Some cases you dont have to observe the envionment. For example if a robot vacuumc cleaner whose reward is the amount of dust it picks up every 30 minutes. Its policy could be to move forward with some probability p every second, or randomly rotate left or right with probability 1 - p. The rotation angle would be a random angle between -r and +r. since this policy involves some randomness, it is called stochastic policy. The robot will have a erratic trajectory.

How would you train such a robot? There are only two policy parameters you can tweak. Probability p and the angle range r. 

You could:
a. try as many different values for these parameters and pick the combo that performs the best - this is policy search with a brute force approach
b. use a genetic algorithm - randomly create a first generation of 100 policies and try them out, then kill the wirst 80 policies and make the 20 survivors produce 4 offspring each. An offspring is a copy of its parent plus some random variation. The surviving policies plus their offspring together constitute the second generation. You can continue to iterate through generations this way until you find a good policy
c. other optimization techniques by evaluating the gradients of the rewards with regard to the policy parameters, then tweaking these parameters by following the gradients toward higher rewards - policy gradient - you could slightly increase p and evaluate whether doing so increases the amount of dust picked up by the robot in 30 minutes; if it does then increase p some more, or else reduce p. We will need to implement a popular PG algorithm using TensorFlow.


## Openai gym
a challenge of reinforcement learning is having a working environment. Training is hard and slow in the real world, that is why we need simulated environments.

For example you might yse PyBullet or MuJoCo for 3D physics simulation

OpenAI gym is a toolkit that provides a wide variety of simulated environments that you can use to train agents.

in TryGym.Py i have setup a cart push environment. The programs comments explains what it does.

`step()` - executes a desired action and returns five values:
1. `obs` new observation. The crt is moving toward the right obs[1] > 0. The pole is still tilted toward the right obs[2] > 0 but its agular velocity is now negative obs[3] < 0
2. `reward` in this environment you get a reward of 1 at every step no matter what you do
3. `done` value will be true when the episode is over. Either the pole tilts too much or goes off the screen, or after x amount of steps (you won)
4. `truncated` value will be true when an episode is interrupted early by an envionment wrapper that imposes a maximum number of steps per episode.
5. `info` environment specific dictionary provides extra information

SimplePolicy.py displays a simple if else policy with the environment

## Neural network policies
This neural network policy will take an observation as input and it will output the action to be executed, just like the poliucy we hardcoded in SimplePolicy.Py
It will estimate a probability for each action and then we will seect an action randomly, according to the estimated probability. There are just two possible actions left or right

so we only need one output neuron. It will ouput the probability p of action 0 (left) and the probability of action 1(right) will be (1 - p)

We are choosing a random action based on the probabiilities given by the network rather than picking the action with the highest score, this allows the agent to find the right balance between exploring new actions and exploiting the actions that are known to work well.

This exploration/exploitation dilemma is central in reinforcement learning

In this particular environment the past actions and observations can be safely ignored, if there were some hidden state, then we would need to consider past actiions and observations as well.

## Credit assignment problem

If we knew what the best action was at each step, we could train the neural network as usual, by minimizing the cross entrpy between the estimated probability of distribution and the target probability distribution. It would be regular supervised learning.

However the only guidance the agent gets in reinforcement learnin is rewards, rewards are typically sparsed and delayed.

If an agent manages to balance the pole for 100 steps how can iit know which of the 100 actions it took were good, and which were bad? All it knows is that the pole fell after the last action but the last action is not entirely responsible. This is called the `credit assignemnt problem` - when the agent gets a reward, it is hard for it to know which actions should get credited for it.

A common strategy is to evaluate an action based on the sum of all the rewards that come after it, usually applying a discount factor, y(gamma) at each step.

The sum of discounted rewards is called the `actions return`

Consider the example below:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```
If an agent decides to go right three times in a row and gets +10 reward after the first step, 0 after the second, and finally -50 after the third. Then assuming we are using a discount factory y = 0.8, the first action will have a return of 

`10 + y * 0 + y^2 * (-50) = -22`

if the dicount factor is close to 0, then future rewards wont count for much compared to immediate rewards, conversely if the discount factor is close to 1, then rewards far into the future will count almost as much as immediate rewards. 

Typical discount factors vary from 0.9 to 0.99 with a discount factor of 0.95, rewards 13 steps into the future count roughly for half as much as immediate rewards (since 0.95^13 = 0.5)

In the cart pole environment, actions have fairly short term effects, so choosing a discount factor of 0.95 seems reasonable

___

A good action may be followed by several bad actions that cuse the pole to fall quickly, resulting in the good action getting a low return. However if we play the game enough times, on average good actions will get a higher return than bad ones. We want toestimate how much better or worse an action is, compared to the other possible actions, on average. This is called `action advantage` 
For this we must run many episodes and normalize all the action returns, by subtracting the mean and dividing by the standard deviation. After that we can resonably assume that actions with a negative advantage were bad while actions with a positive were good.

Now we can train our first agent with policy gradients

## Policy Gradients
PG algorithms optimiize the parameters of a policy by following the gradients toward higher rewards. 