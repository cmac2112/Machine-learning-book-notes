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