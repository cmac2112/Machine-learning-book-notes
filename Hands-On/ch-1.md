## preface
this book is seperated into 2 parts
1. The fundamentals of machine learning
- What machine learning is, what problems it tries to solve, and the main categories and fundamental concepts of its systems
- The steps in a typical machine learning project
- learning by fitting a model to data
- optimize a cost function
- handling, cleaning, and preparing data
- selecting and engineering features
- selecting a modek and tuning hyper parameters using cross-validation
- The challenges of machine learning, in particular underfitting and overfitting. (bias/variable trade-off)
- most common learning algorithms: linear and polynomial regression, logistic regression, k-nearest neighbors, support vector machines, decision trees, random forests, and ensemble methods
- Reducing the dimensionality of the training data to fight the curse of dimensionality
- Other unsupervised learning techniques, including clustering, density estimation, and anomaly detection

Part 2:
- what neural nets are and what they're good for
- Building and training neural nets using TensorFlow and Keras
- the most important neural net architectures: feedfoward neural nets for tabular data, convolutional nets for computer vision, recurrent nets and long short-term memory (LSTM) nets for sequent processing, encoder-decoder and transformers for natural lanuage processing, autoencoders, generatative adversarial networks, and diffusion models for generative learning
- Techniques for training deep neural nets
- How to build an agent (ex: a bot in a game) that can learn good strategies through trial and error, using reinforcement learning
- Loading and preprocessing large amounts of data efficiently
- Training and deploying TensorFlow models at scale



# chapter 1
What is machine learning:

Machine learning is the science and art of programming computers so they can learn from data.

Machine learning shines for problems that are either too complex for traditional approaches or have no known algorithm.
example: Speech recognition

Machine learning can help humans learn by inspeting what the models have learned themselves. Data mining is the concept of digging into large amounts of data to discover hidden patterns and machine learning excels at it.

To summarize machine learnin is great for:
- Problems for which existing solutions require a lot of fine-tuning or a long list of rules.
- Complex problems for which using a traditional approach yields no good solution
- Fluctuating environments (a machine learning system can easily be retrained on new data, always keeping it up to date)
- Getting insights about complex problems and large amounts of data

## Types of machine learning systems
There are manyt different types of machine learning systems that it is useful to classify them in broad categories, based on the following criteria
- How they are supervised during training
- Whether or not they can learn incrementally on the fly (online vs batch learning)
- Whether they work by simply comparing new data points to known data points or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based vs model-based learning)

A state of the art spam filter may learn on the fly using a deep neural network model trained using human-provided examples of spam and ham (actual emails); this makes it an online, model-based, supervised learning system

### Training supervision
ML systems can be classified according to the amount and type of supervision they get during training.
The main ones are :
- supervised
- unsupervised
- self-supervised
- semi-supervised
- reinforcement learning

Supervised learning
in supervised learning, the trainint set you feed to the algorithm includes the desired solutions, called lables

a typical supervised learning task is classification. The spam filter is a good exampleof this: it is trained with many example emails along with their class (spam or ham), and itm ust learn how to classify new emails.

Another typical task is to predict a target numeric value, such as the price of a car, given a set of features (milage, age, brand, etc). This sort of task is called regression. To train the system you need to give it many examples of cars, including both their features and their targets.

Some regression models can be used for classification as well, and vise versa. For example, logistic regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (Ex 20% chance of being spam)