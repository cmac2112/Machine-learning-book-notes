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

### Supervised learning
in supervised learning, the training set you feed to the algorithm includes the desired solutions, called lables

a typical supervised learning task is classification. The spam filter is a good example of this: it is trained with many example emails along with their class (spam or ham), and it must learn how to classify new emails.

Another typical task is to predict a target numeric value, such as the price of a car, given a set of features (milage, age, brand, etc). This sort of task is called regression. To train the system you need to give it many examples of cars, including both their features and their targets.

Some regression models can be used for classification as well, and vise versa. For example, logistic regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (Ex 20% chance of being spam)

### unsupervised learning
in unsupervised learning, the training data is unlabeled. The system tries to learn without a teacher

Example: You have a lot of data about your blog's visitors. You may want to run a <b>Clustering</b> algorithmto try and detect groups of similar visitors. At no point do you tell the algorithm which group a visitor belongs to: it finds those connections without your help.

It might notice that 40% of your visitors are teenages who love comic books and generally read your blog after school, while 20% are adults who enjoy sci-fi and who visit during the weekends.

Using a <b>Hierarchical clustering</b> algorithm could further subdivide each group into smaller groups

<b>Visualization</b> algorithms are examples of unsupervised learning: you feed them a lot of complex and unlabled data, and they output a 2D or 3D represenataion of your data to be plotted.

<b>Dimensionality reduction</b> is the act of simplifying data without losing too much information. Example: you can merge several correlated features into one. A cars milage is strongly correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car's wear and tear. also called <b>Feature Extraction</b>

 <b>Anonmaly Detection</b> another important task of unsupervised learning. Such as detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or removing outliers from a dataset before feeding it to another learning algorithm.

 <b>Association Rule Learning</b> where the goal is to dig into large amounts of data and discover interesting relations between attributes. Ex: you own a supermarket and running association on items sold may reveal that people who purchace bbq sauce and potato chips also tend to buy steak, you may want to put those items next to each other in the store


 ### Semi-Supervised learning
 labelling data is time-consuming and costly, you will often have plenty of unlabed instances. Some algorithms can deal with data thats partially labeled, this is called <b>Semi-supervised learning</b>

 Google photos uses this, when you upload all of your family photos it recognizes the same person appears in photos 1, 4, 6 while person B appears in 2, 5, 7. This is the clustering part, now all the algorithm needs is for you to label one person in a photo and it can name them in all the others.

 Most semi-supervised learning algorithms are combinations of unsupervised and supervised. Clustering (unsupervised) can be used to group similar instances together, then labeled. Then after that its possible to use any supervised learning algorithm.

 ### Self-Supervised Learning
 Involves generating a fully labeled dataset from a fully unlabled one. 

 For example you have a large dataset of unlabled imagess, you can randomly mask parts of each image, then train the model to recover the original image. During training the masked images are used as the inputs to the model, and the original images are used as the labels.

 A self-supervised learning model is not the final goal

 Example: A pet classifaction model. Given any picuture of a pet it will tell tell you what species it belongs to. If you have a large dataset of unlabled photos of pets you can start by training an image repairing model using self-supervised learning. Once it is performing well, it should be able to distinguish different pet species. when it is reparing images of a cat it should know not to fill it with a dogs face.

 ### Reinforcement learning
 differs from all other previous types. The agent can observe the environment, select and perform actions and get rewards in return. It must learn by itself what is the best strategy (called policy) to get the most reward overtime.

 simple example steps of an agent

 1. Observe
 2. Select action using policy
 3. Action
 4. Get reward or penalty
 5. Update Policy
 6. Iterate until optimal policy is found



 ### Batch vs online learning
 used to classfiy learning systems that can learn incrementally from a stream of incoming data

 <b>Batch Learning</b> the system is incapable of learning incrementally. This is typically done offline and then launched into production. Systems performance typically degrades overtime due to the evolving nature of behavior and data in the world.

 ### Online learning
 you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap so the system can learn about new data on the fly as it arrives.

 Model is launched and keeps learning as new data comes in