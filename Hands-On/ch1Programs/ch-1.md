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

 ### Batch learning
 in batch learning the system in incapable of learning incrementally: it must be trained using all the available data. 

 Takes a lot of time and hence typically done offline. 

 Because of its nature of being taught only once and deployed, its performance decays slowly over time. This is called model rot or data drift. The solution is to regularly retrain the model on up to date data.

 ### Online learning
 you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap so the system can learn about new data on the fly as it arrives.

 Model is launched and keeps learning as new data comes in

 online learning is useful for systems that need to adapt to change extremely rapidly

 additionally online learning algorithms can be used to train models on huge datasets that cannot fit in one machines main memory (aka out ofg core learning)

 Another important parameter of online learning systems is how fast they should adapt to chaning data, also known as the <b>Learning Rate</b>
 
### Instance-Baed vs Model-Based Learning
one more way to categorize machine learning systems is by how they generalize. This means that given a number of training examples, the system needs to be able to make good predictions for (generalize to) examples it has never seen before.

There are two main approaches to generalization: Instance-based learning and model-based learning

### Instance-Based learning
Most trivial form of learning is to learn by heart.

Ex: spam filter filters out emails that are identical to emails that have already been flagged by users. Not worst solution but not the best

Instead your spam filter could flag emails that are very similar to known spam emails. This requires a <b>Measure of simularity</b> between emails.

IE: system could count the number of words they have in common etc.

This is called instance based learning: the system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to learned examples.

### model based learning
Another way to generalize from a set of examples is to build a model of examples then use that model to make predictions


## Linear regression example
/ch1Programs/1-1linearRegressionModel.py

Things to note more about linear regression

How to determine a loss function. A loss function is most commonly measured using <b>Mean Squared Error</b>

Equation:
    1. Take the difference between actual and predicted values
    2. Square it so negatives dont cancel positives
    3. averave over all data points

### K nearest neighbors
if we used an instance based learning algorithm instead and used the average of the 3 nearest neighbors

`model = KNeighborsRegressor(n_neighbors=3)`
we get 6.33 which is very close to the original model based prediction

## Data quality
most of the time spent making models is cleaning up data and figuring out what parameters to use.

Garbage in garbage out your system is only as capable as the data your provide it

### Overfitting
means that the model performs well on the trainting data but does not generalize well

Overfitting happens when the model is too complex relative to the amount and noiseness of the training data. Possible solutions are
1. Simply the model by selecting one with fewer parameters (ex a linear model rather than a hiigh degree polynomial) by reduing the number of attributes in the training data, or by constraining the model
2. Gather more training data
3. reduce the noise in training data

Constraining a model to make it simpler is called <b>Regularization</b>. The model we defined before only has two degrees of freedom <b>Theta0 (height)</b> and <b>Theta1(slope)</b>

The amount of regularization to apply during learning is called a <b>HyperParameter</b>. It is not affected by the learning algorithm itself. If you set this to a very large value your model will become quite flat and will not overfit.


### underfitting
Underfittin is the opposite of overfitting, it occurs when your model is too simple to learn the underlying structure of the data.

How to fix:
1.  select a more powerful model with more parameters
2. feed better features to the learning algorithm
3. reduce the constraints on the model

# Chapter quiz

1. How would you define machine learning :
        Machine learning is the action of predicting future values based on the values of the past. Rather than programatically, these systems can be used to attempt to solve problems that dont have dedicated solutions
2. Can you name four types of applications where it shines
        When problems have no algorithmic solution, replace long lists of hand-tuned rules, build systems that adapt to fluctuating environments, and finally help humans learn

3. what is a labeled training set
    Labeled training set is a training set that basically has the answer tied to it. Someone has defined that piece of data as what it is, such as if a picture of a flower, it has been dedicated as a flower.
4. What are the two most common supervised tasks?
    Classification and regression
5. Can you name four uncommon supervised tasks
    Visualization, clustering, Anamoly detection

6. What type of alogriithm would you use to allow a robot to walk in various unknown terriains?
        Reinforcement learning
7. What type of algorithm would you use to segment your customers into multiple groups?
        - unsupervised clustering algorithm
8. Would you frame the problem of spam detection as a supevised learning problem or an unsupervised learning problem?
        A supervised learning problem, users can mark items as spam which can be used as training data on our model, the model will have to be retrained frequently as spam mail evolves however. We should not use unsupervised here because the model could pick up on things that are not spam and block important emails from reaching users.
9. What is an online learning system
    An online learning system is a system that can learn on the fly with new data it is provided with
10. What is out of core learning
    Not all training data can fit onto a machine, some data must live elsewhere and be incrementially trained on

11. What type of algorithm relies on a similarity measure to make predictions?
    Instance Based learining

12. What is the difference between a model parameter and a model hyperparameter
    A model parameter is what the model uses to learn against, a hyper parameter is independent of that but can still affect the output of the model

13. What do model based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions
    The model based algorithm searches for similarity between models. 

14. What are the four main challenges of machine learning
    Data insufficency, non-representive training data, poor data quality, irrelevant features, overfitting/underfitting

15. If your model performs great on the training data buut generalizes poorly to new instances, what is happening? Can you name three possible solutions
    What is happening is the model is overfitting the data. There are 3 things you can do. 1 Simplify the model. 2 Gather more training data. 3 Reduce the noise in the training data

16. What is a test set, and why would you want to use it?
    A test set of data is data that you set aside that the model will not train on. Instead you will feed the model this data after it has been trained to test its accuracy

17. What is the purpose of a validation set?
    The purpose of the validation set is to test groups of models on that perform the best on the validation set, then move that model on to the main test set. 
