from matplotlib.sphinxext import plot_directive
from sklearn.datasets import fetch_openml
import numpy as np
from ImageHelpers import ImageHelpers
import matplotlib.pyplot as plt
from DataProcessors import DataProcessor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

def main():
    mnist = fetch_openml('mnist_784', as_frame=False)

    x, y = mnist.data, mnist.target

    print(f"x-shape{x.shape} : y-shape {y.shape}")

    print(f"x: {x}")

    # x_train, x_test = x[:60000], x[60000:]

    instance = DataProcessor(x, y).return_split_data()

    #make life easier for now we are only trainig and looking for number 5s

    y_train_fives, y_test_fives = instance.return_values_where('5')


    #ImageHelpers(y_test_fives).plot_array()

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(instance.train_x, y_train_fives)

    val = sgd_clf.predict(instance.test_x)


    #determine the model's performance here with cross validation with 3 k folds

    score = cross_val_score(sgd_clf, instance.train_x, instance.train_y, cv=3, scoring='accuracy')
    print(f"Cross Validation Score: {score}")

    #dummy classifier classifies every single image in the most frequent class, which in this case is the negative class (not 5's)
    dummy_clf = DummyClassifier()
    dummy_clf.fit(instance.train_x, y_train_fives)
    print(any(dummy_clf.predict(instance.test_x))) #should print false: no 5's detected


    #should print around 90% since 10% of the images are 5's and the other 90% are not, so if you were to guess if an image is not 5, you would be
    # right around 90% of the time
    print(cross_val_score(dummy_clf, instance.train_x, y_train_fives, cv=5, scoring='accuracy'))


    #using a confusion matrix and cross_val_predict
    y_train_pred = cross_val_predict(sgd_clf, instance.train_x, y_train_fives, cv=3)

    #just like the cross_val_score() function, this one performs k-fold cross-validation, but instead of returning the evaluation scores
    #it returns the predictions made on each test fold.

    cm = confusion_matrix(y_train_fives, y_train_pred)
    print(cm) #each row represents an actual class, while each column represents a predicted class.




    #go through vals, save some true images

    #get index of true vales

    # all_true_indices = np.where(val)[0]
    #
    # first_ten = all_true_indices[:10]
    #
    # for i in range(len(first_ten)):
    #     # plot the items
    #     index = first_ten[i]
    #     five = ImageHelpers(instance.test_x[index])
    #     five.plot_digit()
    #     plt.savefig(f"classified 5 {index}.png")
    #
    # return

if __name__ == "__main__":
    main()