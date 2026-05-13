from matplotlib.sphinxext import plot_directive
from sklearn.datasets import fetch_openml
import numpy as np
from ImageHelpers import ImageHelpers
import matplotlib.pyplot as plt
from DataProcessors import DataProcessor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve

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

    #lets take a look at the precision and recall scores
    print(precision_score(y_train_fives, y_train_pred))

    print(recall_score(y_train_fives, y_train_pred))

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
    
    # precision/recall tradeoff ---------------------
    
    # y_scores = sgd_clf.decision_function([5])
    # print(y_scores)
    # threshold = 0
    # y_some_digit_pred = (y_scores > threshold)
    # print(y_some_digit_pred)
    
    # using cross val predict to decide what threshold to use -----------------------------------------------------
    
    y_scores = cross_val_predict(sgd_clf, instance.train_x, y_train_fives, cv=3, method="decision_function")
    
    precisions, recalls, thresholds = precision_recall_curve(y_train_fives, y_scores)
    
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(thresholds, 0, 1.0, "k", "dotted", label="threshold")
    plt.savefig(f"threshold recall figure.png")
    
    
    
    # also can plot precision against the recall directly
    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    plt.savefig(f"precision-recall-curve.png")
    
    #lets find the most optimal threshold
    
    #if we aim for 90% precision, we can use the first plot to find the threshold but its not precisce. Alternatively we can saerch for the lowest threshold that gives us 90% precision.
    # we can use the NumPy array's argmax() method, returns the first index of the maximum value
    
    index_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = thresholds[index_for_90_precision]

    #now we can find the values    
    y_train_pred_90 = (y_scores >= threshold_for_90_precision)
    
    print(f"precision score {precision_score(y_train_fives, y_train_pred_90)}")
    
    
    recall_at_90_precision = recall_score(y_train_fives, y_train_pred_90)
    print(f" recall at 90 - {recall_at_90_precision}")    
    
    

if __name__ == "__main__":
    main()