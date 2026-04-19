from matplotlib.sphinxext import plot_directive
from sklearn.datasets import fetch_openml
import numpy as np
from ImageHelpers import ImageHelpers
import matplotlib.pyplot as plt
from DataProcessors import DataProcessor
from sklearn.linear_model import SGDClassifier
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

    #go through vals, save some true images

    #get index of true vales

    all_true_indices = np.where(val)[0]

    first_three = all_true_indices[:3]


    var = 0
    for i in first_three:
        #plot the items
        five = ImageHelpers(instance.test_x[i])
        five.plot_digit()
        plt.savefig(f"classified 5 {var}.png")
        var += 1

    return

if __name__ == "__main__":
    main()