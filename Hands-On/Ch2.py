from sklearn.datasets import fetch_california_housing
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
#shuffle the data and set aside test data
#each time this is called it will return a different set of data
def shuffle_and_split_data(data, test_ratio):
    
    #randomly shuffle the data
    shuffled_indicies = np.random.permutation(len(data))

    #total amount of items to set aside = length of data * test_ratio parameter
    test_set_size = int(len(data) * test_ratio)

    #test indicies = all after test_set_size index
    test_indicies = shuffled_indicies[:test_set_size]

    #training is same as above but all before the given index
    train_indicies = shuffled_indicies[test_set_size:]

    return data.iloc[train_indicies], data.iloc[test_indicies]

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")

    if not tarball_path.is_file():

        #make a directory to store the csv if we dont have one
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"

        urllib.request.urlretrieve(url, tarball_path)

        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

    #return the csv contents
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def main():
    california_housing = fetch_california_housing(as_frame=True)
    housing = load_housing_data()

    #plot housing just for visualization

    # housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
    # plt.show()

    # plt.savefig("housingscatter.png")

    # housing.hist(bins=50, figsize=(12,8))
    # plt.show()
    # plt.savefig("housingHistograms.png")
    
    
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income Category")
    plt.ylabel("Number of Districts")
    plt.show()
    plt.savefig("incomeCategory.png")
    #shuffle data and return train and test sets
    train_set, test_set = shuffle_and_split_data(housing, 0.2)
    print(len(train_set), len(test_set))


if __name__ == "__main__":
    main()