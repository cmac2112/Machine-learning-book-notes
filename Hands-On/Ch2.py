from sklearn.datasets import fetch_california_housing
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
def clean_data(data):
    #median = data["total_bedrooms"].median()
    #data["total_bedrooms"].fillna(median, inplace=True)

    #instantiate an imputer (defining missing values as their columns median to not skew data as bad)
    imputer = SimpleImputer(strategy="median")

    #not all columns are int's
    #get only int's to caclulate the median

    housing_num = data.select_dtypes(include=[np.number])
    
    imputer.fit(housing_num)

    #translate ocean proximity to a numerical value
    housing_category = data[["ocean_proximity"]]
    
    ordinal_encoder = OrdinalEncoder()
    
    housing_category_encoded = ordinal_encoder.fit_transform(housing_category)

    #hot encoder to create binary attributes per category

    cat_encoder = OneHotEncoder()
    #returns a sparce matrix
    housing_category_hot = cat_encoder.fit_transform(housing_category)
    
    #next we need to standardize our data, such as the median income
    # we can minmax here aka normalization, values are shifted and rescaled so that they end up ranging from 0 to 1.
    # This is done by subtracting the min value and dividing by the difference between the min and the max
    
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    
    #or use standardization here which is different, first it sutracts the mean, then it divides the result by the standard deviation, so standardized values have a standard deviation equal to 1
    #standardiization does not restrict values to a certain range, however it is much less affected by outliers.\
    
    std_scalar = StandardScaler()
    housing_num_std_scalar = std_scalar.fit_transform(housing_num)
    
    #continuing data transformation next, data needs to be formatted as close as possible to a bell curve
    
    return



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


    #california_housing = fetch_california_housing(as_frame=True)
    housing = load_housing_data()
    
    clean_data(housing)
    return

    #plot housing just for visualization

    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population", c="median_house_value", colormap="jet", colorbar=True,
                 legend=True, sharex=False, figsize=(10,7))
    plt.show()

    plt.savefig("housingscatter.png")

    # housing.hist(bins=50, figsize=(12,8))
    #plt.show()
    #plt.savefig("housingHistograms.png")
    
    
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


    #corr_matrix = housing.corr()

    #vals = corr_matrix["median_house_value"].sort_values(ascending=False)
    
    #scatter matrix to see correlation between values

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
    plt.savefig("scatter_correlations")

    #lets view the correlation between the price and the bedroom ratio (num of beds / num of total rooms)

    housing["bedroom_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]

    housing.plot(kind="scatter", x="median_house_value", y="bedroom_ratio", grid=True)
    plt.show()

    plt.savefig("bedroom_ratio")





if __name__ == "__main__":
    main()