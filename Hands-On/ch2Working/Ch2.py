from sklearn.datasets import fetch_california_housing
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from ClusterSimularity import ClusterSimularity
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.linear_model import LinearRegression
#This program is ripped mostly from the book, once all programming is copied over here, i will perform my own version as if it were a real project

def clean_data(traindata, testdata):
    #median = data["total_bedrooms"].median()
    #data["total_bedrooms"].fillna(median, inplace=True)

    #instantiate an imputer (defining missing values as their columns median to not skew data as bad)
    imputer = SimpleImputer(strategy="median")
    
    housing_labels = traindata["median_house_value"].copy()
    
    traindata = traindata.drop("median_house_value", axis=1)

    #not all columns are int's
    #get only int's to caclulate the median

    housing_num = traindata.select_dtypes(include=[np.number])
    
    imputer.fit(housing_num)

    #translate ocean proximity to a numerical value
    housing_category = traindata[["ocean_proximity"]]
    
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
    
    #here we will write a log-transformer and apply it to the populatiion freature
    # we are doing this because of the heavy tail it has
    
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(traindata[["population"]])
    
    # here is an example of a custom transformer
    cluster_simu = ClusterSimularity(n_clusters=10, gamma=1, random_state=42)
    similarities = cluster_simu.fit_transform(traindata[["latitude", "longitude"]], sample_weight=traindata["population"])
    
    # plt.figure(figsize=(12, 8))
    # scatter = plt.scatter(data["longitude"], data["latitude"], 
    #                      c=similarities.max(axis=1),  # Color by max similarity
    #                      cmap="viridis", s=30, alpha=0.6)
    # plt.colorbar(scatter, label="Max Cluster Similarity")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.title("Cluster Similarities by Geographic Location")
    # plt.show()
    # plt.savefig("cluster_similarities.png")
    
    #print(similarities[:3].round(2))f
    
    
    # there are many transformation steps tha need be executed in the right order fortunately sklearn provides the Pipelineclass to help with such sequences
    
    # below is a small pipeline for numerical attributes which will first impute then scale the imput features
    
    
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    
    # or you can define it like this
    # this will take transformers as positional arguments and creates a pipeline using the names of the transformer's classes
    
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    
    #when we call fit on this, it called fit_transform() sequentially on all the transformers, passing the output of each call as the param to the next call until it reaches the final estimator
    
    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    #print(housing_num_prepared[:2].round(2))
    
    #this only accounts for numerical columns though, so it would be nice to have one that supports all types of columns
    #for this we can use a ColumnTransformer
    
    num_attributes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    cat_attributes = ["ocean_proximity"]
    
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    
    prepocessing = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])
    
    #naming all of the column names kinda sucked, thankfully we can use scikit-learn's make_column_selector
    
    better_preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object))
    )
        
    #now we can apply this ColumnTransformer to the housing data
    
    housing_prepared = better_preprocessing.fit_transform(traindata)
    
    #print(housing_prepared.shape)
    #print(better_preprocessing.get_feature_names_out())
    
    lin_reg = make_pipeline(better_preprocessing, LinearRegression())
    lin_reg.fit(traindata, housing_labels)
    
    housing_predictions = lin_reg.predict(testdata)
    print(housing_predictions[:5].round(2))
    print("------")
    print(testdata.iloc[:5].values)
    
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
#    print(len(train_set), len(test_set))



    print(test_set[:5])
    clean_data(train_set, test_set)
    return
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