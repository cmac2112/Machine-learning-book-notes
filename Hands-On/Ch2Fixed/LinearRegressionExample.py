import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from ClusterSimularity import ClusterSimularity
import joblib

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))
def cat_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"),
                         FunctionTransformer(column_ratio, feature_names_out=ratio_name),
                         StandardScaler())

def process_and_clean_data(traindata, testdata):
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )
    
    cluster_simil = ClusterSimularity(n_clusters=10, gamma=1, random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    print("Columns going into preprocessing:", traindata.columns.tolist())
    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline(), make_column_selector(dtype_include=object))
        
        
    ],
                                      remainder=default_num_pipeline)
    
    return preprocessing.fit_transform(traindata), preprocessing.transform(testdata), preprocessing

def shuffle_and_split_data(data, test_ratio):
    
    #randomly shuffle the data
    shuffled_indices = np.random.permutation(len(data))

    #total amount of items to set aside = length of data * test_ratio parameter
    test_set_size = int(len(data) * test_ratio)

    #test indices = all after test_set_size index
    test_indices = shuffled_indices[:test_set_size]

    #training is same as above but all before the given index
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

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
    #this will serve as a fixed and concise version of the ch2 project
    
    #since we have already visualized our data, we will skip straight to processing
    
    housing = load_housing_data()
    
    #separate into our train sets and test sets
    
    train_set, test_set = shuffle_and_split_data(housing, 0.2)
    
    #get labels we want to predict 
    train_labels = train_set["median_house_value"]
    test_labels = test_set["median_house_value"]

    #value to compute final rmse
    y_test = test_set["median_house_value"].copy()

    #then drop them
    train_set = train_set.drop("median_house_value", axis=1)
    test_set = test_set.drop("median_house_value", axis=1)
    
    train_transformed, test_transformed, preprocessing = process_and_clean_data(train_set, test_set)
    
    lin_reg = LinearRegression()
    
    
    #           data features      predict this
    lin_reg.fit(train_transformed, train_labels)

    #we dropped the median_house_value from all of the data, we are trying to predict that based on the features of
    #the other data

    test_predictions = lin_reg.predict(test_transformed)
    print("predictions:", test_predictions[:10])
    print("actual values:", test_labels[:10])
        # Compare train vs test RMSE to check for overfitting
    train_predictions = lin_reg.predict(train_transformed)
    train_rmse = root_mean_squared_error(train_labels, train_predictions)
    test_rmse = root_mean_squared_error(test_labels, test_predictions)
    
    print(f"Train RMSE: {train_rmse:,.2f}")
    print(f"Test RMSE:  {test_rmse:,.2f}")

    #average rmse of 68k or ~27% ish
    #not bad not great, pretty comparable to what was being predicted before

    preprocessing.fit(train_set)
    joblib.dump(lin_reg, "models/california_housing_linreg.pkl")
    joblib.dump(preprocessing, "models/california_housing_preprocessing.pkl")
if __name__ == "__main__":
    main()