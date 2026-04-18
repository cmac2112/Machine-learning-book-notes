import joblib
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from ClusterSimularity import ClusterSimularity  # Import your custom class

# Define the custom functions used in the preprocessor (copied from LinearRegressionExample.py)
# in a real world scenario it would make sense to have a library or publicly accessable "static" class that shares all of these methods
#that can be called for ease of use
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]
def main():
    #incorperate the trained model and throw some data at it

    model = joblib.load("models/california_housing_linreg.pkl")
    preprocessor = joblib.load("models/california_housing_preprocessing.pkl")
    new_data = pd.read_csv("datasets/housing/housingnew.csv")


    new_data_transformed = preprocessor.transform(new_data)

    prediction = model.predict(new_data_transformed)
    print(prediction)

if  __name__ == "__main__":
    main()