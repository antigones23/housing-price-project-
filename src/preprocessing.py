   
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():

#Fill missing values with median
    df = pd.read_csv("data/housing.csv")

    df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
    )

#Convert categorical variable to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=["ocean_proximity"])

# Separate features from target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

# Split into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

# Standardize features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_names = X.columns
    return X_train, X_test, y_train, y_test, feature_names