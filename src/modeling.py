from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


#Simple Linear Regression 
def simple_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    model = LinearRegression()
    model.fit(X_train, y_train)


    predictions = model.predict(X_test)
    print("\nSimple Linear Regression Predictions:")
    print(predictions[:5])

    print("\nActual values:")
    print(y_test[:5])

#Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, predictions)
    print("\nMean Absolute Error:")
    print(mae)

#Examine coefficients
    print("\nSimple Linear Regression coefficients:")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: {coef}")

#Calculate R-squared
    r2 = r2_score(y_test, predictions)
    print("\nR-squared:")
    print(r2)
    return model, predictions

#Random Forest Regressor 

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return predictions, mae, r2

# Gradient Boosting Regressor
def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


