import pandas as pd

from src.visualization import plot_histogram, plot_income_vs_price, plot_predictions_vs_actual

from src.preprocessing import preprocess_data

from src.modeling import simple_linear_regression

from src.modeling import evaluate_model, train_random_forest

from src.modeling import train_gradient_boosting

def main():

    df = pd.read_csv("data/housing.csv")
    print(df.head())


# Histogram of housing prices
      
    plot_histogram(df)
   
# Scatterplot of housing prices vs median income
    
    plot_income_vs_price(df)

# Feature selection and preprocessing

    X_train, X_test, y_train, y_test, feature_names = preprocess_data()

# Simple Linear Regression

    model, predictions = simple_linear_regression(X_train, X_test, y_train, y_test, feature_names)

# Simple Linear Regression Predictions vs Actual Values

    plot_predictions_vs_actual(y_test, predictions)

# Random Forest Regressor 

    random_forest_model = train_random_forest(X_train, y_train)

    rf_predictions, rf_mae, rf_r2 = evaluate_model(
        random_forest_model,
        X_test,
        y_test
    )

    print("\nRandom Forest Results")
    print("MAE:", rf_mae)
    print("R^2:", rf_r2) 

# Gradient Boosting Regressor

    gb_model = train_gradient_boosting(X_train, y_train)

    gb_predictions, gb_mae, gb_r2 = evaluate_model(
        gb_model,
        X_test,
        y_test
    )

    print("\nGradient Boosting")
    print("MAE:", gb_mae)
    print("R^2:", gb_r2)

if __name__ == "__main__":
    main()


