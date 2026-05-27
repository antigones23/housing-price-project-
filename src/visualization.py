import matplotlib.pyplot as plt

#Histogram of housing prices
def plot_histogram(df):
    df["median_house_value"].hist(bins=30)


    plt.title("Distribution of Median House Values")
    plt.xlabel("House Price")
    plt.ylabel("Frequency")

    plt.savefig("graphics/histogram.png")

 #Scatterplot of housing prices vs median income

def plot_income_vs_price(df): 

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["median_income"],
        df["median_house_value"],
        alpha = 0.2
    )

    plt.title("Median Income vs House Value")
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")

    plt.savefig("graphics/income_vs_price.png")


#Scatterplot of predictions vs actual values
def plot_predictions_vs_actual(y_test, predictions):
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, predictions, alpha=0.2)

    plt.xlabel("Actual House Value")

    plt.ylabel("Predicted House Value")

    plt.title("Predicted vs Actual House Values")

    plt.savefig("graphics/predictions_vs_actual.png")



