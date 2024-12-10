# @Author  : Sixing Wu
# @Time    : 2024/12/3
# @Mac, Python 3.11
# @Function: Using sentiment scores, with the previous day's data (optional) of open, close, high,
# low, market cap, and volume to predict the next day's open price by random forest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt

def random_forest_prediction(data, feature):
    # Load the CSV file into a DataFrame
    data_df = pd.read_csv(data)
    data_df = data_df.sort_values(by='Start', ascending=True).reset_index(drop=True)

    # Convert the 'Start' column to a datetime format
    data_df['Start'] = pd.to_datetime(data_df['Start']).dt.date

    # Generate daily info with sentiment scores and price data
    daily_info = data_df.copy()

    # Shift the features to get the previous day's data for each row
    feature_with_sentiment = feature
    variables = []

    for x in feature_with_sentiment:
        variables.append(f'prev_{x}')
        daily_info[f'prev_{x}'] = daily_info[x].shift(1)

    # Drop rows with NaN values (which will appear in the first row after shifting)
    daily_info = daily_info.dropna()

    # Define X (features) and y (target)
    X = daily_info[variables]
    y = daily_info['Close']  # The current day's close price
    dates = daily_info['Start']  # Store the dates for plotting

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the error for evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    r2 = r2_score(y_test, y_pred)
    print(f"R-Squared: {r2}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Print and plot predictions vs actual values
    results = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_pred}).sort_values(by='Date')
    print(results)

    plt.figure(figsize=(10, 6))
    plt.plot(results['Date'], results['Actual'].values, label='Actual', color='blue', alpha=0.6)
    plt.plot(results['Date'], results['Predicted'].values, label='Predicted', color='red', alpha=0.6)
    plt.title('Actual vs Predicted Open Price')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('RF_without_sentiment.png', format='png', dpi=300)
    plt.show()

# Run the function
data = '../data/Bitcoin_Price/price_sentiment_data.csv'
selected_feature = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
random_forest_prediction(data, selected_feature)
