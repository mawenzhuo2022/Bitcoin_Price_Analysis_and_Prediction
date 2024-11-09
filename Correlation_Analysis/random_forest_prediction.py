# @Author  : Sixing Wu
# @Time    : 2024/11/9
# @Mac, Python 3.11
# @Function: Using sentiment score and various features is price data to explore the correlation,
# output a scatter plot of the sentiment score and selected features

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def random_forest_prediction(tweet_data, price_data, feature):
    # Load the CSV file into a DataFrame
    tweet_df = pd.read_csv(tweet_data)
    price_df = pd.read_csv(price_data)
    price_df = price_df.sort_values(by='Start', ascending=True).reset_index(drop=True)

    # Convert the 'date' column to a datetime format if it's not already
    tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.date

    # Generate daily info with sentiment scores and price data
    daily_info = tweet_df.groupby('date')['sentiment_score'].sum().reset_index()
    daily_info[feature] = price_df[feature]

    # Shift the features to get the previous day's data for each row
    feature.append('sentiment_score')
    variables = []
    for x in feature:
        variables.append(f'prev_{x}')
        daily_info[f'prev_{x}'] = daily_info[x].shift(1)

    # Drop rows with NaN values (which will appear in the first row after shifting)
    daily_info = daily_info.dropna()

    # Define X (features) and y (target)
    X = daily_info[variables]
    y = daily_info['Open']  # The current day's open price

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the mean squared error for evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Optionally, print or plot predictions vs actual values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(results)

tweet_data = "../data/Sentiment_Analysis/sentiment_analysis.csv"
price_data = '../data/Bitcoin_Price/bitcoin_2021-02-08_2021-02-13.csv'
selected_feature = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
random_forest_prediction(tweet_data, price_data, selected_feature)



