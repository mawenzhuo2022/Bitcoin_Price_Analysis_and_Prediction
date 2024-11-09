# @Author  : Sixing Wu
# @Time    : 2024/11/9
# @Mac, Python 3.11
# @Function: Using sentiment score and various features is price data to explore the correlation,
# output a scatter plot of the sentiment score and selected features

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def sentiment_correlatin_analysis(tweet_data, price_data, feature):
    # Load the CSV file into a DataFrame
    tweet_df = pd.read_csv(tweet_data)
    price_df = pd.read_csv(price_data)
    price_df = price_df.sort_values(by='Start', ascending=True).reset_index(drop=True)

    # Convert the 'date' column to a datetime format if it's not already
    tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.date

    # Generate daily info with sentiment scores and price data
    daily_info = tweet_df.groupby('date')['sentiment_score'].sum().reset_index()
    daily_info[feature] = price_df[feature]

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(daily_info['sentiment_score'], daily_info[feature], alpha=0.5)
    plt.title(f'Correlation between Bitcoin {feature} and Sentiment Score')
    plt.xlabel('Sentiment Score')
    plt.ylabel(f'{feature}')
    plt.show()


tweet_data = "../data/Sentiment_Analysis/sentiment_analysis.csv"
price_data = '../data/Bitcoin_Price/bitcoin_2021-02-08_2021-02-13.csv'
sentiment_correlatin_analysis(tweet_data, price_data, feature='Volume')