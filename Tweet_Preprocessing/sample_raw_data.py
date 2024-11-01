import pandas as pd
import csv
nRowsRead= 10000
#remember to change the directory when u r using
df1 = pd.read_csv('/Users/nataliehu/Desktop/emory/Emory homework/2024/CS 470/Final project/Bitcoin_Price_Analysis/Bitcoin_tweets.csv', delimiter=',', nrows = nRowsRead, low_memory= False, )
df1.dataframeName = 'Bitcoin_tweets.csv'
df1= df1.dropna()
#remember to change the directory when u r using
df1.to_csv('../data/sampled_tweets.csv', index=False)





