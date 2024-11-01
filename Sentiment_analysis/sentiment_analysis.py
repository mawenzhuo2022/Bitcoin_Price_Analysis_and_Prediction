# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/1 8:10
# @Function:
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# 确保已经下载了 NLTK 的情感分析库
nltk.download('vader_lexicon')


def feature_transformation_and_sentiment_analysis(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    # 确保 'processed_text' 列存在
    if 'processed_text' in df.columns:
        # 特征转换 - BoW
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(df['processed_text'])

        # 特征转换 - TF-IDF
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)

        # 情感分析
        sid = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['processed_text'].apply(lambda text: sid.polarity_scores(text)['compound'])

        # 输出新的 CSV 文件
        df.to_csv(output_file, index=False)
    else:
        print("Error: The CSV file does not contain a 'processed_text' column.")


# 指定输入和输出文件
input_csv = 'path_to_input_csv.csv'
output_csv = 'path_to_output_csv.csv'

# 调用处理函数
feature_transformation_and_sentiment_analysis(input_csv, output_csv)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# 确保已经下载了 NLTK 的情感分析库
nltk.download('vader_lexicon')


def feature_transformation_and_sentiment_analysis(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    # 确保 'processed_text' 列存在
    if 'processed_text' in df.columns:
        # 特征转换 - BoW
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(df['processed_text'])

        # 特征转换 - TF-IDF
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)

        # 情感分析
        sid = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['processed_text'].apply(lambda text: sid.polarity_scores(text)['compound'])

        # 输出新的 CSV 文件
        df.to_csv(output_file, index=False)
    else:
        print("Error: The CSV file does not contain a 'processed_text' column.")


# 指定输入和输出文件
input_csv = '../data/text_sentiment.csv'
output_csv = '../data/sentiment_analysis.csv'

# 调用处理函数
feature_transformation_and_sentiment_analysis(input_csv, output_csv)
