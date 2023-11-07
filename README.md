# LSTM_Based_Stock_Market_Prediction

## Overview
This Python module defines a trading model that leverages sentiment analysis and Long Short-Term Memory (LSTM) neural networks to predict stock price movements based on public opinion extracted from Twitter data.

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `keras` (LSTM, Dense, Dropout, Sequential from keras.layers)
- `vaderSentiment` (SentimentIntensityAnalyzer)

## Data
The model requires three data frames:
- `df_tweet`: Tweets data.
- `df_company_tweet`: A mapping between tweets and companies.
- `df_company`: Company information.
- `Download Raw Data`: [Google Drive](https://drive.google.com/drive/folders/1INR3u0UW13EGHZ4wsmxeYS3BAo2zC87f)

## Class `Model`
The `Model` class contains all the methods needed to preprocess the data, perform sentiment analysis, train LSTM models, and plot results.

### Initialization
Create an instance of the `Model` class by passing the required data frames as arguments:
```python
model = Model(df_tweet, df_company_tweet, df_company)
