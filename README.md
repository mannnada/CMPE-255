# CMPE-255

## CoinCast: Precision Bitcoin Market Forecasting

### Project Overview

The CoinCast project aims to predict Bitcoin's daily closing price using machine learning and time series forecasting techniques. Given Bitcoin's extreme volatility and increasing prominence as a financial asset, accurate price forecasting has significant implications for investors, traders, and financial analysts.

We explored multiple models—ranging from traditional statistical methods like ARIMA to advanced machine learning models such as LSTM (Long Short-Term Memory). This comparative analysis highlights the strengths and limitations of each model, ultimately showcasing LSTM's superior performance due to its ability to capture temporal dependencies and non-linear trends in Bitcoin price data.

### Introduction and Problem Statement

The Challenge
Bitcoin, a decentralized digital currency, has become a cornerstone of the cryptocurrency market. However, its price is highly volatile, influenced by factors such as market sentiment, regulatory news, and global economic conditions. These unpredictable fluctuations pose challenges for investors trying to make data-driven decisions.

### Goal of the Project

The project seeks to build a robust forecasting system that accurately predicts Bitcoin's closing price using historical price data. We address this challenge using a combination of time series analysis and machine learning techniques, comparing their performance to determine the most effective model.

### Dataset Overview

The dataset used in this project consists of daily Bitcoin price data, capturing various market metrics:

#### Date: The specific day of recorded data.
#### Open Price: The price at which Bitcoin started trading on that day.
#### High Price: The highest price Bitcoin reached during the day.
#### Low Price: The lowest price Bitcoin fell to during the day.
#### Close Price: The price at which Bitcoin closed at the end of the day (target variable).
#### Volume Traded: The total trading volume on that day.
This dataset provides a clear and structured view of Bitcoin's price movements over time. The closing price is chosen as the prediction target because it reflects the final sentiment and movement for each trading day.



### Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to gain insights into the dataset and identify patterns, trends, and relationships:

### 1. Visualizations:

#### 2. Line Charts: Visualized trends in Bitcoin's opening, closing, high, and low prices over time.
#### 3. Moving Averages: Applied moving average smoothing techniques to observe long-term trends in the price data.

### Correlation Analysis:

Generated correlation heatmaps to analyze relationships between features like Open, High, Low, and Close prices.
Observed that prices are strongly correlated, which validates the dataset's consistency.
### Price Volatility Analysis:

Calculated and plotted daily price fluctuations to highlight the volatile nature of Bitcoin.

## Data Preprocessing

To prepare the data for modeling, the following preprocessing steps were applied:

### 1. Handling Missing Values:

Checked for missing or inconsistent data and addressed them to ensure model robustness.
Feature Scaling:

### 2. Feature Scaling:

Applied Min-Max Scaling to normalize price features (Open, High, Low, Close) between 0 and 1, ensuring faster convergence during model training.
Time Series Transformation:

### 3. Time series Transformation:
For models like LSTM and ARIMA, the dataset was transformed into sequences suitable for temporal analysis.
Train-Test Split:

### 4. Train_Test Split:
Split the dataset into 80% training and 20% testing sets to evaluate model generalization.0

## Modeling

The following models were implemented and rigorously tested for their ability to predict Bitcoin prices:

### 1. ARIMA (AutoRegressive Integrated Moving Average)
Description: A traditional time series model that uses past observations to predict future values. ARIMA works well with linear data and stationary time series.
Strengths: Simple, interpretable, and effective for smaller datasets.
Limitations: Struggles with non-linear relationships and high volatility.
### 2. LSTM (Long Short-Term Memory Neural Network)
Description: An advanced deep learning model specifically designed for time series data. LSTMs capture long-term dependencies and non-linear trends effectively.
Strengths: Handles sequential data efficiently and performs well with complex patterns.
Limitations: Computationally intensive and requires careful hyperparameter tuning.
### 3. Random Forest
Description: An ensemble model that combines multiple decision trees to provide robust predictions.
Strengths: Handles non-linear data and avoids overfitting.
Limitations: Does not inherently consider temporal relationships in the data.
### 4. Gradient Boosting
Description: An iterative model that builds strong learners by minimizing errors of previous models.
Strengths: Strong predictive performance for structured data.
Limitations: Computationally expensive for large datasets.

## Results and Comparison

The models were evaluated using Root Mean Squared Error (RMSE) as the performance metric. The results are as follows:

Model	RMSE	Observations
ARIMA	            20,519.2	  Struggled with volatility and non-linearity.
Random Forest	    6,759.12	  Improved performance but lacked temporal awareness.
Gradient Boosting	7,872.76	  Competitive performance with minor lag.
LSTM	            3,787.76	  Captured temporal trends and non-linearities effectively.

### Key Finding

The LSTM model demonstrated the best performance, achieving the lowest RMSE. Its ability to capture long-term dependencies and complex patterns makes it the most suitable model for Bitcoin price forecasting.

## Conclusion

This project successfully applied and compared various time series and machine learning models to forecast Bitcoin's closing price. Key findings include:

1. LSTM emerged as the most effective model due to its superior handling of sequential, non-linear data.
2. Traditional models like ARIMA struggled with Bitcoin's inherent volatility.
3. Ensemble methods (Random Forest, Gradient Boosting) performed well but did not outperform LSTM.

## Future Work

To further enhance the forecasting accuracy and real-world applicability of this project, the following steps can be considered:

### 1. Incorporating External Factors:
Sentiment Analysis: Integrate social media trends, news articles, and market sentiment data.
Macroeconomic Indicators: Include data such as interest rates and inflation rates.

### 2. Advanced Deep Learning Models:
Implement Transformer Networks for better long-range dependency capture.
Explore hybrid models combining LSTM with attention mechanisms.
### 3. Real-Time Forecasting:

Deploy the model as a real-time system using cloud platforms like AWS or GCP for live predictions.

## Technologies and Tools Used

#### Programming Language: Python
#### Libraries:
Data Analysis: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Modeling: TensorFlow/Keras, Scikit-learn, Statsmodels
Models: ARIMA, LSTM, Random Forest, Gradient Boostin
