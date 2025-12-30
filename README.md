# Bitcoin-Price-Prediction-By-ML_
Bitcoin Price Prediction by Machine Learning

Forecasting Bitcoin prices using statistical time-series models and deep learning techniques in Python.

📌 Project Description

This project analyzes historical Bitcoin (BTC-USD) price data and applies multiple time-series forecasting and machine learning models to predict future prices.

The notebook combines:

Exploratory data analysis and visualization

Classical statistical models

Modern deep learning approaches

The objective is to compare different modeling strategies and evaluate their predictive performance on Bitcoin price movements.

📊 Dataset

Source: Yahoo Finance (BTC-USD) using yfinance

Time span: ~8 years of daily data

Features used:

Open

High

Low

Close

Volume

The data is resampled to daily frequency and indexed by timestamp.

🧰 Libraries & Tools

Data handling: pandas, numpy

Data collection: yfinance

Visualization: matplotlib, seaborn, plotly

Statistical modeling: statsmodels

Machine learning: scikit-learn

Forecasting models: ARIMA, Prophet

Deep learning: TensorFlow, Keras (LSTM)

Environment: Jupyter Notebook

📁 Project Structure
Bitcoin-Price-Prediction-By-ML_/
│
├── BitCoinPricePrediction.ipynb   # Main Jupyter Notebook
├── README.md                      # Project documentation

🔍 Methodology
1️⃣ Exploratory Data Analysis (EDA)

Price and volume trends over time

Interactive visualizations using Plotly

Autocorrelation analysis

Focus on recent time windows and long-term behavior

2️⃣ ARIMA Model

Classical time-series forecasting approach

Train/test split: 70% / 30%

Rolling forecast strategy

Performance evaluated using RMSE

Visualization of predicted vs actual prices

3️⃣ Prophet Model

Trend and seasonality-based forecasting

Automatic handling of time-series components

Future price forecasting with confidence intervals

Clear visualization of predicted vs actual prices

4️⃣ LSTM Neural Network

Deep learning model for sequential data

Min-Max scaling of prices

Bidirectional LSTM architecture

Trained for 100 epochs

Predictions compared against real prices

Evaluated using RMSE

📈 Evaluation Metric

Root Mean Squared Error (RMSE)
Used to compare forecasting accuracy across ARIMA, Prophet, and LSTM models.

📊 Visual Outputs

The notebook includes:

Bitcoin price and volume trends

Autocorrelation plots

ARIMA prediction plots

Prophet forecasts with confidence bounds

LSTM predicted vs actual price comparison

🚀 How to Run the Project
1. Clone the repository
git clone https://github.com/FatemeJabbari/Bitcoin-Price-Prediction-By-ML_.git
cd Bitcoin-Price-Prediction-By-ML_

2. Install dependencies
pip install pandas numpy yfinance scikit-learn statsmodels prophet tensorflow plotly seaborn matplotlib

3. Launch Jupyter Notebook
jupyter notebook


Open:

BitCoinPricePrediction.ipynb


Run all cells sequentially.

📝 Key Insights

Bitcoin prices show strong volatility and long-term trends

ARIMA performs reasonably well for short-term forecasting

Prophet effectively captures trend and seasonality

LSTM models nonlinear patterns and performs competitively

Deep learning models require careful tuning but offer flexibility

📌 Future Improvements

Hyperparameter optimization for all models

Incorporating additional features (macroeconomic indicators)

Multivariate LSTM models

Longer-horizon forecasting

Real-time prediction pipeline

Model explainability and uncertainty analysis

📬 Contact

Fateme Jabbari
GitHub: https://github.com/FatemeJabbari
