import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cufflinks as cf
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU, Layer
import keras.optimizers as optimizers
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import RegressorMixin, BaseEstimator
import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings('ignore')

# App title
st.markdown('''
# STOCK PRICE PREDICTION APP
Revealed are the stock price data for query companies!

**Credits**
- App Built by [Seyone Ganeshamoorthy](https://www.linkedin.com/in/seyone-ganeshamoorthy-6a3576129/)
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas`, `tensorflow`, `sklearn` and `datetime`
''')
st.write('---')

# Sidebar for user inputs
st.sidebar.header("Stock Ticker Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "^GSPC")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-07-31"))

# Add a button to fetch data & align it to center
if st.sidebar.button("Get Data"):
    progress_bar = st.sidebar.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        progress_bar.progress(percent_complete + 1)
    st.sidebar.success("Data successfully fetched!")

#st.sidebar.title("Navigation")
#page = st.sidebar.radio("Go to", ["Home", "Model"])
selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Model 1", "Model 2", "Model 3", "Stock Analysis"],  # required
            icons=["house", "book", "book", "book", "activity"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
page = selected

# Function to get the data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to preprocess the data
def preprocess_data(df):
    # Feature Engineering
    df['Return'] = df['Close'].pct_change() # simple daily return
    df['Moving_Avg_5'] = df['Close'].rolling(window=5).mean() # 5 days moving average
    df['Moving_Avg_10'] = df['Close'].rolling(window=10).mean() # 10 days moving average
    df['Moving_STD_10'] = df['Close'].rolling(window=10).std() # 10 days moving standard deviation

    # Replace NA values with 0
    df.fillna(0, inplace=True)

    # Removing first 9 rows since they contain 0 values
    df1 = df[9:]

    # Define the features and the target variable
    features = ['Open', 'High', 'Low', 'Return', 'Moving_Avg_5', 'Moving_Avg_10', 'Moving_STD_10']

    X = df1[features]
    y = df1['Close']

    # Feature Scaling
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)

    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1,1))

    # Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False, random_state=42)

    # Reshape the data for LSTM model
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled

def home_page(df):
    # Fetch data and display it
    df = fetch_data(ticker, start_date, end_date)
    st.write(f"Data for {ticker} from {start_date} to {end_date}")
    st.dataframe(df)

    # Plot closing price
    st.header("**Closing Price**")
    st.line_chart(df['Close'])

    # Plot volume
    st.header("**Volume**")
    st.line_chart(df['Volume'])

    # Bollinger bands
    st.header('**Bollinger Bands**')
    st.write("A technical analysis tool that consists of a centerline based on a moving average, flanked by two price channels (bands) above and below it, which expand or contract based on the volatility of the stock price.")
    qf=cf.QuantFig(df,title='Bollinger Bands',legend='top',name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

    # Simple Moving Averages
    st.header('**Simple Moving Averages**')
    st.write("An arithmetic moving average calculated by adding recent closing prices of a security and then dividing that by the number of time periods in the calculation average.")
    qf=cf.QuantFig(df,title='Simple Moving Averages - SMA',legend='top',name='GS')
    qf.add_sma([10,20],width=2,color=['green','lightgreen'],legendgroup=True)
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

    # Exponential Moving Averages
    st.header('**Exponential Moving Averages**')
    st.write("Similar to a simple moving average, but giving more weight to the most recent prices, thus responding more quickly to price changes.")
    qf=cf.QuantFig(df,title='Exponential Moving Averages - EMA',legend='top',name='GS')
    qf.add_ema([10,20],width=2,color=['green','lightgreen'],legendgroup=True)
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

    # Relative Strength Index
    st.header('**Relative Strength Index**')
    st.write("A momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.")
    qf=cf.QuantFig(df,title='Relative Strength Index - RSI',legend='top',name='GS')
    qf.add_rsi(periods=14,color=['green'],legendgroup=True)
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

    # MACD
    st.header('**Moving Average Convergence Divergence**')
    st.write("A trend-following momentum indicator that shows the relationship between two moving averages of a security`s price.")
    qf=cf.QuantFig(df,title='MACD',legend='top',name='GS')
    qf.add_macd()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

def train_and_predict(X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled):
    # Create a Estimator class for LSTM model
    class LSTMEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, optimizer='adam'):
            self.optimizer = optimizer
            self.model = Sequential()
            self.model.add(LSTM(150, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
            self.model.compile(optimizer=self.optimizer, loss='mse')

        def fit(self, X, y):
            self.model.fit(X, y, epochs=100, batch_size=64, verbose=0)
            return self

        def predict(self, X):
            return self.model.predict(X).flatten()
        
    # LSTM model with hyperparameter tuning
    # Define the parameter grid
    param_grid = {'optimizer': ['Adam']}

    # Grid search on LSTM
    grid = GridSearchCV(estimator=LSTMEstimator(), param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train_lstm, y_train)

    # Using the best estimator for LSTM from the grid search
    best_lstm_model = grid_result.best_estimator_

    # Predicting the test data
    pred_best_lstm = best_lstm_model.predict(X_test_lstm)

    # Inverse transform the predictions
    pred_best_lstm_inv = target_scaler.inverse_transform(pred_best_lstm.reshape(-1,1))

    # Define CNN model function
    def create_cnn_model(optimizer='Adam'):
        model = Sequential()
        model.add(Dense(64, input_dim=7, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return model

    # CNN model with hyperparameter tuning
    # Define the parameter grid
    param_grid = {'batch_size': [64, 128, 256],
                'epochs': [100, 200, 300],
                'optimizer': ['Adam','RMSprop','SGD']}

    # Grid search on CNN
    model = KerasRegressor(build_fn=create_cnn_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    # Using the best estimator for CNN from the grid search
    best_cnn_model = grid_result.best_estimator_

    # Predicting the test data
    pred_best_cnn = best_cnn_model.predict(X_test)

    # Inverse transform the predictions
    pred_best_cnn_inv = target_scaler.inverse_transform(pred_best_cnn.reshape(-1,1))
    
    # Stacked LSTM-CNN model with hyperparameter tuning
    stacked_tuned_pred = np.column_stack((pred_best_lstm, pred_best_cnn))

    # Train final regressor based on stacked predictions
    final_regressor_tuned = LinearRegression().fit(stacked_tuned_pred, y_test)

    # Make predictions
    final_pred_tuned = final_regressor_tuned.predict(stacked_tuned_pred)

    # Inverse transform the predictions
    final_pred_tuned_inv = target_scaler.inverse_transform(final_pred_tuned.reshape(-1,1))

    # Inverse transform the predictions
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1,1))

    return pred_best_lstm_inv, pred_best_cnn_inv, final_pred_tuned_inv, y_test_inv, best_lstm_model, best_cnn_model, final_regressor_tuned, X_scaled

def model_1(df):
    # Preprocess data
    X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled = preprocess_data(df)

    # Train models and get predictions
    pred_best_lstm_inv, pred_best_cnn_inv, final_pred_tuned_inv, y_test_inv, best_lstm_model, best_cnn_model, final_regressor_tuned, X_scaled = train_and_predict(X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled)

    # LSTM Model Predictions Vs Actual
    st.subheader("LSTM Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='blue', linewidth=2)
    plt.plot(df.index[-len(y_test):], pred_best_lstm_inv, label="Predicted", color='red', linestyle='--', linewidth=2)
    plt.title("LSTM Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Convolutional Neural Networks Model Predictions Vs Actual
    st.subheader("CNN Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='blue', linewidth=2)
    plt.plot(df.index[-len(y_test):], pred_best_cnn_inv, label="Predicted", color='green', linestyle='--', linewidth=2)
    plt.title("CNN Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Stacked Model Predictions Vs Actual
    st.subheader("LSTM-CNN Stacked Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='blue', linewidth=2)
    plt.plot(df.index[-len(y_test):], final_pred_tuned_inv, label="Predicted", color='purple', linestyle='--', linewidth=2)
    plt.title("LSTM-CNN Stacked Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)

    # Calculate the Evaluation Metrics for LSTM model
    lstm_mse = mean_squared_error(y_test_inv, pred_best_lstm_inv)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(y_test_inv, pred_best_lstm_inv)
    lstm_r2 = r2_score(y_test_inv, pred_best_lstm_inv)

    st.subheader("LSTM Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(lstm_mse))
    st.write("RMSE: {:.4f}".format(lstm_rmse))
    st.write("MAE: {:.4f}".format(lstm_mae))
    st.write("R2: {:.4f}".format(lstm_r2))

    # Calculate the Evaluation Metrics for CNN model
    cnn_mse = mean_squared_error(y_test_inv, pred_best_cnn_inv)
    cnn_rmse = np.sqrt(cnn_mse)
    cnn_mae = mean_absolute_error(y_test_inv, pred_best_cnn_inv)
    cnn_r2 = r2_score(y_test_inv, pred_best_cnn_inv)
    #Leave a space
    st.write("")
    st.subheader("CNN Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(cnn_mse))
    st.write("RMSE: {:.4f}".format(cnn_rmse))
    st.write("MAE: {:.4f}".format(cnn_mae))
    st.write("R2: {:.4f}".format(cnn_r2))

    # Calculate the Evaluation Metrics for Stacked model
    final_mse = mean_squared_error(y_test_inv, final_pred_tuned_inv)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test_inv, final_pred_tuned_inv)
    final_r2 = r2_score(y_test_inv, final_pred_tuned_inv)
    st.write("")
    st.subheader("LSTM-CNN Stacked Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(final_mse))
    st.write("RMSE: {:.4f}".format(final_rmse))
    st.write("MAE: {:.4f}".format(final_mae))
    st.write("R2: {:.4f}".format(final_r2))

    # Define X_scaled
    # X_scaled = preprocess_data(df)[7]

    last_features = X_scaled[-1:]

    # Predict using the Stacked LSTM-CNN model
    predicted_prices_stacked = []

    for day in range(2):
        # Reshape the last features for LSTM input shape
        last_features_lstm = last_features.reshape(1, 1, X_train_lstm.shape[2])
        
        # Predict using LSTM & CNN models
        lstm_pred_for_stacking = best_lstm_model.predict(last_features_lstm).flatten()
        cnn_pred_for_stacking = best_cnn_model.predict(last_features).flatten()

        # Stacking LSTM and CNN predictions
        stacked_predictions = np.column_stack((lstm_pred_for_stacking, cnn_pred_for_stacking))
        
        # Predict using the final regressor
        predicted_stacked = final_regressor_tuned.predict(stacked_predictions)

        # Reshape the predicted price for inverse scaling
        predicted_price_reshaped_stacked = predicted_stacked.reshape(-1, 1)
        
        # Saving the predicted price after inverse transformation
        inverse_transformed_price_stacked = target_scaler.inverse_transform(predicted_price_reshaped_stacked)[0][0]
        predicted_prices_stacked.append(inverse_transformed_price_stacked)

        # For the next prediction, update the features with the newly predicted price.
        # We reuse the logic from LSTM to compute new features for simplicity.
        new_features_stacked = np.array([
            last_features[0, 1], 
            last_features[0, 2], 
            predicted_price_reshaped_stacked[0, 0], 
            (predicted_price_reshaped_stacked[0, 0] - last_features[0, 3]) / last_features[0, 3],
            np.mean(np.append(last_features[0, 4:6], predicted_price_reshaped_stacked[0, 0])),
            np.mean(np.append(last_features[0, 5:], predicted_price_reshaped_stacked[0, 0])),
            np.std(np.append(last_features[0, 6:], predicted_price_reshaped_stacked[0, 0]))
        ], dtype=np.float32).reshape(1, 7)

        last_features = new_features_stacked

    st.write("")
    st.subheader("\nPredicted Closing Prices for the Next 2 Days (Stacked LSTM-CNN Model):")
    for i, price in enumerate(predicted_prices_stacked):
        st.write(f"Day {i+1}:" " {:.3f}".format(price))

    # Write bullet points
    st.write("")
    st.write(" - Stacked LSTM-CNN model performs better than the individual models (LSTM and CNN) but not by much")

    st.write(" ")
    st.write("* For good model performance, MSE, RMSE & MAE should be low. A high MSE indicates that the model is making large errors in its predictions. A smaller RMSE indicates that the observed values are closer to the predicted values. A smaller MAE indicates that the observed values are closer to the predicted values.")
    st.write("* R2 score should be high, closer to 1. A value of 0 indicates that the model is as good as one that always predicts the mean of the actual values. A negative R2 can indicate a model that is performing worse than this baseline.")

def train_and_predict_2(X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled):
    # Create a Estimator class for LSTM model
    class LSTMEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, optimizer='adam'):
            self.optimizer = optimizer
            self.model = Sequential()
            self.model.add(LSTM(150, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
            self.model.compile(optimizer=self.optimizer, loss='mse')

        def fit(self, X, y):
            self.model.fit(X, y, epochs=100, batch_size=64, verbose=0)
            return self

        def predict(self, X):
            return self.model.predict(X).flatten()
        
    # LSTM model with hyperparameter tuning
    param_grid = {'optimizer': ['Adam']}
    # Grid search on LSTM
    grid = GridSearchCV(estimator=LSTMEstimator(), param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train_lstm, y_train)

    # Using the best estimator for LSTM from the grid search
    best_lstm_model = grid_result.best_estimator_

    # Predicting the test data
    pred_best_lstm = best_lstm_model.predict(X_test_lstm)

    # Inverse transform the predictions
    pred_best_lstm_inv = target_scaler.inverse_transform(pred_best_lstm.reshape(-1,1))

    # Use auto_arima to find the best model for the data
    stepwise_model = auto_arima(y_train,
                                exogenous = X_train,
                                start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                m=12, start_P=0, seasonal=False,
                                d=1, D=1, trace=True,
                                error_action='ignore', suppress_warnings=True,
                                stepwise=True)

    # Fit ARIMAX model
    best_arimax_model = SARIMAX(y_train,
                    exog = X_train,
                    order=stepwise_model.order)
    best_arimax_fit = best_arimax_model.fit(disp=False)

    # Make predictions
    arimax_predictions = best_arimax_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)

    # Denormalize predictions
    arimax_pred_inv = target_scaler.inverse_transform(np.array(arimax_predictions).reshape(-1, 1))
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Stacked LSTM-ARIMAX Model with Hyperparameter Tuning
    stacked_tuned_predictions = np.column_stack((arimax_predictions, pred_best_lstm))

    # Train a final regressor based on the stacked predictions
    final_tuned_regressor = LinearRegression().fit(stacked_tuned_predictions, y_test)

    # Predict on the test data
    final_tuned_pred = final_tuned_regressor.predict(stacked_tuned_predictions)

    # Denormalize the predicted values
    final_tuned_pred_inv = target_scaler.inverse_transform(final_tuned_pred.reshape(-1,1))

    return pred_best_lstm_inv, arimax_pred_inv, final_tuned_pred_inv, y_test_inv, best_lstm_model, best_arimax_fit, final_tuned_regressor, X_scaled

def model_2(df):
    # Preprocess data
    X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled = preprocess_data(df)

    # Train models and get predictions
    pred_best_lstm_inv, arimax_pred_inv, final_tuned_pred_inv, y_test_inv, best_lstm_model, best_arimax_fit, final_tuned_regressor, X_scaled = train_and_predict_2(X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled)

    # LSTM Model Predictions Vs Actual
    st.subheader("LSTM Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='red', linewidth=2)
    plt.plot(df.index[-len(y_test):], pred_best_lstm_inv, label="Predicted", color='black', linestyle='--', linewidth=2)
    plt.title("LSTM Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Plot the predictions of ARIMAX model
    plt.figure(figsize=(16,8))
    plt.title('ARIMAX Model Prediction vs Actual', fontsize=18)
    plt.plot(df.index[-len(y_test):], y_test_inv, linewidth=2, color='darkgreen', label='Actual', linestyle='solid')
    plt.plot(df.index[-len(y_test):], arimax_pred_inv, linewidth=2, color='teal', label='Predicted', linestyle='dashed')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.legend(loc='best')
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Stacked Model Predictions Vs Actual
    st.subheader("LSTM-ARIMAX Stacked Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='red', linewidth=2)
    plt.plot(df.index[-len(y_test):], final_tuned_pred_inv, label="Predicted", color='darkblue', linestyle='--', linewidth=2)
    plt.title("LSTM-ARIMAX Stacked Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)

    # Calculate the Evaluation Metrics for LSTM model
    lstm_mse = mean_squared_error(y_test_inv, pred_best_lstm_inv)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(y_test_inv, pred_best_lstm_inv)
    lstm_r2 = r2_score(y_test_inv, pred_best_lstm_inv)

    st.subheader("LSTM Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(lstm_mse))
    st.write("RMSE: {:.4f}".format(lstm_rmse))
    st.write("MAE: {:.4f}".format(lstm_mae))
    st.write("R2: {:.4f}".format(lstm_r2))

    # Calculate Evaluation Metrics for ARIMA model
    arimax_mse = mean_squared_error(y_test_inv, arimax_pred_inv)
    arimax_rmse = np.sqrt(arimax_mse)    
    arimax_mae = mean_absolute_error(y_test_inv, arimax_pred_inv)
    arimax_r2 = r2_score(y_test_inv, arimax_pred_inv)
    #Leave a space
    st.write("")
    st.subheader("ARIMAX Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(arimax_mse))
    st.write("RMSE: {:.4f}".format(arimax_rmse))
    st.write("MAE: {:.4f}".format(arimax_mae))
    st.write("R2: {:.4f}".format(arimax_r2))

    # Calculate the Evaluation Metrics for Stacked model
    final_mse = mean_squared_error(y_test_inv, final_tuned_pred_inv)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test_inv, final_tuned_pred_inv)
    final_r2 = r2_score(y_test_inv, final_tuned_pred_inv)

    st.write("")
    st.subheader("LSTM-ARIMAX Stacked Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(final_mse))
    st.write("RMSE: {:.4f}".format(final_rmse))
    st.write("MAE: {:.4f}".format(final_mae))
    st.write("R2: {:.4f}".format(final_r2))

    last_features = X_scaled[-1:]

    # Predict stock prices using the Stacked model for the next 2 days
    predicted_prices_stacked = []

    last_features_stacked = last_features.copy()

    future_exog_day1 = last_features_stacked.copy()
    future_exog_day2 = last_features_stacked.copy()
    future_exog_data = [future_exog_day1, future_exog_day2]

    for i in range(2):
        # Predict using LSTM model
        last_features_stacked_lstm = last_features_stacked.reshape(1, 1, X_train_lstm.shape[2])
        predicted_price_stacked_lstm = best_lstm_model.predict(last_features_stacked_lstm)

        # Predict using ARIMAX model
        future_exog = np.vstack(future_exog_data[i])
        forecasted_price_stacked_arimax = best_arimax_fit.forecast(steps=1, exog=future_exog)

        # Combine the predictions
        stacked_predictions = np.column_stack((forecasted_price_stacked_arimax, predicted_price_stacked_lstm))

        # Predict using the final regressor
        final_pred_stacked = final_tuned_regressor.predict(stacked_predictions)

        # Reshape the predicted price for inverse scaling
        final_pred_stacked_reshaped = final_pred_stacked.reshape(-1, 1)

        # Saving the predicted price after inverse transformation
        inverse_transformed_price = target_scaler.inverse_transform(final_pred_stacked_reshaped)[0][0]
        predicted_prices_stacked.append(inverse_transformed_price)

        # Computing new features based on the predicted price
        new_features = np.array([
            last_features_stacked[0, 1],  # 'Open' value for the next day (shifted from 'High' of the current day)
            last_features_stacked[0, 2],  # 'High' value for the next day (shifted from 'Low' of the current day)
            final_pred_stacked_reshaped[0, 0], # 'Low' value (using the predicted closing price as an approximation)
            (final_pred_stacked_reshaped[0, 0] - last_features_stacked[0, 3]) / last_features_stacked[0, 3],  # Return for the predicted day
            np.mean(np.append(last_features_stacked[0, 4:6], final_pred_stacked_reshaped[0, 0])),  # 5-day moving average
            np.mean(np.append(last_features_stacked[0, 5:], final_pred_stacked_reshaped[0, 0])),  # 10-day moving average
            np.std(np.append(last_features_stacked[0, 6:], final_pred_stacked_reshaped[0, 0]))    # 10-day moving standard deviation
        ], dtype=np.float32).reshape(1, 7)

        last_features_stacked = new_features

    st.subheader("Predicted Closing Prices for the Next 2 Days (Stacked LSTM-ARIMAX Model):")
    for i, price in enumerate(predicted_prices_stacked):
        st.write(f"Day {i+1}:" " {:.3f}".format(price))

    # Write bullet points
    st.write("")
    st.write(" - Stacked LSTM-ARIMAX model performs better than the individual models")
    
    st.write("")
    st.write("* For good model performance, MSE, RMSE & MAE should be low. A high MSE indicates that the model is making large errors in its predictions. A smaller RMSE indicates that the observed values are closer to the predicted values. A smaller MAE indicates that the observed values are closer to the predicted values.")
    st.write("* R2 score should be high, closer to 1. A value of 0 indicates that the model is as good as one that always predicts the mean of the actual values. A negative R2 can indicate a model that is performing worse than this baseline.")

def train_and_predict_3(X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled):
    # Create a Estimator class for LSTM model
    class LSTMEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, optimizer='adam'):
            self.optimizer = optimizer
            self.model = Sequential()
            self.model.add(LSTM(150, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
            self.model.compile(optimizer=self.optimizer, loss='mse')

        def fit(self, X, y):
            self.model.fit(X, y, epochs=100, batch_size=64, verbose=0)
            return self

        def predict(self, X):
            return self.model.predict(X).flatten()
        
    # LSTM model with hyperparameter tuning
    param_grid = {'optimizer': ['Adam']}
    # Grid search on LSTM
    grid = GridSearchCV(estimator=LSTMEstimator(), param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train_lstm, y_train)

    # Using the best estimator for LSTM from the grid search
    best_lstm_model = grid_result.best_estimator_

    # Predicting the test data
    pred_best_lstm = best_lstm_model.predict(X_test_lstm)

    # Inverse transform the predictions
    pred_best_lstm_inv = target_scaler.inverse_transform(pred_best_lstm.reshape(-1,1))

    class RHNL(Layer):
        def __init__(self, units, depth, **kwargs):
            super(RHNL, self).__init__(**kwargs)
            self.units = units
            self.depth = depth

        def build(self, input_shape):
            self.W_h = self.add_weight(name='W_h', shape=(input_shape[-1], self.units), 
                                    initializer='uniform', trainable=True)
            
            self.W_t = [self.add_weight(name=f'W_t_{i}', shape=(self.units, self.units),
                                        initializer='uniform', trainable=True) for i in range(self.depth)]
            
            self.W_c = [self.add_weight(name=f'W_c_{i}', shape=(self.units, self.units),
                                        initializer='uniform', trainable=True) for i in range(self.depth)]
        
        def call(self, x):
            h = tf.matmul(x, self.W_h)
            for i in range(self.depth):
                t = tf.sigmoid(tf.matmul(h, self.W_t[i]))
                c = tf.tanh(tf.matmul(h, self.W_c[i]))
                h = h * t + c * (1 - t)
            return h
        
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'depth': self.depth})
            return config
    
    def create_rhn_model(rhn_units, rhn_depth, output_units=1):
        model = Sequential()
        model.add(RHNL(rhn_units, rhn_depth, input_shape=(X_train.shape[1], )))
        model.add(Dense(output_units, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    # RHN model with hyperparameter tuning
    # Define the parameter grid
    param_grid = {'rhn_units': [64, 128, 256], 'rhn_depth': [1, 2, 3]}

    # Create the KerasRegressor
    rhn_model = KerasRegressor(build_fn=create_rhn_model, epochs=100, batch_size=64, verbose=0)

    # Grid search on RHN
    grid = GridSearchCV(estimator=rhn_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    # Using the best estimator for RHN from the grid search
    best_rhn_model = grid_result.best_estimator_

    # Predicting the test data
    pred_best_rhn = best_rhn_model.predict(X_test)

    # Inverse transform the predictions
    pred_best_rhn_inv = target_scaler.inverse_transform(pred_best_rhn.reshape(-1,1))

    # Stacked LSTM-RHN model with hyperparameter tuning
    stacked_tuned_pred = np.column_stack((pred_best_lstm, pred_best_rhn))

    # Train final regressor based on stacked predictions
    final_regressor_tuned = LinearRegression().fit(stacked_tuned_pred, y_test)

    # Predict using the Stacked LSTM-RHN model
    final_tuned_pred = final_regressor_tuned.predict(stacked_tuned_pred)

    # Inverse the predictions
    final_tuned_pred_inv = target_scaler.inverse_transform(final_tuned_pred.reshape(-1,1))
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1,1))

    return pred_best_lstm_inv, pred_best_rhn_inv, final_tuned_pred_inv, y_test_inv, best_lstm_model, best_rhn_model, final_regressor_tuned, X_scaled

def model_3(df):
    # Preprocess data
    X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled = preprocess_data(df)

    # Train models and get predictions
    pred_best_lstm_inv, pred_best_rhn_inv, final_tuned_pred_inv, y_test_inv, best_lstm_model, best_rhn_model, final_regressor_tuned, X_scaled = train_and_predict_3(X_train_lstm, X_test_lstm, X_train, y_train, X_test, y_test, target_scaler, X_scaled)

    # LSTM Model Predictions Vs Actual
    st.subheader("LSTM Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='blue', linewidth=2)
    plt.plot(df.index[-len(y_test):], pred_best_lstm_inv, label="Predicted", color='red', linestyle='--', linewidth=2)
    plt.title("LSTM Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Random Forest Model Predictions Vs Actual
    st.subheader("Recurrent Highway Networks Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='blue', linewidth=2)
    plt.plot(df.index[-len(y_test):], pred_best_rhn_inv, label="Predicted", color='green', linestyle='--', linewidth=2)
    plt.title("RHN Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Stacked Model Predictions Vs Actual
    st.subheader("LSTM-RHN Stacked Model Predictions Vs Actual")
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test_inv, label="Actual", color='blue', linewidth=2)
    plt.plot(df.index[-len(y_test):], final_tuned_pred_inv, label="Predicted", color='purple', linestyle='--', linewidth=2)
    plt.title("LSTM-RHN Stacked Model Predictions Vs Actual", fontsize=18)
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt)

    # Calculate the Evaluation Metrics for LSTM model
    lstm_mse = mean_squared_error(y_test_inv, pred_best_lstm_inv)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(y_test_inv, pred_best_lstm_inv)
    lstm_r2 = r2_score(y_test_inv, pred_best_lstm_inv)

    st.subheader("LSTM Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(lstm_mse))
    st.write("RMSE: {:.4f}".format(lstm_rmse))
    st.write("MAE: {:.4f}".format(lstm_mae))
    st.write("R2: {:.4f}".format(lstm_r2))

    # Calculate the Evaluation Metrics for RHN model
    rhn_mse = mean_squared_error(y_test_inv, pred_best_rhn_inv)
    rhn_rmse = np.sqrt(rhn_mse)
    rhn_mae = mean_absolute_error(y_test_inv, pred_best_rhn_inv)
    rhn_r2 = r2_score(y_test_inv, pred_best_rhn_inv)

    #Leave a space
    st.write("")
    st.subheader("RHN Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(rhn_mse))
    st.write("RMSE: {:.4f}".format(rhn_rmse))
    st.write("MAE: {:.4f}".format(rhn_mae))
    st.write("R2: {:.4f}".format(rhn_r2))

    # Calculate the Evaluation Metrics for Stacked model
    final_mse = mean_squared_error(y_test_inv, final_tuned_pred_inv)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test_inv, final_tuned_pred_inv)
    final_r2 = r2_score(y_test_inv, final_tuned_pred_inv)

    st.write("")
    st.subheader("LSTM-RHN Stacked Model Evaluation Metrics")
    st.write("MSE: {:.4f}".format(final_mse))
    st.write("RMSE: {:.4f}".format(final_rmse))
    st.write("MAE: {:.4f}".format(final_mae))
    st.write("R2: {:.4f}".format(final_r2))

    # Predicting stock prices for the next 2 days
    last_features = X_scaled[-1:]

    # Predict using the Stacked LSTM-RHN model
    predicted_prices_stacked = []

    for day in range(2):
        # Reshape the last features for LSTM input shape
        last_features_lstm = last_features.reshape(1, 1, X_train_lstm.shape[2])
        
        # Predict using LSTM & CNN models
        lstm_pred_for_stacking = best_lstm_model.predict(last_features_lstm).flatten()
        rhn_pred_for_stacking = best_rhn_model.predict(last_features).flatten()

        # Stacking LSTM and rhn predictions
        stacked_predictions = np.column_stack((lstm_pred_for_stacking, rhn_pred_for_stacking))
        
        # Predict using the final regressor
        predicted_stacked = final_regressor_tuned.predict(stacked_predictions)

        # Reshape the predicted price for inverse scaling
        predicted_price_reshaped_stacked = predicted_stacked.reshape(-1, 1)
        
        # Saving the predicted price after inverse transformation
        inverse_transformed_price_stacked = target_scaler.inverse_transform(predicted_price_reshaped_stacked)[0][0]
        predicted_prices_stacked.append(inverse_transformed_price_stacked)

        # For the next prediction, update the features with the newly predicted price.
        # We reuse the logic from LSTM to compute new features for simplicity.
        new_features_stacked = np.array([
            last_features[0, 1], 
            last_features[0, 2], 
            predicted_price_reshaped_stacked[0, 0], 
            (predicted_price_reshaped_stacked[0, 0] - last_features[0, 3]) / last_features[0, 3],
            np.mean(np.append(last_features[0, 4:6], predicted_price_reshaped_stacked[0, 0])),
            np.mean(np.append(last_features[0, 5:], predicted_price_reshaped_stacked[0, 0])),
            np.std(np.append(last_features[0, 6:], predicted_price_reshaped_stacked[0, 0]))
        ], dtype=np.float32).reshape(1, 7)

        last_features = new_features_stacked

    st.subheader("\nPredicted Closing Prices for the Next 2 Days (Stacked LSTM-RHN Model):")
    for i, price in enumerate(predicted_prices_stacked):
        st.write(f"Day {i+1}:" " {:.3f}".format(price))

    # Write bullet points
    st.write("")
    st.write(" - Stacked LSTM-RHN model performs better than LSTM and RHN models in terms of RMSE, MAE and R2 score")

    st.write(" ")
    st.write("* For good model performance, MSE, RMSE & MAE should be low. A high MSE indicates that the model is making large errors in its predictions. A smaller RMSE indicates that the observed values are closer to the predicted values. A smaller MAE indicates that the observed values are closer to the predicted values.")
    st.write("* R2 score should be high, closer to 1. A value of 0 indicates that the model is as good as one that always predicts the mean of the actual values. A negative R2 can indicate a model that is performing worse than this baseline.")

def stock_analysis():
    st.title("Stock Analysis")

    # Define the ticker list
    ticker_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX', 'AMD', 'INTC', 'CSCO']

    # Create an empty dataframe to store the list
    results = pd.DataFrame(columns=['Ticker', 'Moving Average', 'RSI', 'A/D Lines', 'ADX', 'Recommendation'])

    # Iterate through the list
    for ticker in ticker_list:
        # Fetch historical data
        data = yf.download(ticker, start="2021-01-01", interval='1d')

        # Calculate moving average for 10 days
        data['MA'] = data['Close'].rolling(10).mean()

        # Calculate RSI for 10 days
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = delta.where(delta < 0, 0)
        avg_gain = gain.rolling(10).mean()
        avg_loss = loss.rolling(10).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate A/D lines
        data['UpMove'] = data['High'] - data['High'].shift(1)
        data['DownMove'] = data['Low'].shift(1) - data['Low']
        data['UpVolume'] = data['UpMove'] * data['Volume']
        data['DownVolume'] = data['DownMove'] * data['Volume']
        data['PosDM'] = data['UpMove']
        data['NegDM'] = data['DownMove']
        data.loc[(data['PosDM'] < data['NegDM']), 'PosDM'] = 0
        data.loc[(data['PosDM'] > data['NegDM']), 'NegDM'] = 0
        data['PosDI'] = data['PosDM'].rolling(10).mean()
        data['NegDI'] = data['NegDM'].rolling(10).mean()
        data['AD'] = (data['PosDI'] - data['NegDI']) / (data['NegDI'] + data['PosDI'])

        # Calculate ADX
        data['ADX'] = 100 * (data['PosDI'] - data['NegDI']) / (data['PosDI'] + data['NegDI'])
        data['ADX'] = data['ADX'].rolling(window=10).mean()

        # Determine the recommendation
        if data['Close'].iloc[-1] > data['MA'].iloc[-1] and rsi.iloc[-1] > 70 and data['AD'].iloc[-1] > data['AD'].iloc[-2] and data['ADX'].iloc[-1] > 25:
            recommendation = 'Buy'
        elif data['Close'].iloc[-1] < data['MA'].iloc[-1] and rsi.iloc[-1] < 30 and data['AD'].iloc[-1] < data['AD'].iloc[-2] and data['ADX'].iloc[-1] > 25:
            recommendation = 'Sell'
        else:
            recommendation = 'Hold'

        # Append the result to the dataframe
        results = results.append({'Ticker': ticker, 'Moving Average': data['MA'].iloc[-1], 'RSI': rsi.iloc[-1], 'A/D Lines': data['AD'].iloc[-1], 'ADX': data['ADX'].iloc[-1], 'Recommendation': recommendation}, ignore_index=True)

        # Display a progress bar
        progress_bar = st.sidebar.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
        progress_bar.empty()

    # Display the results
    st.dataframe(results)

if page == "Home":
    df = fetch_data(ticker, start_date, end_date)
    home_page(df)
elif page == "Model 1":
    df = fetch_data(ticker, start_date, end_date)
    model_1(df)
elif page == "Model 2":
    df = fetch_data(ticker, start_date, end_date)
    model_2(df)
elif page == "Model 3":
    df = fetch_data(ticker, start_date, end_date)
    model_3(df)
elif page == "Stock Analysis":
    stock_analysis()