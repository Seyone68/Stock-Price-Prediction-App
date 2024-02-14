<h1><b>Enhanced Stock Price Prediction using Machine-Learning</b></h1>

<h2>Introduction</h2>
<p>This repository contains the implementation of an advanced stock price prediction model, leveraging a fusion of ensemble machine learning models and strategic feature engineering. Our goal is to enhance the accuracy of stock price forecasts using a combination of sophisticated algorithms and data analysis techniques.</p>

   <h2>Project Overview</h2>
   <p>The project incorporates various machine learning algorithms to develop an ensemble of models that predict stock prices more accurately than traditional methods. The algorithms used include:</p>
    <ul>
        <li>Long Short-Term Memory (LSTM)</li>
        <li>Random Forest (RF)</li>
        <li>Support Vector Regression (SVR)</li>
        <li>Extreme Gradient Boosting (XGBoost)</li>
        <li>AutoRegressive Integrated Moving Average with eXogenous inputs (ARIMAX)</li>
        <li>Gated Recurrent Unit (GRU)</li>
        <li>Bidirectional LSTM (BiLSTM)</li>
        <li>Convolutional Neural Network (CNN)</li>
        <li>Recurrent Highway Network (RHN)</li>
        <li>Temporal Convolutional Network (TCN)</li>
    </ul>
    <p>These algorithms were paired to create stacking ensemble models, enhancing the predictive power through their combined strengths.</p>

  <h2>Features</h2>
    <p>The model incorporates the following features to predict stock prices:</p>
    <ul>
        <li>Daily return</li>
        <li>5-day moving average</li>
        <li>10-day moving average</li>
        <li>10-day moving standard deviation</li>
    </ul>
    <p>These features were selected to capture both the short-term volatility and long-term trends in stock prices.</p>

  <h2>User Interface</h2>
    <p>An interactive user interface was developed using the Streamlit library, allowing users to easily interact with the model, input parameters, and view predictions.</p>

  <h2>Installation</h2>
    <p>To set up this project, follow these steps:</p>
    <ol>
        <li>Clone this repository: <code>git clone &lt;repository-url&gt;</code></li>
        <li>Install required Python packages: <code>pip install -r requirements.txt</code></li>
    </ol>

  <h2>Usage</h2>
    <p>To run the Streamlit application:</p>
    <pre><code>streamlit run app.py</code></pre>
    <p>Navigate to the URL provided by Streamlit to interact with the application.</p>

  <h2>Ensemble Models</h2>
    <p>The following ensemble models were developed:</p>
    <ul>
        <li>LSTM-RF</li>
        <li>LSTM-SVR</li>
        <li>LSTM-XGBoost</li>
        <li>LSTM-ARIMAX</li>
        <li>RF-SVR</li>
        <li>RF-XGBoost</li>
        <li>SVR-XGBoost</li>
        <li>XGBoost-ARIMAX</li>
        <li>LSTM-GRU-BiLSTM</li>
        <li>LSTM-CNN</li>
        <li>LSTM-RHN</li>
        <li>LSTM-TCN</li>
    </ul>

  <h2>Best Performing Models</h2>
    <p>The models that showed the best performance in our experiments were:</p>
    <ul>
        <li>LSTM-ARIMAX</li>
        <li>LSTM-RHN</li>
        <li>LSTM-CNN</li>
    </ul>
