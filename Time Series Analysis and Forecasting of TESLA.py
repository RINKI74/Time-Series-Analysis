#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


data = pd.read_csv("TSLA.csv", index_col ="Date", parse_dates = True)


# In[7]:


data.shape


# In[8]:


data.head()


# In[ ]:


Dataset contains following fields
Date -
Each trading day Open 
Open price of stock High 
High price of stock in the particular day Low 
Low price of the stock in the particular day Close 
Close price of the stock at end of the day Adj Close 
Adjusted close price of stock * 
Volume - Volume traded in the entire day

The price quoted at the end of the trading day is the price of the last lot of stock is referred to as the stock's closing price.
The adjusted closing price is often used when performing a detailed analysis of historical returns.


# In[9]:


plt.figure(figsize=(20, 15))

plt.subplot(2,1,1)
plt.plot(data['Adj Close'], label='Adj Close', color="purple")
plt.legend(loc="upper right")
plt.title('Adj Close Prices of Tesla')

plt.subplot(2,1,2)
plt.plot(data['Volume'], label='Volume', color="Orange")
plt.legend(loc="upper right")
plt.title('Volume Of Shares Traded')


# A Time-series data is a series of data points or observations recorded at different or regular time intervals. In general, a time series is a sequence of data points taken at equally spaced time intervals. The frequency of recorded data points may be hourly, daily, weekly, monthly, quarterly or annually.
# 
# Need of Time series is:
# To understand seasonal patterns.
# Evaluate current progress.
# Forecasting of observations
# 
# Types of Time Series:
# Trend - The trend shows a general direction of the time series data over a long period of time. A trend can be increasing(upward), decreasing(downward), or horizontal(stationary).
# 
# Seasonality - The seasonality component exhibits a trend that repeats with respect to timing, direction, and magnitude. Some examples include an increase in water consumption in summer due to hot weather conditions.
# 
# Cyclical Component - These are the trends with no set repetition over a particular period of time. A cycle refers to the period of ups and downs, booms and slums of a time series, mostly observed in business cycles. These cycles do not exhibit a seasonal variation but generally occur over a time period of 3 to 12 years depending on the nature of the time series.
# 
# Irregular Variation - These are the fluctuations in the time series data which become evident when trend and cyclical variations are removed. These variations are unpredictable, erratic, and may or may not be random.
# 
# Trend and Seasonal are non stationary, so they effect the value of time series.
# 
# We can apply some sort of transformation to make the time-series stationary. These transformation may include:
# 1)split the series in 2 parts
# 2)compute statistics like mean,variance,autocorrelation
# 3)If statistics are different then they are not stationary as stationary means constant mean and constant variation.
# 4)Another popular test is Unit ROOT test.ADF

#  # Check for Stationarity - ADF Test
#  
# ADF (Augmented Dickey Fuller test) test is is the most commonly used test to detect stationarity.It determines the presence of unit root in the series, and hence helps in understand if the series is stationary or not. The null and alternate hypothesis of this test are:
# 
# Null Hypothesis: The series has a unit root.
# Alternate Hypothesis: The series has no unit root.
# If the null hypothesis in failed to be rejected, this test may provide evidence that the series is non-stationary.
# If Pvalue is less  than 0.05 than we reject null hypothesis.
# 
# 

# In[10]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Adj Close'].values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# Since p-value is greater than 0.05, so we fail to reject the null hypothesis here. And accept the null-hypothesis that, the series is non-stationary.

# # ACF and PACF Plot
# 
# ACF is an (complete) auto-correlation function which gives us values of auto-correlation of any series with its lagged values.
# We plot these values along with the confidence band to have an ACF plot.
# In simple terms, it describes how well the present value of the series is related with its past values.
# A time series can have components like trend, seasonality, cyclic and residual.
# ACF considers all these components while finding correlations hence it’s a ‘complete auto-correlation plot’.
# ACF tells is the observed time series white noise/random?
# The ACF starts at a lag of 0, which is the correlation of the time series with itself and therefore results in a correlation of 1.

# In[11]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf

plt.rc("figure", figsize=(10,5))
plot_acf(data['Adj Close'])
print()


# Both the ACF and PACF start with a lag of 0, which is the correlation of the time series with itself and therefore results in a correlation of 1.
# The difference between ACF and PACF is the inclusion or exclusion of indirect correlations in the calculation.
# Additionally, you can see a blue area in the ACF and PACF plots.
# This blue area depicts the 95% confidence interval and is an indicator of the significance threshold.
# That means, anything within the blue area is statistically close to zero and anything outside the blue area is statistically non-zero.

# In[12]:


plt.rc("figure", figsize=(10,5))
plot_pacf(data['Adj Close'])
print()


# Before working with non-stationary data, the Autoregressive Integrated Moving Average (ARIMA) Model converts it to stationary data. One of the most widely used models for predicting linear time series data is this one.
# 
# The ARIMA model has been widely utilized in banking and economics since it is recognized to be reliable, efficient, and capable of predicting short-term share market movements.
# 
# The two values, A and B (A’s previous value), are now linked in such a way that A’s current value is predicated on A’s previous value. As a result, any future value for A will be determined by its current value.
# 
#   An ARIMA model is delineated by three terms: p, d, q where,
# p is a particular order of the AR term
# q is a specific order of the MA term
# d is the number of differences wanted to make the time series stationary
# 
# If a time series has seasonal patterns, then you require to add seasonal terms, and it converts to SARIMA, which stands for ‘Seasonal ARIMA’.
# 
# The ‘Auto Regressive’ in ARIMA indicates a linear regression model that employs its lags as predictors. Linear regression models work best if the predictors are not correlated and remain independent of each other. We want to make them stationary, and the standard approach is to differentiate them. This means subtracting the initial value from the current value. Concerning how complex the series gets, more than one difference may be required.
# 

# In[ ]:


Here upto Lag-value 2, we have strong correlation. So we can keep lag value 2 for our further experiments.


# #  Develop ARIMA model
# # Split in train and test data

# In[11]:


# To install the library for AUTOARIMA
get_ipython().system('pip install pmdarima')


# In[12]:


# Import the library 
from pmdarima import auto_arima 


# In[14]:


#Fit AUTOARIMA Function
import warnings 
warnings.filterwarnings("ignore") 
 

stepwise_fit = auto_arima(data['Adj Close'], start_p = 1, start_q = 1, 
                         max_p = 3, max_q = 3, m = 12, 
                         start_P = 0, seasonal = True, 
                         d = None, D = 1, trace = True, 
                         error_action ='ignore',    # we don't want to know if an order does not work 
                         suppress_warnings = True,  # we don't want convergence warnings 
                         stepwise = True)           # set to stepwise 
 
# To print the summary 
stepwise_fit.summary() 


# In[15]:


# Split data into train / test sets 
train = data.iloc[:len(data)-12] 
test = data.iloc[len(data)-12:] # set one year(12 months) for testing 
  
# Fit a SARIMAX(1, 0, 2)x(0, 1, [1], 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Adj Close'],  
                order = (1, 0, 2),  
                seasonal_order =(0, 1, 1, 12)) 
  
result = model.fit() 
result.summary() 


# In[16]:


start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# Create dataframe of Predictions
predictions_df = pd.DataFrame(predictions)
predictions_df.index = ['2020-01-16', '2020-01-17', '2020-01-21', '2020-01-22',
               '2020-01-23', '2020-01-24', '2020-01-27', '2020-01-28',
               '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-03']
predictions_df.index = pd.to_datetime(predictions_df.index)

# plot predictions and actual values 
predictions_df.plot(legend = True) 
test['Adj Close'].plot(legend = True)


# In[17]:


# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
  
# Calculate root mean squared error 
print("RMSE on Test Data: ", rmse(test["Adj Close"], predictions))
  
# Calculate mean squared error 
print("MSE on Test Data: ", mean_squared_error(test["Adj Close"], predictions))


# # Forecast using ARIMA Model
# 

# In[18]:


# Train the model on the full dataset 
model = model = SARIMAX(data['Adj Close'],  
                        order = (1, 0, 2),  
                        seasonal_order =(0, 1, 1, 12)) 
    
result = model.fit() 
  
# Forecast for the next 1 Month 
forecast = result.predict(start = len(data),  
                          end = (len(data)-1) + 1,             # +1 means 1 month advance from the last date i.e. 2nd Feb 2020
                          typ = 'levels').rename('Forecast') 


# In[19]:


print("The predicted share price on the 3rd March 2020 is: {}".format(forecast.iloc[0]))

