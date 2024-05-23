from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# Load the GDP data
data = pd.read_csv('./data/gdp.csv')

# Assuming an annual seasonality
seasonal_periods = 4  # or the appropriate seasonality for your data

# Fit the Holt-Winters Additive Model
hw_additive_model = ExponentialSmoothing(
    data['GDP'],
    seasonal_periods=seasonal_periods,
    trend='add',
    seasonal='add'
).fit()

# Forecast the next few periods
forecast_years = 3  # Forecasting for 3 years into the future
hw_forecast = hw_additive_model.forecast(steps=forecast_years)

# Creating a DataFrame for the forecast
forecast_index = range(data['Year'].iloc[-1] + 1, data['Year'].iloc[-1] + forecast_years + 1)
forecast_df = pd.DataFrame({'Year': forecast_index, 'Forecasted_GDP': hw_forecast})

print(forecast_df)
