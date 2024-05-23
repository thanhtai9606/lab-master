import pandas as pd

# Load the data
data = pd.read_csv('./data/gdp.csv')

# Calculate the 3-year SMA
data['SMA_3'] = data['GDP'].rolling(window=3).mean()

# Display the data with the SMA_3 column
sma_data = data[['Year', 'GDP', 'SMA_3']]
print(sma_data.tail())  # Show the last few rows to include some of the calculated SMA values
