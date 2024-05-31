import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Đọc dữ liệu từ file CSV
data = pd.read_csv('./data/gdp.csv')

# Định dạng ngày tháng cho cột năm và đặt làm chỉ mục
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Vẽ biểu đồ ACF
plt.figure(figsize=(12, 6))
plot_acf(data['GDP'], lags=20)  # Sử dụng 20 lags để quan sát
plt.title('Autocorrelation Function for GDP')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()
