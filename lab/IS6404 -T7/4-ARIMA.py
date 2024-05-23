import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('./data/gdp.csv')

# Hiển thị dữ liệu để kiểm tra
print(data.head())
print(data.info())

# Chuyển đổi dữ liệu GDP thành chuỗi thời gian
# Định dạng index ngày tháng dựa trên thông tin năm
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Phân tích chuỗi thời gian bằng mô hình ARIMA
# Lựa chọn tham số p, d, q cho mô hình ARIMA
model = ARIMA(data['GDP'], order=(1, 1, 1))  # Tham số order=(p, d, q) cần được lựa chọn dựa trên ACF và PACF
fit_model = model.fit()

# Dự báo cho 5 năm tiếp theo
forecast = fit_model.get_forecast(steps=5)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Hiển thị kết quả dự báo
print('Dự báo GDP:')
print(forecast_mean)

# Vẽ biểu đồ dự báo
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['GDP'], label='Historical GDP')
plt.plot(pd.date_range(data.index[-1], periods=6, freq='A')[1:], forecast_mean, label='Forecasted GDP', color='red')
plt.fill_between(pd.date_range(data.index[-1], periods=6, freq='A')[1:], conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('GDP Forecast Using ARIMA')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.show()
