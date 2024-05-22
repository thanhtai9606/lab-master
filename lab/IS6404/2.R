# Tải các thư viện cần thiết
library(forecast)

# Đọc dữ liệu từ file CSV
data <- read.csv("./data/gdp.csv")

# Kiểm tra dữ liệu
print(head(data))
print(str(data))

# Tạo đối tượng chuỗi thời gian với tần số hàng năm
# Giả sử dữ liệu được thu thập hàng năm bắt đầu từ năm 1985
gdp_ts <- ts(data$GDP, start = 1985, frequency = 1)

# Áp dụng mô hình Holt-Winters với thành phần mùa vụ kiểu cộng
hw_model <- HoltWinters(gdp_ts)

# Dự báo 5 năm tiếp theo
hw_forecast <- forecast(hw_model, h = 5)

# In kết quả dự báo
print(hw_forecast)

# Vẽ biểu đồ dự báo
plot(hw_forecast)
lines(hw_forecast$fitted, col = 'red')
legend("topright", legend = c("Observed GDP", "Fitted Model"), col = c("black", "red"), lty = 1)

# Vẽ thêm biểu đồ dữ liệu gốc và dự báo
plot(gdp_ts, main = "GDP Time Series and Forecasts", xlab = "Year", ylab = "GDP")
lines(hw_forecast$mean, col = 'blue')
legend("topright", c("Observed", "Forecast"), col = c("black", "blue"), lty = 1)
