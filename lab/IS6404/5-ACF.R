# Tải thư viện cần thiết
if (!require("forecast")) install.packages("forecast", dependencies=TRUE)
library(forecast)
if (!require("tseries")) install.packages("tseries", dependencies=TRUE)
library(tseries)

# Đọc dữ liệu từ file CSV
data <- read.csv("./data/gdp.csv")

# Chuyển đổi dữ liệu thành chuỗi thời gian, giả sử dữ liệu là hàng năm
gdp_ts <- ts(data$GDP, start=c(1985), frequency=1)

# Vẽ biểu đồ ACF và PACF
acf(gdp_ts, main="Autocorrelation Function")
pacf(gdp_ts, main="Partial Autocorrelation Function")

# Kiểm tra tính tĩnh của chuỗi thời gian
adf_test_result <- adf.test(gdp_ts, alternative="stationary")

# Nếu chuỗi không tĩnh, lấy sai phân bậc 1
if (adf_test_result$p.value > 0.05) {
  gdp_ts <- diff(gdp_ts, differences=1)
  adf_test_result_diff <- adf.test(gdp_ts, alternative="stationary")
  print(paste("After differencing, p-value:", adf_test_result_diff$p.value))
}

# Fit mô hình ARIMA sau khi đã xác định chuỗi tĩnh
fit_arima <- auto.arima(gdp_ts)
print(summary(fit_arima))

# Dự báo 5 năm tiếp theo
forecast_arima <- forecast(fit_arima, h=5)
plot(forecast_arima)
title(main="ARIMA Forecast")
