# Tải các thư viện cần thiết
library(forecast)

# Đọc dữ liệu từ file CSV
data <- read.csv("./data/gdp.csv")

# Hiển thị cấu trúc dữ liệu để xác nhận các cột và dữ liệu
print(head(data))
print(str(data))

# Chuyển đổi dữ liệu GDP thành chuỗi thời gian
gdp_ts <- ts(data$GDP, start = 1985, frequency = 1)  # Giả sử dữ liệu là hàng năm

# Phân tích chuỗi thời gian bằng mô hình ARIMA tự động
fit_arima <- auto.arima(gdp_ts)

# Dự báo cho 5 năm tiếp theo
forecast_arima <- forecast(fit_arima, h = 5)

# Hiển thị kết quả dự báo
print(forecast_arima)

# Vẽ biểu đồ dự báo
plot(forecast_arima)

# Thêm tiêu đề và nhãn cho biểu đồ
title(main = "Dự báo GDP bằng Mô Hình ARIMA", xlab = "Năm", ylab = "GDP")
legend("bottomleft", legend=c("Dự báo"), col=c("red"), lty=1, cex=0.8)
