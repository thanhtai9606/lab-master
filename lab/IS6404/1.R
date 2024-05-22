# Load necessary libraries
library(zoo)  # For SMA calculation

# Read the GDP data from the file
gdp_data <- read.csv('./data/gdp.csv')

# Ensure the column you want to calculate SMA for is named 'GDP'
# Calculate the 3-year Simple Moving Average (SMA)
gdp_data$SMA_3 <- rollapply(gdp_data$GDP, width = 3, FUN = mean, fill = NA, align = 'right')

# Print the data to see the SMA column
print(gdp_data)

# Optionally, plot the GDP and its SMA
if ("graphics" %in% rownames(installed.packages())) {
  plot(gdp_data$Year, gdp_data$GDP, type = 'l', col = 'blue', xlab = 'Year', ylab = 'GDP', main = 'GDP and SMA')
  lines(gdp_data$Year, gdp_data$SMA_3, col = 'red')
  legend("topleft", legend = c("GDP", "SMA 3 Years"), col = c("blue", "red"), lty = 1)
}
