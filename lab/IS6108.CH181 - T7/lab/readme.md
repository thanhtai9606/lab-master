
# Hướng dẫn cài đặt Apache Spark

## 1. Tải xuống Apache Spark

- Truy cập trang web chính thức của Apache Spark tại [Apache Spark Downloads](https://spark.apache.org/downloads.html).
- Chọn phiên bản Spark mà bạn muốn tải xuống.
- Chọn gói phân phối phù hợp với hệ điều hành của bạn (ví dụ: Pre-built for Apache Hadoop).

## 2. Giải nén Apache Spark

- Sau khi tải xuống, giải nén tệp tải về vào thư mục mong muốn trên máy của bạn.
- Ví dụ: 
  ```bash
  tar xvf spark-<version>-bin-hadoop<version>.tgz
  mv spark-<version>-bin-hadoop<version> /path/to/your/spark-directory
  ```

## 3. Thay thế đường dẫn đến thư mục Spark

- Đặt biến môi trường SPARK_HOME để trỏ tới thư mục Spark bạn vừa giải nén.
- Thêm dòng sau vào tệp cấu hình shell của bạn (ví dụ: `~/.bashrc` hoặc `~/.zshrc`):
  ```bash
  export SPARK_HOME=/path/to/your/spark-directory
  export PATH=$SPARK_HOME/bin:$PATH
  ```

- Sau đó, nạp lại tệp cấu hình shell:
  ```bash
  source ~/.bashrc
  ```

- Kiểm tra xem Spark đã được cài đặt đúng chưa bằng cách chạy lệnh sau:
  ```bash
  spark-submit --version
  ```

Bây giờ, bạn đã sẵn sàng sử dụng Apache Spark trên máy của mình!
