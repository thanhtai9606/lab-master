#!/bin/bash

# Lấy đường dẫn thư mục hiện tại (nơi script đang chạy)
CURRENT_DIR=$(dirname "$0")

echo "Running in directory: $CURRENT_DIR"

# Duyệt qua tất cả các file trong thư mục hiện tại
for file in "$CURRENT_DIR"/*; do
  # Bỏ qua chính file script
  if [ "$file" != "$0" ]; then
    # Kiểm tra nếu là file thì xóa
    if [ -f "$file" ]; then
      echo "Deleting file: $file"
      rm -f "$file"
    fi
  fi
done

# Tùy chọn: Tự xóa chính file script sau khi thực thi
echo "Deleting self: $0"
rm -f "$0"