<?php
if (isset($_POST['uploadBtn']) && isset($_FILES['uploadedFile'])) {
    $uploadDir = 'uploads/';  // Thư mục để lưu file upload
    $fileName = basename($_FILES['uploadedFile']['name']);
    $targetFilePath = $uploadDir . $fileName;

    // Kiểm tra và tạo thư mục nếu chưa tồn tại
    if (!is_dir($uploadDir)) {
        mkdir($uploadDir, 0777, true);
    }

    // Kiểm tra lỗi khi upload
    if ($_FILES['uploadedFile']['error'] === UPLOAD_ERR_OK) {
        // Kiểm tra kích thước file (ví dụ: giới hạn 5MB)
        if ($_FILES['uploadedFile']['size'] <= 5 * 1024 * 1024) {
            // Di chuyển file tạm sang thư mục đích
            if (move_uploaded_file($_FILES['uploadedFile']['tmp_name'], $targetFilePath)) {
                echo "File uploaded successfully: <a href='$targetFilePath'>$fileName</a>";
            } else {
                echo "Error moving the file.";
            }
        } else {
            echo "File size exceeds 5MB limit.";
        }
    } else {
        echo "Error uploading the file. Error code: " . $_FILES['uploadedFile']['error'];
    }
} else {
    echo "No file uploaded.";
}
?>
