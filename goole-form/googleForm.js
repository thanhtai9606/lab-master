function getUnixTimestamp() {
    return Math.floor(Date.now() / 1000);  // Lấy Unix timestamp hiện tại
}

function generateRandomPhoneNumber() {
    const prefixes = ['09', '08', '07', '03'];
    const randomPrefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    const randomNumber = Math.floor(Math.random() * 100000000).toString().padStart(8, '0');
    return randomPrefix + randomNumber;
}

function getRandomRating() {
    const ratings = [8, 9, 10];
    return ratings[Math.floor(Math.random() * ratings.length)];
}

function getRandomEntryValue() {
    const values = ['42151', '48759', '39935'];
    return values[Math.floor(Math.random() * values.length)];
}

function isWeekend() {
    const today = new Date();
    const day = today.getDay();
    return day === 0 || day === 6;  // 0 là Chủ nhật, 6 là Thứ 7
}

function isWithinWorkingHours() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const start = 9 * 60; // 9:00 AM in minutes
    const end = 16 * 60 + 30; // 4:30 PM in minutes
    const currentTimeInMinutes = hours * 60 + minutes;

    return currentTimeInMinutes >= start && currentTimeInMinutes <= end;
}

function submitGoogleForm() {
    if (isWeekend() || !isWithinWorkingHours()) {
        console.log("Outside of working hours or it's a weekend. Skipping submission.");
        return;
    }

    const formUrl = 'https://docs.google.com/forms/d/e/1FAIpQLSc5cvJIJN97c6Y_UfT0kitcI6KeNJNbF7JldVjuLF7oyRerog/formResponse';

    const formData = {
        'entry.1582431947': getRandomEntryValue(),  // Chọn ngẫu nhiên một trong ba giá trị
        'entry.1591633300': generateRandomPhoneNumber(),  // Tạo số điện thoại ngẫu nhiên
        'entry.2029809369': '',
        'entry.2112121023': '5 (Rất Hài Lòng)',
        'entry.601701377': [
            'Thái độ phục vụ của cán bộ',
            'Trình độ chuyên môn, tư vấn của cán bộ',
            'Hình thức, tác phong của cán bộ'
        ].join(', '),  // Ghép các giá trị của entry.601701377 lại thành một chuỗi
        'entry.541726183': '7',
        'entry.1798553241': getRandomRating(),  // Random 8, 9 hoặc 10
        'dlut': getUnixTimestamp()  // Lấy Unix timestamp hiện tại
    };

    console.log(formData);

    // Gọi hàm để gửi form
    PostFormData(formUrl, formData);
}

function PostFormData(formUrl, formData) {
    const formBody = Object.keys(formData).map(key => encodeURIComponent(key) + '=' + encodeURIComponent(formData[key])).join('&');

    fetch(formUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: formBody
    })
    .then(response => {
        if (response.ok) {
            console.log('Form submitted successfully!');
        } else {
            console.log('Error submitting form');
        }
    })
    .catch(error => console.error('Error:', error));
}

let submitCount = 0;
const maxSubmissions = 21;

function startBackgroundSubmission() {
    if (submitCount >= maxSubmissions) {
        console.log("Reached maximum submissions for today.");
        return;
    }

    const randomDelay = Math.random() * (60 * 60 * 1000); // Thời gian delay ngẫu nhiên trong khoảng 1 giờ (0 đến 60 phút)

    setTimeout(() => {
        submitGoogleForm();
        submitCount++;
        startBackgroundSubmission();
    }, randomDelay);
}

function startDailySubmission() {
    if (!isWeekend()) {
        submitCount = 0;  // Reset số lần gửi vào mỗi ngày mới
        startBackgroundSubmission();  // Bắt đầu quy trình submit
    }
}

// Thiết lập kiểm tra mỗi ngày vào lúc nửa đêm để bắt đầu lại quy trình submit
setInterval(startDailySubmission, 24 * 60 * 60 * 1000); // Mỗi 24 giờ kiểm tra lại để bắt đầu

// Bắt đầu lần đầu tiên
startDailySubmission();
