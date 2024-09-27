function getUnixTimestamp() {
    return Math.floor(Date.now() / 1000);  // Lấy Unix timestamp hiện tại
}

function generateRandomPhoneNumber() {
    const prefixes = ['09', '03'];
    const randomPrefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    const randomNumber = Math.floor(Math.random() * 100000000).toString().padStart(8, '0');
    return randomPrefix + randomNumber;
}
function getRandomRating() {
    const ratings = [8, 9, 10];
    return ratings[Math.floor(Math.random() * ratings.length)];
}

function submitGoogleForm() {
    const formUrl = 'https://docs.google.com/forms/d/e/1FAIpQLSc5cvJIJN97c6Y_UfT0kitcI6KeNJNbF7JldVjuLF7oyRerog/formResponse';

    const formData = {
        'entry.1582431947': '42151',
        'entry.1591633300': generateRandomPhoneNumber(),  // Tạo số điện thoại ngẫu nhiên
        'entry.2029809369': '',
        'entry.2112121023': '5 (Rất Hài Lòng)',
        'entry.601701377': [
            'Thái độ phục vụ của cán bộ',
            'Trình độ chuyên môn, tư vấn của cán bộ',
            'Hình thức, tác phong của cán bộ'
        ].join(', '),  // Ghép các giá trị của entry.601701377 lại thành một chuỗi
        'entry.541726183': '7',
        'entry.1798553241': getRandomRating(), // 8,9,10
        'dlut': getUnixTimestamp()  // Lấy Unix timestamp hiện tại
    };

    console.log(formData)

   //PostFormData(formUrl, formData);
}
function PostFormData(formUrl, formData)
{
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

function submitGoogleFormTest(){
        const formUrl = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLSc5cvJIJN97c6Y_UfT0kitcI6KeNJNbF7JldVjuLF7oyRerog/formResponse';
        const formData = {
            'entry.445127754': 'Chấp nhận mọi thử thách',
            'entry.1374041517': 'Có'
        };

        PostFormData(formUrl, formData);
}

// Gọi hàm để gửi form
// submitGoogleForm();