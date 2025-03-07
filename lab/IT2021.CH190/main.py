import os
import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import wave
import struct
from tqdm import tqdm

# ===================== Lọc Nhiễu bằng Spectral Subtraction =====================
def spectral_subtraction(y, sr, noise_estimate=None, alpha=1.2):
    """
    Lọc nhiễu trong tín hiệu âm thanh bằng phương pháp Spectral Subtraction.
    :param y: Tín hiệu âm thanh
    :param sr: Sample rate
    :param noise_estimate: Dự đoán phổ nhiễu
    :param alpha: Hệ số giảm nhiễu
    :return: Tín hiệu đã lọc nhiễu
    """
    n_fft = 2048
    hop_length = 512
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    if noise_estimate is None:
        noise_frames = int(0.5 * sr / hop_length)  # Lấy 0.5 giây đầu tiên làm ước lượng nhiễu
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    magnitude_denoised = np.maximum(magnitude - alpha * noise_estimate, 0)

    D_denoised = magnitude_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(D_denoised, hop_length=hop_length)

    return y_denoised

# ===================== Loại Bỏ Khoảng Lặng bằng WebRTC VAD =====================
def remove_silence(input_wav, output_wav, aggressiveness=3):
    """
    Loại bỏ khoảng lặng bằng WebRTC VAD.
    :param input_wav: File âm thanh đầu vào
    :param output_wav: File đầu ra sau khi xử lý
    :param aggressiveness: Mức độ nhạy của VAD (0-3)
    """
    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(input_wav, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    frame_duration = 30  # 30ms mỗi frame
    frame_size = int(sample_rate * frame_duration / 1000) * width
    frames = [frames[i:i + frame_size] for i in range(0, len(frames), frame_size)]

    voiced_frames = [frame for frame in frames if vad.is_speech(frame, sample_rate)]

    with wave.open(output_wav, "wb") as wf_out:
        wf_out.setnchannels(channels)
        wf_out.setsampwidth(width)
        wf_out.setframerate(sample_rate)
        for frame in voiced_frames:
            wf_out.writeframes(frame)

# ===================== Xử Lý Toàn Bộ File Trong Thư Mục =====================
def process_audio_folder(input_folder, output_folder):
    """
    Xử lý tất cả các file .wav trong thư mục bằng cách lọc nhiễu và loại bỏ khoảng lặng.
    :param input_folder: Thư mục chứa file gốc
    :param output_folder: Thư mục lưu file đã xử lý
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    for file in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_folder, file)
        denoised_path = os.path.join(output_folder, "denoised_" + file)
        final_output_path = os.path.join(output_folder, "cleaned_" + file)

        # Đọc file âm thanh
        y, sr = librosa.load(input_path, sr=None)

        # Lọc nhiễu
        y_denoised = spectral_subtraction(y, sr)
        sf.write(denoised_path, y_denoised, sr)

        # Loại bỏ khoảng lặng
        remove_silence(denoised_path, final_output_path)

        # Xóa file trung gian (file chỉ lọc nhiễu)
        os.remove(denoised_path)

    print(f"✅ Đã xử lý xong tất cả các file! Kết quả lưu tại: {output_folder}")

# ===================== Chạy chương trình =====================
input_directory = "./data/wham/cv"  # Đổi thành thư mục chứa file âm thanh
output_directory = "./data/wham_output"  # Đổi thành thư mục để lưu kết quả

process_audio_folder(input_directory, output_directory)
