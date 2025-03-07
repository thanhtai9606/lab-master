import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import wave
import scipy.signal as signal
from tqdm import tqdm
from pykalman import KalmanFilter
import subprocess

def normalize_volume(input_wav, output_wav, target_db=-3):
    """
    Chuẩn hóa mức âm lượng của file âm thanh bằng SoX.
    :param input_wav: Đường dẫn file đầu vào
    :param output_wav: Đường dẫn file đầu ra sau khi chuẩn hóa
    :param target_db: Mức dB tối ưu (mặc định: -3dB)
    """
    try:
        command = ["sox", input_wav, output_wav, "gain", "-n", str(target_db)]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"❌ Lỗi khi chuẩn hóa âm lượng {input_wav}: {e}")

# ===================== 1️⃣ Chuyển đổi âm thanh về PCM 16-bit Mono =====================
def convert_audio_to_pcm16_mono(input_wav, output_wav, target_sr=16000):
    """
    Chuyển đổi file âm thanh về PCM 16-bit mono (bắt buộc cho WebRTC VAD).
    """
    y, sr = librosa.load(input_wav, sr=target_sr, mono=True)
    sf.write(output_wav, y, target_sr, subtype='PCM_16')  # Đảm bảo PCM 16-bit

# ===================== 2️⃣ Lọc Nhiễu bằng nhiều phương pháp =====================
def spectral_subtraction(y, sr, noise_estimate=None, alpha=1.2):
    """
    Lọc nhiễu bằng phương pháp Spectral Subtraction.
    """
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    if noise_estimate is None:
        noise_frames = int(0.5 * sr / hop_length)
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    magnitude_denoised = np.maximum(magnitude - alpha * noise_estimate, 0)
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    return librosa.istft(D_denoised, hop_length=hop_length)

def wiener_filter(y, sr):
    """
    Lọc nhiễu bằng bộ lọc Wiener.
    """
    return signal.wiener(y)

def median_filter(y, sr, kernel_size=3):
    """
    Lọc nhiễu bằng bộ lọc trung vị.
    """
    return signal.medfilt(y, kernel_size)

def adaptive_lms_filter(y, sr, mu=0.01):
    """
    Lọc nhiễu bằng thuật toán Least Mean Squares (LMS).
    """
    n = len(y)
    d = np.copy(y)  
    w = np.zeros(10)  
    for i in range(10, n):
        x = y[i-10:i]  
        e = d[i] - np.dot(w, x)  
        w += 2 * mu * e * x  
        y[i] = e
    return y

def kalman_filter(y, sr):
    """
    Lọc nhiễu bằng Kalman Filter.
    """
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    y_filtered, _ = kf.filter(y)
    return y_filtered.flatten()

def apply_noise_reduction(y, sr, method="spectral"):
    """
    Áp dụng phương pháp lọc nhiễu do người dùng chọn.
    """
    if method == "spectral":
        return spectral_subtraction(y, sr)
    elif method == "wiener":
        return wiener_filter(y, sr)
    elif method == "median":
        return median_filter(y, sr)
    elif method == "lms":
        return adaptive_lms_filter(y, sr)
    elif method == "kalman":
        return kalman_filter(y, sr)
    else:
        print(f"❌ Lỗi: Phương pháp lọc nhiễu '{method}' không hợp lệ!")
        return y

# ===================== 3️⃣ Loại Bỏ Khoảng Lặng =====================
def process_audio_folder(input_folder, output_folder, noise_method="spectral", remove_silence_flag=False):
    """
    Xử lý file âm thanh: Chuẩn hóa âm lượng, lọc nhiễu, (tùy chọn) loại bỏ khoảng lặng.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    for file in tqdm(files, desc=f"Processing files (Noise: {noise_method})"):
        input_path = os.path.join(input_folder, file)
        normalized_path = os.path.join(output_folder, "normalized_" + file)
        converted_path = os.path.join(output_folder, "converted_" + file)
        denoised_path = os.path.join(output_folder, "denoised_" + file)
        final_output_path = os.path.join(output_folder, "cleaned_" + file)

        # Kiểm tra nếu file rỗng hoặc có lỗi
        if os.path.getsize(input_path) < 44:
            print(f"❌ Lỗi: File {file} có kích thước quá nhỏ hoặc không hợp lệ!")
            continue

        # Chuẩn hóa mức âm lượng bằng SoX
        normalize_volume(input_path, normalized_path)

        # Chuyển đổi sang PCM 16-bit mono
        convert_audio_to_pcm16_mono(normalized_path, converted_path)

        # Đọc file âm thanh sau khi chuyển đổi
        y, sr = librosa.load(converted_path, sr=None)

        # Lọc nhiễu theo phương pháp đã chọn
        y_denoised = apply_noise_reduction(y, sr, noise_method)
        sf.write(denoised_path, y_denoised, sr)

        # Bỏ khoảng lặng nếu được yêu cầu
        if remove_silence_flag:
            remove_silence(denoised_path, final_output_path)
        else:
            os.rename(denoised_path, final_output_path)  # Giữ nguyên file đã lọc nhiễu

        os.remove(normalized_path)  # Xóa file trung gian
        os.remove(converted_path)

    print(f"✅ Đã xử lý xong tất cả các file! Kết quả lưu tại: {output_folder}")


# ===================== 5️⃣ Chạy chương trình =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc nhiễu và loại bỏ khoảng lặng trong file âm thanh.")
    parser.add_argument("--noise", type=str, default="spectral", choices=["spectral", "wiener", "median", "lms", "kalman"],
                        help="Phương pháp lọc nhiễu: spectral, wiener, median, lms, kalman")
    parser.add_argument("--silence", type=str, default="vad", choices=["vad", "energy", "none"],
                        help="Phương pháp loại bỏ khoảng lặng: vad, energy, hoặc 'none' nếu không muốn bỏ khoảng lặng")

    args = parser.parse_args()
    
    remove_silence_flag = args.silence != "none"  # Nếu chọn "none", không loại bỏ khoảng lặng

    process_audio_folder("./data/wham/cv", "./data/wham_output/cv", args.noise, remove_silence_flag)
