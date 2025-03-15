import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import subprocess
from tqdm import tqdm
from pykalman import KalmanFilter

# ===================== 1️⃣ Chuyển đổi âm thanh =====================
def convert_to_wav(input_audio, output_wav, target_sr=16000):
    """
    Chuyển đổi file âm thanh bất kỳ sang WAV PCM 16-bit mono.
    """
    try:
        command = ["ffmpeg", "-y", "-i", input_audio, "-ac", "1", "-ar", str(target_sr), "-sample_fmt", "s16", output_wav]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi {input_audio} sang WAV: {e}")

def convert_audio_back(input_wav, output_audio):
    """
    Chuyển file WAV đã xử lý về định dạng ban đầu.
    """
    try:
        command = ["ffmpeg", "-y", "-i", input_wav, output_audio]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi {input_wav} về {output_audio}: {e}")

# ===================== 2️⃣ Lọc Nhiễu =====================
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
    Áp dụng phương pháp lọc nhiễu.
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
def remove_silence(input_wav, output_wav):
    """
    Loại bỏ khoảng lặng bằng WebRTC VAD.
    """
    try:
        command = ["ffmpeg", "-i", input_wav, "-af", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB", output_wav]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Lỗi khi loại bỏ khoảng lặng {input_wav}: {e}")

# ===================== 4️⃣ Xử Lý Toàn Bộ File Âm Thanh =====================
def process_audio_folder(input_folder, output_folder, noise_method="spectral", remove_silence_flag=False):
    """
    Xử lý tất cả các file âm thanh: Chuyển đổi, lọc nhiễu, loại bỏ khoảng lặng.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "m4a"))]

    for file in tqdm(files, desc=f"Processing files (Noise: {noise_method})"):
        input_path = os.path.join(input_folder, file)
        original_ext = file.split(".")[-1]
        wav_path = os.path.join(output_folder, file.replace(original_ext, "wav"))
        denoised_path = os.path.join(output_folder, "denoised_" + file.replace(original_ext, "wav"))
        final_output_path = os.path.join(output_folder, "cleaned_" + file)

        # Chuyển sang WAV nếu không phải file WAV
        if original_ext != "wav":
            convert_to_wav(input_path, wav_path)
        else:
            wav_path = input_path  # Nếu đã là WAV, giữ nguyên

        # Kiểm tra file hợp lệ
        if os.path.getsize(wav_path) < 44:
            print(f"❌ Lỗi: File {file} có kích thước quá nhỏ hoặc không hợp lệ!")
            continue

        # Đọc file WAV
        y, sr = librosa.load(wav_path, sr=None)

        # Lọc nhiễu
        y_denoised = apply_noise_reduction(y, sr, noise_method)
        sf.write(denoised_path, y_denoised, sr)

        # Loại bỏ khoảng lặng (nếu có)
        if remove_silence_flag:
            remove_silence(denoised_path, final_output_path)
            os.remove(denoised_path)
        else:
            os.rename(denoised_path, final_output_path)

        # Chuyển ngược lại về định dạng ban đầu nếu cần
        if original_ext != "wav":
            convert_audio_back(final_output_path, final_output_path.replace("cleaned_", ""))
            os.remove(final_output_path)

        if original_ext != "wav":
            os.remove(wav_path)  # Xóa file WAV trung gian nếu gốc không phải WAV

    print(f"✅ Xử lý hoàn tất! Kết quả lưu tại: {output_folder}")

# ===================== 5️⃣ Chạy Chương Trình =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc nhiễu và loại bỏ khoảng lặng cho file âm thanh.")
    
    # Thêm tham số noise với đầy đủ phương pháp, bao gồm "kalman"
    parser.add_argument("--noise", type=str, default="spectral", choices=["spectral", "wiener", "median", "lms", "kalman"],
                        help="Phương pháp lọc nhiễu: spectral, wiener, median, lms, kalman")

    # Thêm tham số silence
    parser.add_argument("--silence", type=str, default="vad", choices=["vad", "energy", "none"],
                        help="Phương pháp loại bỏ khoảng lặng: vad, energy, hoặc 'none' nếu không muốn bỏ khoảng lặng")

    args = parser.parse_args()
    remove_silence_flag = args.silence != "none"

    process_audio_folder("./data/fpt", "./data/fpt_output", args.noise, remove_silence_flag)


    process_audio_folder("./data/fpt", "./data/fpt_output", args.noise, remove_silence_flag)

