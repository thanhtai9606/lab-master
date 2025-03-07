import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import wave
import scipy.signal as signal
from tqdm import tqdm

# ===================== 1️⃣ Chuyển đổi âm thanh về PCM 16-bit Mono =====================
def convert_audio_to_pcm16_mono(input_wav, output_wav, target_sr=16000):
    """
    Chuyển đổi file âm thanh về PCM 16-bit mono.
    """
    y, sr = librosa.load(input_wav, sr=target_sr, mono=True)
    sf.write(output_wav, y, target_sr, subtype='PCM_16')

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
    else:
        print(f"❌ Lỗi: Phương pháp lọc nhiễu '{method}' không hợp lệ!")
        return y

# ===================== 3️⃣ Loại Bỏ Khoảng Lặng bằng WebRTC VAD =====================
def remove_silence(input_wav, output_wav, aggressiveness=3):
    """
    Loại bỏ khoảng lặng bằng WebRTC VAD.
    """
    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(input_wav, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()

        if channels != 1 or width != 2 or sample_rate not in [8000, 16000, 32000, 48000]:
            print(f"❌ Lỗi: {input_wav} không phải PCM 16-bit mono!")
            return

        frames = wf.readframes(wf.getnframes())

    frame_duration = 30  # 30ms mỗi frame
    frame_size = int(sample_rate * frame_duration / 1000) * width
    frames = [frames[i:i + frame_size] for i in range(0, len(frames), frame_size)]

    if not frames or len(frames[0]) != frame_size:
        print(f"❌ Lỗi: Frame không hợp lệ trong {input_wav}")
        return

    voiced_frames = []
    for frame in frames:
        if len(frame) == frame_size:
            try:
                if vad.is_speech(frame, sample_rate):
                    voiced_frames.append(frame)
            except Exception as e:
                print(f"⚠️ Lỗi xử lý frame: {e}, bỏ qua frame này.")

    if not voiced_frames:
        print(f"⚠️ Cảnh báo: Không tìm thấy giọng nói trong {input_wav}, giữ nguyên file gốc.")
        return

    with wave.open(output_wav, "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(sample_rate)
        for frame in voiced_frames:
            wf_out.writeframes(frame)

# ===================== 4️⃣ Xử Lý Toàn Bộ File Trong Thư Mục =====================
def process_audio_folder(input_folder, output_folder, method="spectral"):
    """
    Xử lý tất cả các file .wav trong thư mục bằng cách lọc nhiễu và loại bỏ khoảng lặng.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    for file in tqdm(files, desc=f"Processing files (Method: {method})"):
        input_path = os.path.join(input_folder, file)
        converted_path = os.path.join(output_folder, "converted_" + file)
        denoised_path = os.path.join(output_folder, "denoised_" + file)
        final_output_path = os.path.join(output_folder, "cleaned_" + file)

        # Chuyển đổi sang PCM 16-bit mono
        convert_audio_to_pcm16_mono(input_path, converted_path)

        # Đọc file âm thanh sau khi chuyển đổi
        y, sr = librosa.load(converted_path, sr=None)

        # Lọc nhiễu theo phương pháp đã chọn
        y_denoised = apply_noise_reduction(y, sr, method)
        sf.write(denoised_path, y_denoised, sr)

        # Loại bỏ khoảng lặng
        remove_silence(denoised_path, final_output_path)

        # Xóa file trung gian
        os.remove(converted_path)
        os.remove(denoised_path)

    print(f"✅ Đã xử lý xong tất cả các file! Kết quả lưu tại: {output_folder}")

# ===================== 5️⃣ Chạy chương trình =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc nhiễu và loại bỏ khoảng lặng cho file âm thanh.")
    parser.add_argument("--method", type=str, default="spectral", choices=["spectral", "wiener", "median"],
                        help="Phương pháp lọc nhiễu: spectral, wiener, median")

    args = parser.parse_args()
    
    input_directory = "./data/wham/cv"  # Thay đổi nếu cần
    output_directory = "./data/wham_output/cv"  # Thay đổi nếu cần

    if not os.path.exists(input_directory):
        print(f"❌ Thư mục {input_directory} không tồn tại. Hãy đặt các file .wav vào thư mục này!")
        exit()

    process_audio_folder(input_directory, output_directory, args.method)
