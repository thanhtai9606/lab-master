python3 audio_processing.py --method spectral  # Lọc nhiễu bằng Spectral Subtraction
python3 audio_processing.py --method wiener    # Lọc nhiễu bằng Wiener Filter
python3 audio_processing.py --method median    # Lọc nhiễu bằng Median Filter

# Cài đặt sox để chuẩn hoá âm lượng
```
brew install ffmpeg sox
```
# config
    input_directory = "./data/fpt"  # Thay đổi nếu cần
    output_directory = "./data/fpt_output"  # Thay đổi nếu cần

    
```
python3 audio_processing.py --noise spectral --silence vad
python3 audio_processing.py --noise kalman --silence energy
python3 audio_processing.py --noise spectral --silence none

