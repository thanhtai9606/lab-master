
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
python3 audio_processing.py --noise wiener --silence energy
python3 audio_processing.py --noise median --silence energy
python3 audio_processing.py --noise spectral --silence none


python3 audio_processing.py --noise spectral,wiener,kalman --silence none
```

# Setup phần Fine-tune

Vào thư mục `./fine-tune`

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

*Note: Trong lúc thử nghiệm, code được chạy trên Google Colab*