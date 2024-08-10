import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# 모델 및 라벨 인코더 로드
model_path = 'C:/Users/gabri/OneDrive/바탕 화면/coding/Timeline_Generator/model/model_00.keras'
model = tf.keras.models.load_model(model_path)

# 라벨 인코더 설정
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['intro', 'verse', 'prechorus', 'chorus', 'interlude', 'bridge', 'outro'])

# 특징 추출 함수
def extract_features(audio_file, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=model.input_shape[2], axis=1)  # 길이 맞추기
    return mel_spectrogram

# 예측 함수
def predict_segments(audio_file):
    mel_spectrogram = extract_features(audio_file)
    mel_spectrogram = mel_spectrogram.reshape(1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1)
    predictions = model.predict(mel_spectrogram)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=-1)[0])
    return predicted_labels

# 시각화 함수
def visualize_predictions(audio_file):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    predicted_labels = predict_segments(audio_file)
    
    times = np.linspace(0, duration, len(predicted_labels))
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, predicted_labels, color='b')
    plt.title('Predicted Music Sections')
    plt.xlabel('Time (s)')
    plt.ylabel('Section')
    plt.xticks(np.arange(0, duration, step=10))
    plt.grid(True)
    plt.show()

# 음악 파일 경로
audio_file = 'C:/Users/gabri/OneDrive/바탕 화면/coding/Timeline_Generator/dataset/music/music.mp3'

# 시각화 실행
visualize_predictions(audio_file)
