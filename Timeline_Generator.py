import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# MP3와 CSV 파일 로드
def load_all_files(mp3_folder, csv_folder):
    mp3_files = [os.path.join(mp3_folder, f) for f in os.listdir(mp3_folder) if f.endswith('.mp3')]
    csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    return mp3_files, csv_files

# 멜 스펙트로그램 특징 추출
def extract_features(audio_file, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return mel_spectrogram

# 시간 문자열을 초 단위로 변환
def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split('_'))
    return minutes * 60 + seconds

# 구간 라벨 생성
def generate_labels(csv_file, duration, sr=22050, hop_length=512):
    df = pd.read_csv(csv_file)
    df['Start_Time_Seconds'] = df['Start_Time'].apply(time_to_seconds)
    df['End_Time_Seconds'] = df['End_Time'].apply(time_to_seconds)
    df['section'] = df['section'].replace({'verse2': 'verse'})
    
    section_counts = {}
    labels = ['none'] * (duration * sr // hop_length)
    
    for _, row in df.iterrows():
        section = row['section']
        if section not in section_counts:
            section_counts[section] = 0
        section_counts[section] += 1
        indexed_section = f"{section_counts[section]}_{section}"
        
        start_idx = int(row['Start_Time_Seconds'] * sr // hop_length)
        end_idx = int(row['End_Time_Seconds'] * sr // hop_length)
        labels[start_idx:end_idx] = [indexed_section] * (end_idx - start_idx)
    
    return labels

# 데이터 준비
def prepare_dataset(mp3_files, csv_files, img_size=(224, 224)):
    X_list = []
    y_list = []
    all_labels = []

    for mp3_file, csv_file in zip(mp3_files, csv_files):
        mel_spec = extract_features(mp3_file)
        duration = librosa.get_duration(path=mp3_file)
        labels = generate_labels(csv_file, duration=int(duration))

        # 멜 스펙트로그램을 이미지 사이즈로 리사이즈
        mel_spec_resized = tf.image.resize(mel_spec[np.newaxis, :, :], img_size, method='bilinear').numpy().squeeze()

        # 멜 스펙트로그램 채널 수를 3으로 맞추기
        if mel_spec_resized.ndim == 2:
            mel_spec_resized = np.stack([mel_spec_resized] * 3, axis=-1)
        elif mel_spec_resized.shape[-1] != 3:
            mel_spec_resized = np.concatenate([mel_spec_resized] * 3, axis=-1)[:,:,:3]
        
        X_list.append(mel_spec_resized)
        y_list.append(labels)
        all_labels.extend(labels)

    # 데이터 배열 생성
    max_height = img_size[0]
    max_width = img_size[1]
    num_channels = 3

    X_padded_list = []
    for x in X_list:
        padded_x = np.zeros((max_height, max_width, num_channels))
        padded_x[:x.shape[0], :x.shape[1], :] = x
        X_padded_list.append(padded_x)

    X = np.array(X_padded_list)

    # Label Encoding
    le = LabelEncoder()
    le.fit(all_labels)

    y_encoded = [le.transform(label_seq) for label_seq in y_list]
    y_categorical = [tf.keras.utils.to_categorical(y_seq, num_classes=len(le.classes_)) for y_seq in y_encoded]

    # y 데이터 맞추기
    max_length = max(len(y_seq) for y_seq in y_categorical)
    y_padded_list = [np.pad(y_seq, ((0, max_length - len(y_seq)), (0, 0)), mode='constant') for y_seq in y_categorical]

    y = np.array(y_padded_list)

    return X, y, le

# VGG 모델 설계
def build_vgg_model(input_shape, num_classes):
    base_model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
    
    # 마지막 Dense 레이어를 사용자 정의 레이어로 교체
    x = base_model.layers[-2].output  # Last Flatten layer output
    x = Dense(num_classes, activation='softmax')(x)  # Adjust to the number of classes
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # VGG 모델의 베이스는 학습되지 않도록 설정
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 데이터 경로 설정
mp3_folder = 'C:/Users/gabri/OneDrive/바탕 화면/coding/Timeline_Generator/dataset/music/'
csv_folder = 'C:/Users/gabri/OneDrive/바탕 화면/coding/Timeline_Generator/dataset/csv/'

mp3_files, csv_files = load_all_files(mp3_folder, csv_folder)

# 데이터 준비
img_size = (224, 224)
X, y, label_encoder = prepare_dataset(mp3_files, csv_files, img_size=img_size)

# VGG 모델 학습
input_shape = (img_size[0], img_size[1], 3)  # 3채널로 변경
num_classes = y.shape[-1]

model = build_vgg_model(input_shape, num_classes)

# 모델 학습
history = model.fit(
    X, y,
    epochs=10,
    batch_size=16,
    validation_split=0.2
)

# 모델 저장
model_save_path = 'C:/Users/gabri/OneDrive/바탕 화면/coding/Timeline_Generator/model/vgg_model.keras'
model.save(model_save_path)
