import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 오디오 파일 로드
audio_path = "C:/Users/gabri/OneDrive/바탕 화면/coding/Timeline_Generator/dataset/music/music_001.mp3"
y, sr = librosa.load(audio_path, sr=16000)

# 1. Mel-Spectrogram 계산
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 2. MFCC 계산
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 3. 비트 트래킹
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# 비트 트래킹의 시간 계산
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 시각화
plt.figure(figsize=(15, 10))

# 1. Mel-Spectrogram 시각화
plt.subplot(3, 1, 1)
librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram')

# 2. MFCC 시각화
plt.subplot(3, 1, 2)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')

# 3. 비트 트래킹 시각화
plt.subplot(3, 1, 3)
plt.plot(librosa.times_like(y, sr=sr), y, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
plt.title('Beat Tracking')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

# 레이아웃 조정 및 출력
plt.tight_layout()
plt.show()
