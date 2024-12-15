import librosa
import soundfile as sf
import os
import json
from tqdm import tqdm

import numpy as np

def trim_zeros_1d(arr):
    """
    1D numpy 배열에서 뒤쪽의 0들을 제거.
    Args:
        arr (numpy.ndarray): 입력 1D 배열
    Returns:
        numpy.ndarray: 뒤쪽의 0이 제거된 배열
    """
    # 0이 아닌 마지막 인덱스 찾기
    last_nonzero_idx = np.nonzero(arr)[0][-1] if np.any(arr) else -1
    # 0 제거
    return arr[:last_nonzero_idx + 1] if last_nonzero_idx != -1 else np.array([])


def convert_wav_librosa(input_path, target_sample_rate=16000):
    # 파일 로드 (원본 샘플링 레이트로)
    audio, sr = librosa.load(input_path, sr=None, mono=False)  # stereo 유지
    
    # 모노로 변환
    audio_mono = librosa.to_mono(audio)
    
    # 샘플링 레이트 변경
    audio_resampled = librosa.resample(audio_mono, orig_sr=sr, target_sr=target_sample_rate)

    return audio_resampled
    

if __name__ == "__main__":
    splits = ["Training", "Validation"]
    for split in splits:
        audio_dir = f"./data/{split}/wav/"
        text_dir = f"./data/{split}/label/"
        os.makedirs(f"./data/{split}/array/", exist_ok=True)
        for file_name in tqdm(os.listdir(audio_dir)):
            if file_name.endswith(".wav"):
                audio_path = os.path.join(audio_dir, file_name)
                audio_array = convert_wav_librosa(audio_path)
                audio_array = trim_zeros_1d(audio_array)
                if audio_array.size == 0 or np.isnan(audio_array).any():
                    continue
                array_file_name = file_name.replace(".wav", ".npy")
                with open(os.path.join(f"./data/{split}/array/", array_file_name), 'wb') as f:
                    np.save(f, audio_array)
