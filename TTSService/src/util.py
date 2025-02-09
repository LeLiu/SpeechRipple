import os
import io
import torch
import torchaudio

from pydub import AudioSegment, silence

from config import tts as tts_config

def get_project_path():
    current_file_path = os.path.abspath(__file__)
    src_path = os.path.dirname(current_file_path)
    project_path = os.path.dirname(src_path)
    return project_path


class AppPath:
    def __init__(self):
        self.project_dir = get_project_path()
        self.temp_dir = self.__get_abs_path('temp')
        self.model_dir = self.__get_abs_path('model')
        self.res_dir = self.__get_abs_path('res')

    def __get_abs_path(self, sub_path):
        return os.path.join(self.project_dir, sub_path)


app_paths = AppPath()


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio

def clip_audio_segment(audio_segment):
    # 1. try to find long silence for clipping
    non_silent_segments = silence.split_on_silence(
        audio_segment, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for segment in non_silent_segments:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + segment) > 15000:
            break
        non_silent_wave += segment

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 15000:
        non_silent_segments = silence.split_on_silence(
            audio_segment, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for segment in non_silent_segments:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + segment) > 15000:
                break
            non_silent_wave += segment

    # 3. if no proper silence found for clipping
    if len(non_silent_wave) > 15000:
        non_silent_wave = non_silent_wave[:15000]

    non_silent_wave = remove_silence_edges(non_silent_wave) + AudioSegment.silent(duration=50)

    return non_silent_wave

def preprocess_ref_audio(ref_audio_path):
    audio_seg = AudioSegment.from_file(ref_audio_path)
    audio_seg = clip_audio_segment(audio_seg)

    bytes_io = io.BytesIO()
    audio_seg.export(bytes_io, format='wav')
    audio, sr = torchaudio.load(bytes_io)
    bytes_io.close()

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < tts_config['target_rms']:
        audio = audio * tts_config['target_rms'] / rms
    if sr != tts_config['mel_spec']['target_sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, tts_config['mel_spec']['target_sample_rate'])
        audio = resampler(audio)
    return audio

def preprocess_ref_text(ref_text):
    ref_text = ref_text.strip()
    if not ref_text.endswith(".") and not ref_text.endswith("。"):
        ref_text += "."
    return ref_text