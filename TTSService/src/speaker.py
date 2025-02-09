import os

from util import preprocess_ref_audio, preprocess_ref_text
from util import app_paths

def _get_ref_paths_by_name(name):
    ref_audio_dir = os.path.join(app_paths.res_dir, 'ref_audio')
    ref_audio_path = os.path.join(ref_audio_dir, f'ref_{name}.wav')
    ref_text_path = os.path.join(ref_audio_dir, f'ref_{name}.txt')
    return ref_audio_path, ref_text_path


class Speaker:
    def __init__(self, name, speed=1.0):
        self._name = name
        self._audio = None
        self._text = None
        self._speed = speed

    @property
    def name(self):
        return self._name

    @property
    def audio(self):
        if self._audio is None:
            self._load_audio()
        return self._audio

    @property
    def text(self):
        if self._text is None:
            self._load_text()
        return self._text

    @property
    def speed(self):
        return self._speed

    def _load_text(self):
        _, ref_text_path = _get_ref_paths_by_name(self._name)
        with open(ref_text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self._text = preprocess_ref_text(text)

    def _load_audio(self):
        ref_audio_path, _ = _get_ref_paths_by_name(self._name)
        self._audio = preprocess_ref_audio(ref_audio_path)

speakers = {'默认女声': Speaker('doubao_wrtz', speed=0.8)}


