import os
import torch

from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

# import modules from F5-TTS.
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin

from config import tts as tts_config
from src.util import app_paths


def _load_vocos(vocos_path, device):
    config_path = os.path.join(vocos_path, 'config.yaml')
    model_path = os.path.join(vocos_path, 'pytorch_model.bin')

    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)

    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    return vocoder

def _load_vocoder(vocoder_path, device):
    # now only support 'vocos'.
    return _load_vocos(vocoder_path, device)


def _load_model(model_path, vocab_path, device, use_ema=True):
    transformer_cls = tts_config['model']['cls']
    transformer_cfg = tts_config['model']['cfg']
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, tokenizer='custom')
    transformer_net = transformer_cls(
        **transformer_cfg, text_num_embeds=vocab_size, mel_dim=tts_config['mel_spec']['n_mel_channels'])

    model = CFM(transformer=transformer_net,
                mel_spec_kwargs=tts_config['mel_spec'],
                odeint_kwargs=dict(method=tts_config['ode_method']),
                vocab_char_map=vocab_char_map
                ).to(device)

    dtype = torch.float32
    if tts_config['mel_spec']['mel_spec_type'] == "bigvgan":
        if "cuda" in device and torch.cuda.get_device_properties(device).major >= 6:
            dtype = torch.float16
    model = model.to(dtype)

    ckpt_type = model_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(model_path, device=device)
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}

        # Load EMA weights
        checkpoint["model_state_dict"] = {}
        for k, v in checkpoint["ema_model_state_dict"].items():
            if k not in ["initted", "step"]:
                checkpoint["model_state_dict"][k.replace("ema_model.", "")] = v

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


class TTSInferencer:
    _vocoder_path = os.path.join(app_paths.project_dir, tts_config['vocoder_path'])
    _model_path = os.path.join(app_paths.project_dir, tts_config['model_path'])
    _vocab_path = os.path.join(app_paths.project_dir, tts_config['vocab_path'])
    _device = tts_config['device']

    def __init__(self, vocoder_path=None, model_path=None, vocab_path=None, device=None):
        self._vocoder_path = vocoder_path if vocoder_path is not None else TTSInferencer._vocoder_path
        self._model_path = model_path if model_path is not None else TTSInferencer._model_path
        self._vocab_path = vocab_path if vocab_path is not None else TTSInferencer._vocab_path
        self._device = device if device is not None else TTSInferencer._device

        self._vocoder = _load_vocoder(self._vocoder_path, self._device)
        self._model = _load_model(self._model_path, self._vocab_path, self._device)


    def infer(self, ref_audio, ref_text, gen_text, speed=1.0):
        text = ' '.join([ref_text, gen_text])
        text_list = convert_char_to_pinyin([text])

        hop_length = tts_config['mel_spec']['hop_length']
        ref_audio_len = ref_audio.shape[-1] // hop_length

        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))

        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        with torch.inference_mode():
            generated, _ = self._model.sample(
                cond=ref_audio.to(self._device),
                text=text_list,
                duration=duration,
                steps=tts_config['nfe_step'],
                cfg_strength=tts_config['cfg_strength'],
                sway_sampling_coef=tts_config['sway_sampling_coef'],
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)

            generated_wave = self._vocoder.decode(generated_mel_spec) # only support 'vocos' vocoder.

            rms = torch.sqrt(torch.mean(torch.square(generated_wave)))
            if rms < tts_config['target_rms']:
                generated_wave = generated_wave * rms / tts_config['target_rms']

            generated_wave = generated_wave.squeeze().cpu().numpy()

            return generated_wave


