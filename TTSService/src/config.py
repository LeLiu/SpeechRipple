from f5_tts.model import DiT

tts = dict(
    target_rms = 0.1,
    cross_fade_duration = 0.15,
    ode_method = "euler",
    nfe_step = 32,  # 16, 32
    cfg_strength = 2.0,
    sway_sampling_coef = -1.0,
    speed = 1.0,
    fix_duration = None,

    vocoder_path='model/vocos-mel-24khz',
    model_path='model/f5tts-base/model_1200000.safetensors',
    vocab_path='model/vocab.txt',

    mel_spec = dict(
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        target_sample_rate = 24000,
        mel_spec_type = "vocos",
    ),

    model = dict(
        cls = DiT,
        cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),

    ),

    device = 'cuda:0',
)

http = dict(
    host = '127.0.0.1',
    port = 8000,
    wave_dir = 'temp/gen_wave'
)