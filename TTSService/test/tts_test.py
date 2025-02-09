from ttsinfer import TTSInferencer
from util import preprocess_ref_audio
import soundfile as sf

from speaker import speakers



def test_tts():
    vocoder_path = '../model/vocos-mel-24khz'
    model_path = '../model/f5tts-base/model_1200000.safetensors'
    vocab_path = '../model/vocab.txt'

    device = 'cuda:0'
    tts_inferencer = TTSInferencer(vocoder_path, model_path, vocab_path, device)

    ref_audio = preprocess_ref_audio('../res/ref_audio/ref_doubao_wrtz.wav')
    ref_text = '各位听众朋友们，最新消息，今天我国在科技 领域取得重大突破，新型量子计算机研发成功，将为大数据处理带来革新。'
    gen_text = '您好，现在室内温度是二十五摄氏度。'
    speed = 0.8

    wav = tts_inferencer.infer(ref_audio, ref_text, gen_text,speed)

    print(wav)

    with open('./test.wav', "wb") as f:
        sf.write(f.name, wav, 24000)

def test_tts_with_speaker():
    vocoder_path = '../model/vocos-mel-24khz'
    model_path = '../model/f5tts-base/model_1200000.safetensors'
    vocab_path = '../model/vocab.txt'

    device = 'cuda:0'
    tts_inferencer = TTSInferencer(vocoder_path, model_path, vocab_path, device)
    speaker = speakers['默认女声']
    gen_text = '您好，现在室内温度是二十五摄氏度。'
    speed = 0.8

    wav = tts_inferencer.infer(speaker.audio, speaker.text, gen_text, speed)

    with open('./test_tts_with_speaker.wav', "wb") as f:
        sf.write(f.name, wav, 24000)

def main():
    test_tts_with_speaker()

if __name__ == '__main__':
    main()