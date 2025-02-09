from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel

import soundfile as sf
import os
import uuid
import base64
import io

from config import http as http_config
from config import tts as tts_config
from util import app_paths
from ttsinfer import TTSInferencer
from speaker import speakers

app = FastAPI()

# 定义请求体模型
class TTSRequest(BaseModel):
    speaker_id: str
    gen_text: str
    wave_type: str
    speed: float = None

# 初始化 TTSInferencer

tts_inferencer = TTSInferencer()

# 定义路由处理函数
@app.post('/tts/')
async def tts(request: TTSRequest):
    speaker_id = request.speaker_id
    gen_text = request.gen_text
    wave_type = request.wave_type
    speed = request.speed  # 获取 speed 参数

    if wave_type not in ["base64", "url"]:
        raise HTTPException(status_code=400, detail="Invalid wave_type. Only 'base64' and 'url' are supported.")

    # 根据 speaker_id 获取 speaker 对象
    speaker = speakers.get(speaker_id)
    if speaker is None:
        raise HTTPException(status_code=404, detail="Speaker not found.")

    if speed is None:
        speed = speaker.speed

    data = tts_inferencer.infer(speaker.audio, speaker.text, gen_text, speed)
    sr = tts_config['mel_spec']['target_sample_rate']

    if wave_type == 'base64':
        bytes_io = io.BytesIO()
        bytes_io.name = 'bytes_io.wav'
        sf.write(bytes_io, data, sr)
        bytes_io.seek(0)
        wave = bytes_io.read()
        bytes_io.close()

        wave_base64 = base64.b64encode(wave).decode('utf-8')
        return {'wave': wave_base64}
    else:
        wave_dir = http_config['wave_dir']
        wave_dir = os.path.join(app_paths.project_dir, wave_dir)
        os.makedirs(wave_dir, exist_ok=True)

        wave_file_name = f'{uuid.uuid4()}.wav'
        wave_path = os.path.join(str(wave_dir), wave_file_name)
        sf.write(wave_path, data, sr)

        wave_url = f'http://{http_config["host"]}:{http_config["port"]}/wave/{wave_file_name}'
        return {'wave': wave_url}

@app.get('/wave/{wave_file_name}')
async def get_wave(wave_file_name: str):
    wave_dir = http_config['wave_dir']
    wave_dir = os.path.join(app_paths.project_dir, wave_dir)
    wave_path = os.path.join(str(wave_dir), wave_file_name)

    if not os.path.exists(wave_path):
        raise HTTPException(status_code=404, detail="Wave file not found.")

    # 返回语音文件
    return FileResponse(wave_path, media_type='audio/wav')

@app.get('/status')
async def get_status():
    return {'status': 'ok'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=http_config['host'], port=http_config['port'])
