import requests
from config import http as http_config

# test http with wave_type:url
def test_tts_http_01():
    # 构建测试数据
    test_data = {
        "speaker_id": "默认女声",
        "gen_text": "您好，现在室内温度是二十五摄氏度。",
        "wave_type": "url"
    }

    # 发送 POST 请求到 HTTP 服务
    response = requests.post(f"http://{http_config['host']}:{http_config['port']}/tts/", json=test_data)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析响应数据
        data = response.json()
        wave_url = data["wave"]

        print(wave_url)

        # 下载语音文件
        wave_response = requests.get(wave_url)

        # 检查语音文件下载是否成功
        if wave_response.status_code == 200:
            # 保存语音文件
            wave_path ='./test_tts_http_01.wav'
            with open(wave_path, "wb") as f:
                f.write(wave_response.content)
            print(f"语音文件已保存到: {wave_path}")
        else:
            print(f"下载语音文件失败，状态码: {wave_response.status_code}")
    else:
        print(f"HTTP 请求失败，状态码: {response.status_code}")

# test http with wave_type:base64
def test_tts_http_02():
    import base64
    # 构建测试数据
    test_data = {
        "speaker_id": "默认女声",
        "gen_text": "很高兴认识你，我是你的智能助手桃子，有什么事情都可以问我哦。",
        "wave_type": "base64"
    }

    # 发送 POST 请求到 HTTP 服务
    response = requests.post(f"http://{http_config['host']}:{http_config['port']}/tts/", json=test_data)

    if response.status_code == 200:
        # 解析响应数据
        data = response.json()
        base64_data = data["wave"].encode('utf-8')
        wave_data = base64.b64decode(base64_data)
        wave_path = './test_tts_http_02.wav'
        with open(wave_path, "wb") as f:
            f.write(wave_data)
        print(f"语音文件已保存到: {wave_path}")
    else:
        print(f"HTTP 请求失败，状态码: {response.status_code}")


def test_http_status():
    response = requests.get(f"http://{http_config['host']}:{http_config['port']}/status/")

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"HTTP 请求失败，状态码: {response.status_code}")

def main():
    test_tts_http_01()
    test_tts_http_02()
    test_http_status()

if __name__ == '__main__':
    main()

