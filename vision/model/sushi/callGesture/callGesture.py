# callGesture.py
import time
import json
import random
import requests
import winsound
import tempfile
import os

ENGINE_URL = "http://127.0.0.1:50021"  # VOICEVOX ENGINE のURL
INTERVAL_SEC = 5                       # 5秒ごとに呼びかけ

LINES = [
    "ポーズ、いくよ！ はい、決めるのだ！",
    "今のうちにポーズをとるのだ！",
    "もう一回いくのだ。せーの、ポーズ！",
    "カメラ見て、キメポーズするのだ！",
    "準備はいい？ ポーズ、スタートなのだ！",
]

def get_zundamon_style_id(engine_url=ENGINE_URL, prefer_style="ノーマル"):
    try:
        resp = requests.get(f"{engine_url}/speakers")
        resp.raise_for_status()
        for sp in resp.json():
            if sp.get("name") == "ずんだもん":
                styles = sp.get("styles", [])
                for st in styles:
                    if st.get("name") == prefer_style:
                        return st.get("id")
                if styles:
                    return styles[0].get("id")
        return None
    except Exception as e:
        print(f"[ERROR] /speakers 取得失敗: {e}")
        return None

def tts_wav_bytes(text, style_id, engine_url=ENGINE_URL, speed_scale=1.0):
    try:
        q = requests.post(
            f"{engine_url}/audio_query",
            params={"text": text, "speaker": style_id},
        )
        q.raise_for_status()
        query = q.json()
        query["speedScale"] = speed_scale

        s = requests.post(
            f"{engine_url}/synthesis",
            params={"speaker": style_id},
            data=json.dumps(query),
            headers={"Content-Type": "application/json"},
        )
        s.raise_for_status()
        return s.content  # WAV bytes
    except Exception as e:
        print(f"[ERROR] 音声合成失敗: {e}")
        return None

def play_wav_bytes_winsound(wav_bytes):
    """
    winsoundでWAVを再生。まず SND_MEMORY を試し、
    失敗する環境ではテンポラリに書き出して再生。
    """
    # フラグを安全に取得（無ければ合理的なデフォルトにフォールバック）
    SND_MEMORY   = getattr(winsound, "SND_MEMORY",   0x0004)      # 通常は存在
    SND_FILENAME = getattr(winsound, "SND_FILENAME", 0x00020000)  # 通常は存在
    SND_SYNC     = getattr(winsound, "SND_SYNC",     0)           # 無ければ0=同期
    # 必要なら非同期にしたいときは SND_ASYNC を足す
    # SND_ASYNC  = getattr(winsound, "SND_ASYNC", 0x0001)

    try:
        # メモリ再生（対応環境ならこちらが最速）
        winsound.PlaySound(wav_bytes, SND_MEMORY | SND_SYNC)
    except Exception:
        # フォールバック：一時ファイル経由
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_bytes)
            path = f.name
        try:
            winsound.PlaySound(path, SND_FILENAME | SND_SYNC)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

def main():
    style_id = get_zundamon_style_id()
    if style_id is None:
        print("ずんだもんのスタイルが見つかりません。VOICEVOX ENGINE を起動し、ずんだもんが利用可能か確認してください。")
        return

    print(f"ずんだもん style_id: {style_id}")
    print("5秒ごとに呼びかけます。停止は Ctrl+C。")

    try:
        while True:
            text = random.choice(LINES)
            print(f"[CALL] {text}")
            wav = tts_wav_bytes(text, style_id, speed_scale=1.05)
            if wav:
                play_wav_bytes_winsound(wav)
            time.sleep(INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n停止しました。")

if __name__ == "__main__":
    main()
