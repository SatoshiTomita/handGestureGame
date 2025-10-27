# game_main.py
import random
import time
import cv2
import numpy as np

from recognizer import SushiRecognizer, CFG
from referee import judge_cclemon

# （TTSは任意。ここではフォールバック内蔵の例）
try:
    from callGesture.callGesture import (
        get_zundamon_style_id, tts_wav_bytes, play_wav_bytes_winsound, ENGINE_URL,
    )
    TTS_AVAILABLE = True
except Exception:
    ENGINE_URL = "http://127.0.0.1:50021"
    TTS_AVAILABLE = False
    def get_zundamon_style_id(*a, **k): return None
    def tts_wav_bytes(*a, **k): return None
    def play_wav_bytes_winsound(*a, **k): pass

def tts_say(text, style_id, speed=1.05):
    if not TTS_AVAILABLE or style_id is None: return
    wav = tts_wav_bytes(text, style_id, speed_scale=speed)
    if wav: play_wav_bytes_winsound(wav)

def pick_cpu_move(cpu_power: int) -> str:
    if cpu_power <= 0: return random.choice(["CHARGE", "DEFENSE", "CHARGE"])
    return random.choice(["ATTACK", "DEFENSE", "CHARGE"])

def show_overlay_preview(cap, win_title: str, text: str, ms: int = 1500):
    """指定ミリ秒だけ、カメラ映像にテキストを重ねて表示"""
    t_end = time.time() + (ms / 1000.0)
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)  # 先にウィンドウを作る
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            break
        img = frame.copy()
        # 画面中央に表示
        H, W = img.shape[:2]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
        x = (W - tw) // 2
        y = int(H * 0.18)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow(win_title, img)
        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
            break
def main():
    style_id = get_zundamon_style_id(ENGINE_URL, "ノーマル") if TTS_AVAILABLE else None
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("カメラを開けませんでした。接続を確認してください。")
        return

    if style_id: tts_say("寿司じゃんけん、はじめるのだ！", style_id)
    show_overlay_preview(cap, "Sushi-Janken", "welcome", ms=2000)
    rec = SushiRecognizer(draw_debug=False, cfg=CFG())
    user_score = cpu_score = 0

    try:
        user_power = 0
        cpu_power  = 0
        while user_score < 2 and cpu_score < 2:
            if style_id: tts_say("パンパン、レモン！", style_id, speed=1.05)
            show_overlay_preview(cap, "Sushi-Janken", "Pose now", ms=1300)
            if style_id: tts_say("ぽん！", style_id, speed=1.15)
            show_overlay_preview(cap, "Sushi-Janken", "GO", ms=500)

            label = rec.sample_label(cap)
            if label == "__quit__":
                break

            user_move = {"CHARGE":"CHARGE","ATTACK":"ATTACK","GUARD":"DEFENSE"}.get(label)
            cpu_move  = pick_cpu_move(cpu_power)

            outcome, user_power, cpu_power = judge_cclemon(user_move, cpu_move, user_power, cpu_power)

            ok, frame = cap.read()
            if not ok:
                frame = 255 * (np.ones((720, 1280, 3), dtype="uint8"))
            img = frame.copy()

            def overlay_center_text(img, text, y, scale=1.6, color=(0,0,0), thick=5):
                H,W = img.shape[:2]
                (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                x=(W-tw)//2
                cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

            overlay_center_text(img, "result！", 120)
            overlay_center_text(img, f"You: {user_move or 'unknown'} / CPU: {cpu_move}", 220, scale=1.2, color=(0,120,255), thick=3)
            overlay_center_text(img, f"Power You:{user_power}  CPU:{cpu_power}", 300, scale=1.2)

            if outcome == "win":
                user_score += 1
                overlay_center_text(img, "You win", 400, color=(0,200,0), thick=6)
            elif outcome == "lose":
                cpu_score += 1
                overlay_center_text(img, "You lose", 400, color=(0,0,200), thick=6)
            else:
                overlay_center_text(img, "Draw", 400, color=(60,60,60), thick=6)

            overlay_center_text(img, f"{user_score} - {cpu_score}", 500, scale=1.8)
            cv2.imshow("Sushi-Janken", img)
            cv2.waitKey(600)

            if style_id:
                say = {
                    "win":"攻撃が刺さったのだ！あなたの勝ち！",
                    "lose":"相手の攻撃が刺さったのだ…！",
                    "draw-continue":"勝負は続行なのだ！",
                    "no-move":"手が見えなかったのだ。もう一回！",
                }[outcome]
                tts_say(f"{say}", style_id)

            time.sleep(0.4)
    finally:
        rec.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
