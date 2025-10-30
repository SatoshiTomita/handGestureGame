# game_main.py
import random
import time
import cv2
import numpy as np

from recognizer import SushiRecognizer, CFG
from referee import judge_cclemon

# ====== TTS（任意） ===========================================================
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

# ====== ユーティリティ =========================================================
def overlay_center_text(img, text, y, scale=1.6, color=(0,0,0), thick=5):
    H, W = img.shape[:2]
    (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def put_label_top(img, text, y_ratio=0.12, scale=1.2, color=(0,255,255), thick=3):
    H, W = img.shape[:2]
    (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2
    y = int(H * y_ratio)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def show_overlay_preview_dual(cap1, cap2, win_title: str, text_left: str, text_right: str, ms: int = 1500):
    """左右のプレビューにテキストを重ね、結合して1ウィンドウで表示"""
    t_end = time.time() + (ms / 1000.0)
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    while time.time() < t_end:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()

        if not ok1:
            f1 = 255 * np.ones((720, 1280, 3), dtype="uint8")
        if not ok2:
            f2 = 255 * np.ones((720, 1280, 3), dtype="uint8")

        # 同じ高さに合わせて横結合
        h = 480
        def _resize_keep_ar(img, h):
            H, W = img.shape[:2]
            new_w = int(W * (h / H))
            return cv2.resize(img, (new_w, h))
        f1r = _resize_keep_ar(f1, h)
        f2r = _resize_keep_ar(f2, h)

        # 左右にラベル
        put_label_top(f1r, text_left)
        put_label_top(f2r, text_right)

        # ガイド線（中央仕切り）
        concat = cv2.hconcat([f1r, f2r])
        Hc, Wc = concat.shape[:2]
        cv2.line(concat, (Wc//2, 0), (Wc//2, Hc), (0,0,0), 4)

        cv2.imshow(win_title, concat)
        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
            break

def dual_sample_labels(cap1, cap2, rec1: SushiRecognizer, rec2: SushiRecognizer, cfg: CFG):
    """
    両カメラから短時間サンプルして、左右プレイヤーの安定ラベルを返す。
    return: (label_left, label_right or None) それぞれ None の可能性あり
    """
    t_end = time.time() + cfg.SAMPLING_SEC

    from collections import deque
    hist = {
        "L": {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)},
        "R": {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)},
    }

    def _label_from_flags(guard_now, attack_now, charge_now):
        if charge_now: return "CHARGE"
        if attack_now: return "ATTACK"
        if guard_now:  return "GUARD"
        return None

    decided = {"L": None, "R": None}

    while time.time() < t_end:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not ok1 and not ok2:
            break

        if ok1:
            g, a, c, _ = rec1.process_frame(f1)
            hist["L"]["g"].append(1 if g else 0)
            hist["L"]["a"].append(1 if a else 0)
            hist["L"]["c"].append(1 if c else 0)
            g_now = len(hist["L"]["g"])==cfg.HOLD_FRAMES and sum(hist["L"]["g"])==cfg.HOLD_FRAMES
            a_now = len(hist["L"]["a"])==cfg.HOLD_FRAMES and sum(hist["L"]["a"])==cfg.HOLD_FRAMES
            c_now = len(hist["L"]["c"])==cfg.HOLD_FRAMES and sum(hist["L"]["c"])==cfg.HOLD_FRAMES
            lab = _label_from_flags(g_now, a_now, c_now)
            if lab is not None: decided["L"] = lab

        if ok2:
            g, a, c, _ = rec2.process_frame(f2)
            hist["R"]["g"].append(1 if g else 0)
            hist["R"]["a"].append(1 if a else 0)
            hist["R"]["c"].append(1 if c else 0)
            g_now = len(hist["R"]["g"])==cfg.HOLD_FRAMES and sum(hist["R"]["g"])==cfg.HOLD_FRAMES
            a_now = len(hist["R"]["a"])==cfg.HOLD_FRAMES and sum(hist["R"]["a"])==cfg.HOLD_FRAMES
            c_now = len(hist["R"]["c"])==cfg.HOLD_FRAMES and sum(hist["R"]["c"])==cfg.HOLD_FRAMES
            lab = _label_from_flags(g_now, a_now, c_now)
            if lab is not None: decided["R"] = lab

        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
            return "__quit__", "__quit__"

    return decided["L"], decided["R"]

def map_label_to_move(label: str):
    # recognizerの "GUARD" を審判用 "DEFENSE" に寄せる
    return {"CHARGE": "CHARGE", "ATTACK": "ATTACK", "GUARD": "DEFENSE"}.get(label) if label else None

# ====== 本体 ===================================================================
def main():
    style_id = get_zundamon_style_id(ENGINE_URL, "ノーマル") if TTS_AVAILABLE else None

    # カメラを2台オープン（番号は環境に合わせて変えてOK）
    cap_left  = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    # 解像度（お好みで）
    for cap in (cap_left, cap_right):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("2台のカメラをオープンできませんでした。接続やカメラ番号を確認してください。")
        if cap_left.isOpened():  cap_left.release()
        if cap_right.isOpened(): cap_right.release()
        return

    if style_id: tts_say("寿司じゃんけん、二人対戦をはじめるのだ！", style_id)
    show_overlay_preview_dual(cap_left, cap_right, "Sushi-Janken (2P)", "welcome (Player1)", "welcome (Player2)", ms=2000)

    cfg = CFG()
    rec_left  = SushiRecognizer(draw_debug=False, cfg=cfg)
    rec_right = SushiRecognizer(draw_debug=False, cfg=cfg)

    score_L = score_R = 0
    power_L = power_R = 0

    try:
        while score_L < 2 and score_R < 2:
            # 掛け声 & 準備
            if style_id: tts_say("パンパン、レモン！", style_id, speed=1.05)
            show_overlay_preview_dual(cap_left, cap_right, "Sushi-Janken (2P)", "Pose now (P1)", "Pose now (P2)", ms=1300)
            if style_id: tts_say("ぽん！", style_id, speed=1.15)
            show_overlay_preview_dual(cap_left, cap_right, "Sushi-Janken (2P)", "GO (P1)", "GO (P2)", ms=500)

            # 同時サンプリング
            label_L, label_R = dual_sample_labels(cap_left, cap_right, rec_left, rec_right, cfg)
            if label_L == "__quit__" or label_R == "__quit__":
                break

            move_L = map_label_to_move(label_L)
            move_R = map_label_to_move(label_R)

            # 判定（左を“user”、右を“cpu”の引数に対応させて再利用）
            outcome, new_power_L, new_power_R = judge_cclemon(move_L, move_R, power_L, power_R)
            power_L, power_R = new_power_L, new_power_R

            # 表示用フレーム取得
            ok1, f1 = cap_left.read()
            ok2, f2 = cap_right.read()
            if not ok1:
                f1 = 255 * np.ones((720, 1280, 3), dtype="uint8")
            if not ok2:
                f2 = 255 * np.ones((720, 1280, 3), dtype="uint8")

            h = 480
            def _resize_keep_ar(img, h):
                H, W = img.shape[:2]
                new_w = int(W * (h / H))
                return cv2.resize(img, (new_w, h))
            f1r = _resize_keep_ar(f1, h)
            f2r = _resize_keep_ar(f2, h)

            # 各プレイヤーの手・パワー
            put_label_top(f1r, f"P1: {move_L or 'unknown'} / Power {power_L}", y_ratio=0.10, scale=1.0, color=(0,120,255))
            put_label_top(f2r, f"P2: {move_R or 'unknown'} / Power {power_R}", y_ratio=0.10, scale=1.0, color=(255,120,0))

            # 勝敗
            concat = cv2.hconcat([f1r, f2r])
            Hc, Wc = concat.shape[:2]
            cv2.line(concat, (Wc//2, 0), (Wc//2, Hc), (0,0,0), 4)

            overlay_center_text(concat, "result！", 80)
            y_msg = 150
            if outcome == "win":
                score_L += 1
                overlay_center_text(concat, "P1 scores!", y_msg, color=(0,200,0), thick=6)
            elif outcome == "lose":
                score_R += 1
                overlay_center_text(concat, "P2 scores!", y_msg, color=(0,0,200), thick=6)
            elif outcome == "draw-continue":
                overlay_center_text(concat, "Draw", y_msg, color=(60,60,60), thick=6)
            else:
                overlay_center_text(concat, "No move detected", y_msg, color=(60,60,60), thick=6)

            overlay_center_text(concat, f"{score_L} - {score_R}", 240, scale=1.8)

            cv2.imshow("Sushi-Janken (2P)", concat)
            cv2.waitKey(600)

            if style_id:
                say = {
                    "win": "プレイヤー1の攻撃が刺さったのだ！",
                    "lose": "プレイヤー2の攻撃が刺さったのだ！",
                    "draw-continue": "勝負は続行なのだ！",
                    "no-move": "手が見えなかったのだ。もう一回！",
                }[outcome]
                tts_say(say, style_id)
            time.sleep(0.4)

    finally:
        rec_left.close()
        rec_right.close()
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
