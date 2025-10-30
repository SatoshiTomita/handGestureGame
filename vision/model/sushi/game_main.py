# game_main.py
import os, time, numpy as np, cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

from recognizer import SushiRecognizer, CFG
from referee import judge_cclemon
from graphics import concat_side_by_side, put_label_top, overlay_center_text, put_pose_label, draw_vs_bar, draw_pose_history
from camera_stream import CameraStream
from fsm import RoundFSM
from tts_async import TTSWorker

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

def make_speak_fn(style_id):
    def _speak(text):
        if not TTS_AVAILABLE or style_id is None: return
        wav = tts_wav_bytes(text, style_id, speed_scale=1.05)
        if wav: play_wav_bytes_winsound(wav)
    return _speak

def map_label_to_move(label: str):
    return {"CHARGE": "CHARGE", "ATTACK": "ATTACK", "GUARD": "DEFENSE"}.get(label) if label else None

def main():
    # Window
    win = "Sushi-Janken (2P)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    style_id = get_zundamon_style_id(ENGINE_URL, "ノーマル") if TTS_AVAILABLE else None
    tts = TTSWorker(make_speak_fn(style_id), timeout=0.9)

    # カメラ（非同期）
    left  = CameraStream(0, 960, 540).start()
    right = CameraStream(1, 960, 540).start()

    cfg = CFG()
    recL = SushiRecognizer(draw_debug=False, cfg=cfg)
    recR = SushiRecognizer(draw_debug=False, cfg=cfg)

    fsm = RoundFSM(sampling_sec=cfg.SAMPLING_SEC, best_of=3)
    power_L = power_R = 0

    # サンプリング状態
    from collections import deque
    histL = {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)}
    histR = {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)}
    decidedL = decidedR = None
    history=[]

    def reset_sampling():
        nonlocal histL, histR, decidedL, decidedR
        histL = {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)}
        histR = {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)}
        decidedL = decidedR = None

    def stable_label(h):
        g_now = len(h["g"])==cfg.HOLD_FRAMES and sum(h["g"])==cfg.HOLD_FRAMES
        a_now = len(h["a"])==cfg.HOLD_FRAMES and sum(h["a"])==cfg.HOLD_FRAMES
        c_now = len(h["c"])==cfg.HOLD_FRAMES and sum(h["c"])==cfg.HOLD_FRAMES
        if c_now: return "CHARGE"
        if a_now: return "ATTACK"
        if g_now: return "GUARD"
        return None

    # ゲーム開始アナウンス（非同期）
    tts.say("寿司じゃんけん、二人対戦をはじめるのだ！")
    fsm.start_round()

    def majority_label(h, min_ratio=0.6):
        n = max(len(h["g"]), len(h["a"]), len(h["c"]))
        if n == 0: return None
        scores = {"GUARD": sum(h["g"]), "ATTACK": sum(h["a"]), "CHARGE": sum(h["c"])}
        lab, sc = max(scores.items(), key=lambda x: x[1])
        return lab if sc >= int(np.ceil(n * min_ratio)) else None

    try:
        while True:
            ok1, f1 = left.read()
            ok2, f2 = right.read()
            if not ok1:
                f1 = 255 * np.ones((540, 960, 3), dtype="uint8")
            if not ok2:
                f2 = 255 * np.ones((540, 960, 3), dtype="uint8")

            concat, f1r, f2r = concat_side_by_side(f1, f2)

            # フェーズに応じた処理（ノンブロッキング）
            ph = fsm.phase.name

            if ph == "READY":
                put_label_top(f1r, "Ready (P1)")
                put_label_top(f2r, "Ready (P2)")
            elif ph == "GO":
                put_label_top(f1r, "GO (P1)")
                put_label_top(f2r, "GO (P2)")
            elif ph == "SAMPLE":
                # ここだけ認識を回す（毎フレーム）
                g,a,c,_ = recL.process_frame(f1)
                histL["g"].append(1 if g else 0)
                histL["a"].append(1 if a else 0)
                histL["c"].append(1 if c else 0)
                # 現在推定（瞬間値）
                currL = "CHARGE" if c else ("ATTACK" if a else ("GUARD" if g else None))
                labL = stable_label(histL)
                if labL is not None: decidedL = labL

                g,a,c,_ = recR.process_frame(f2)
                histR["g"].append(1 if g else 0)
                histR["a"].append(1 if a else 0)
                histR["c"].append(1 if c else 0)
                currR = "CHARGE" if c else ("ATTACK" if a else ("GUARD" if g else None))
                labR = stable_label(histR)
                if labR is not None: decidedR = labR

                put_label_top(f1r, f"P1 sampling...")
                put_label_top(f2r, f"P2 sampling...")
                # 表示: 現在のポーズと確定状態
                put_pose_label(f1r, currL if decidedL is None else decidedL, decided=(decidedL is not None))
                put_pose_label(f2r, currR if decidedR is None else decidedR, decided=(decidedR is not None))

            elif ph == "SHOW_RESULT":
                # SHOW_RESULT に入った瞬間に一度だけ判定
                if fsm.outcome_msg == "":
                    finalL = decidedL or majority_label(histL, 0.6)
                    finalR = decidedR or majority_label(histR, 0.6)
                    move_L = map_label_to_move(finalL)
                    move_R = map_label_to_move(finalR)
                    history.append((finalL, finalR))
                    outcome, power_L, power_R = judge_cclemon(move_L, move_R, power_L, power_R)

                    fsm.set_result_and_scores(outcome)
                    # TTSは非同期
                    say = {
                        "win": "プレイヤー1の攻撃が刺さったのだ！",
                        "lose": "プレイヤー2の攻撃が刺さったのだ！",
                        "draw-continue": "勝負は続行なのだ！",
                        "no-move": "手が見えなかったのだ。もう一回！",
                    }.get(outcome, "")
                    if say: tts.say(say)

                overlay_center_text(concat, "result!", 80)
                msg = {
                    "win": "P1 scores!",
                    "lose": "P2 scores!",
                    "draw-continue": "Draw",
                    "no-move": "No move detected"
                }.get(fsm.outcome_msg, "")
                overlay_center_text(concat, msg, 150, color=(60,60,60), thick=6)
                draw_vs_bar(concat, decidedL, decidedR, y=300, scale=1.1)
                # 最終ポーズ表示
                put_pose_label(f1r, decidedL, decided=True)
                put_pose_label(f2r, decidedR, decided=True)

            elif ph == "GAME_END":
                overlay_center_text(concat, "Game Over", 120, scale=2.0, color=(0,0,0), thick=6)

            # スコアとパワーは常時表示
            overlay_center_text(concat, f"{fsm.score_L} - {fsm.score_R}", 240, scale=1.8)
            put_label_top(f1r, f"Power {power_L}", y_ratio=0.08, scale=0.9, color=(0,120,255))
            put_label_top(f2r, f"Power {power_R}", y_ratio=0.08, scale=0.9, color=(255,120,0))

            # 表示（非ブロッキング）
            cv2.imshow("Sushi-Janken (2P)", concat)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

            prev = ph
            fsm.advance()
            if prev != fsm.phase.name:
                if fsm.phase.name == "READY":
                    tts.say("ゲーム始めるのだ！")
                    reset_sampling()
                    fsm.outcome_msg = ""
                elif fsm.phase.name == "GO":
                    tts.say("ぽん！")
                elif fsm.phase.name == "SAMPLE":
                    reset_sampling()
                elif fsm.phase.name == "IDLE":
                    fsm.start_round()

        # ループ終わり
    finally:
        tts.close()
        recL.close(); recR.close()
        left.release(); right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
