# game_main.py
import os, time, numpy as np, cv2
from recognizer import SushiRecognizer, CFG
from referee import judge_cclemon
from graphics import (
    concat_side_by_side, put_label_top, overlay_center_text,
    put_pose_label, draw_vs_bar, draw_charge_icons
)
from camera_stream import CameraStream
from fsm import RoundFSM
from tts_async import TTSWorker
from effect import load_image, overlay_icon_anchored, overlay_icon_center

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

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
    # ====== Window / Assets =====================================================
    win = "Sushi-Janken (2P)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    BASE_DIR = os.path.dirname(__file__)
    IMG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "images"))
    CHARGE_PATH = os.path.join(IMG_DIR, "charge.png")
    SHIELD_PATH = os.path.join(IMG_DIR, "goldshield.png")
    BEAM_PATH   = os.path.join(IMG_DIR, "beam.webp")

    charge_icon = load_image(CHARGE_PATH, height=72)
    shield_icon = load_image(SHIELD_PATH, height=72)
    beam_icon   = load_image(BEAM_PATH,   height=72)

    style_id = get_zundamon_style_id(ENGINE_URL, "ノーマル") if TTS_AVAILABLE else None
    tts = TTSWorker(make_speak_fn(style_id), timeout=0.9)

    # ====== Cameras (async) =====================================================
    left  = CameraStream(0, 960, 540).start()
    right = CameraStream(1, 960, 540).start()

    # ====== Recognizer / FSM ====================================================
    cfg = CFG()
    recL = SushiRecognizer(draw_debug=False, cfg=cfg)
    recR = SushiRecognizer(draw_debug=False, cfg=cfg)

    fsm = RoundFSM(sampling_sec=cfg.SAMPLING_SEC, best_of=3)
    power_L = power_R = 0

    # ====== Sampling buffers ====================================================
    from collections import deque
    histL = {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)}
    histR = {"g": deque(maxlen=cfg.HOLD_FRAMES), "a": deque(maxlen=cfg.HOLD_FRAMES), "c": deque(maxlen=cfg.HOLD_FRAMES)}
    decidedL = decidedR = None
    history = []

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

    def majority_label(h, min_ratio=0.6):
        n = max(len(h["g"]), len(h["a"]), len(h["c"]))
        if n == 0: return None
        scores = {"GUARD": sum(h["g"]), "ATTACK": sum(h["a"]), "CHARGE": sum(h["c"])}
        lab, sc = max(scores.items(), key=lambda x: x[1])
        return lab if sc >= int(np.ceil(n * min_ratio)) else None

    # ====== Result effect hold parameters ======================================
    EFFECT_HOLD_SEC = 1.2       # 表示を維持する秒数（好みで調整）
    result_show_until = 0.0     # この時刻までSHOW_RESULTを継続
    final_pt_L = None           # SHOW_RESULT用 固定座標
    final_pt_R = None
    prev_finalL = None          # SHOW_RESULT用 固定ラベル
    prev_finalR = None
    last_pt_L = None            # SAMPLEで得た最新手座標（パネル座標系）
    last_pt_R = None

    # ====== Start ==============================================================
    tts.say("寿司じゃんけん、二人対戦をはじめるのだ！")
    fsm.start_round()

    try:
        while True:
            ok1, f1 = left.read()
            ok2, f2 = right.read()
            if not ok1: f1 = 255 * np.ones((540, 960, 3), dtype="uint8")
            if not ok2: f2 = 255 * np.ones((540, 960, 3), dtype="uint8")

            concat, f1r, f2r = concat_side_by_side(f1, f2)
            Hc, Wc = concat.shape[:2]
            left_panel  = concat[:, : Wc//2]
            right_panel = concat[:, Wc//2 :]

            ph = fsm.phase.name

            if ph == "READY":
                put_label_top(f1r, "Ready (P1)")
                put_label_top(f2r, "Ready (P2)")

            elif ph == "GO":
                put_label_top(f1r, "GO (P1)")
                put_label_top(f2r, "GO (P2)")

            elif ph == "SAMPLE":
                # 認識
                gL,aL,cL,infoL = recL.process_frame(f1)
                gR,aR,cR,infoR = recR.process_frame(f2)

                histL["g"].append(1 if gL else 0)
                histL["a"].append(1 if aL else 0)
                histL["c"].append(1 if cL else 0)
                histR["g"].append(1 if gR else 0)
                histR["a"].append(1 if aR else 0)
                histR["c"].append(1 if cR else 0)

                currL = "CHARGE" if cL else ("ATTACK" if aL else ("GUARD" if gL else None))
                currR = "CHARGE" if cR else ("ATTACK" if aR else ("GUARD" if gR else None))
                labL = stable_label(histL);  labR = stable_label(histR)
                if labL is not None: decidedL = labL
                if labR is not None: decidedR = labR

                put_label_top(f1r, "P1 sampling...")
                put_label_top(f2r, "P2 sampling...")
                put_pose_label(f1r, currL if decidedL is None else decidedL, decided=(decidedL is not None))
                put_pose_label(f2r, currR if decidedR is None else decidedR, decided=(decidedR is not None))

                # 手座標を「パネル座標系」に変換して保存
                fxL = f1r.shape[1] / float(f1.shape[1]); fyL = f1r.shape[0] / float(f1.shape[0])
                fxR = f2r.shape[1] / float(f2.shape[1]); fyR = f2r.shape[0] / float(f2.shape[0])

                ptL = infoL.get("hand_xy") if infoL else None
                ptR = infoR.get("hand_xy") if infoR else None
                if ptL: last_pt_L = (int(ptL[0] * fxL), int(ptL[1] * fyL))
                if ptR: last_pt_R = (int(ptR[0] * fxR), int(ptR[1] * fyR))

            elif ph == "SHOW_RESULT":
                # 初回だけ判定＆固定化
                if fsm.outcome_msg == "":
                    finalL = decidedL or majority_label(histL, 0.6)
                    finalR = decidedR or majority_label(histR, 0.6)
                    prev_finalL, prev_finalR = finalL, finalR

                    move_L = map_label_to_move(finalL)
                    move_R = map_label_to_move(finalR)
                    history.append((finalL, finalR))
                    outcome, power_L, power_R = judge_cclemon(move_L, move_R, power_L, power_R)

                    fsm.set_result_and_scores(outcome)

                    # 表示キープの締切時刻と固定座標を決定
                    result_show_until = time.time() + EFFECT_HOLD_SEC
                    final_pt_L = last_pt_L
                    final_pt_R = last_pt_R

                    say = {
                        "win": "プレイヤー1の攻撃が刺さったのだ！",
                        "lose": "プレイヤー2の攻撃が刺さったのだ！",
                        "draw-continue": "勝負は続行なのだ！",
                        "no-move": "手が見えなかったのだ。もう一回！",
                    }.get(outcome, "")
                    if say: tts.say(say)

                # 毎フレーム、同じラベル＆座標で描画
                def draw_final(panel, label, pt, side):
                    if not label: return
                    if pt:
                        if label == "CHARGE": overlay_icon_center(panel, charge_icon, *pt)
                        elif label == "GUARD": overlay_icon_center(panel, shield_icon, *pt)
                        elif label == "ATTACK": overlay_icon_center(panel, beam_icon, *pt)
                    else:
                        anch = "left" if side == "L" else "right"
                        y = 90
                        if label == "CHARGE": overlay_icon_anchored(panel, charge_icon, anchor=anch, y=y)
                        elif label == "GUARD": overlay_icon_anchored(panel, shield_icon, anchor=anch, y=y)
                        elif label == "ATTACK": overlay_icon_anchored(panel, beam_icon, anchor=anch, y=y)

                draw_final(left_panel,  prev_finalL, final_pt_L, "L")
                draw_final(right_panel, prev_finalR, final_pt_R, "R")

                overlay_center_text(concat, "result!", 80)
                msg = {
                    "win": "P1 scores!",
                    "lose": "P2 scores!",
                    "draw-continue": "Draw",
                    "no-move": "No move detected"
                }.get(fsm.outcome_msg, "")
                overlay_center_text(concat, msg, 150, color=(60,60,60), thick=6)
                draw_vs_bar(concat, prev_finalL, prev_finalR, y=300, scale=1.1)
                put_pose_label(f1r, prev_finalL, decided=True)
                put_pose_label(f2r, prev_finalR, decided=True)

            elif ph == "GAME_END":
                overlay_center_text(concat, "Game Over", 120, scale=2.0, color=(0,0,0), thick=6)

            # ====== 常時表示（スコア/パワー等） ==================================
            overlay_center_text(concat, f"{fsm.score_L} - {fsm.score_R}", 240, scale=1.8)
            put_label_top(f1r, f"Power {power_L}", y_ratio=0.08, scale=0.9, color=(0,120,255))
            put_label_top(f2r, f"Power {power_R}", y_ratio=0.08, scale=0.9, color=(255,120,0))
            draw_charge_icons(left_panel,  count=power_L, max_count=5, anchor="left",  y=40, radius=10, gap=10)
            draw_charge_icons(right_panel, count=power_R, max_count=5, anchor="right", y=40, radius=10, gap=10)

            # ====== 表示（非ブロッキング） ======================================
            cv2.imshow(win, concat)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

            # ====== フェーズ進行制御（表示ホールド） =============================
            if fsm.phase.name == "SHOW_RESULT" and time.time() < result_show_until:
                # ここではフェーズを進めず、次フレームに回す
                continue

            prev_phase = ph
            fsm.advance()

            # SHOW_RESULTを抜けたときの後片付け
            if prev_phase == "SHOW_RESULT" and fsm.phase.name != "SHOW_RESULT":
                final_pt_L = final_pt_R = None
                prev_finalL = prev_finalR = None
                last_pt_L = last_pt_R = None

            # フェーズ遷移時の音声/初期化
            if prev_phase != fsm.phase.name:
                if fsm.phase.name == "READY":
                    tts.say("ゲーム始めるのだ！")
                    reset_sampling()
                    fsm.outcome_msg = ""
                elif fsm.phase.name == "GO":
                    tts.say("ぽん！")
                    fsm.outcome_msg = ""
                elif fsm.phase.name == "SAMPLE":
                    reset_sampling()
                elif fsm.phase.name == "IDLE":
                    fsm.start_round()

    finally:
        tts.close()
        recL.close(); recR.close()
        left.release(); right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
