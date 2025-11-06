# guard_detect.py
import cv2
import math
from collections import deque
import mediapipe as mp

# =========================
# パラメータ（調整ポイント）
# =========================
HOLD_FRAMES = 3            # 何フレーム連続で成立したら確定表示するか

# ★★★ Attack（小指特化型: 手の形だけで判定） ★★★
ATTACK_PINKY_CLOSE_RATIO = 0.30  # 小指TIP間距離 / 手幅 <= しきい（小指が近い）
ATTACK_NON_PINKY_FAR_RATIO = 0.7 # 他3本TIPの平均距離 / 手幅 >= しきい（他が遠い）

# Charge（4本指のTIPを左右で近づける）
CHARGE_TIP_RATIO  = 0.45   # 4本のTIPペア距離の平均 / 手幅 <= しきい（近い）

DRAW_DEBUG = True          # デバッグ描画

# =========================
# ユーティリティ
# =========================
def to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def dist2d_lm(a, b):
    # 正規化座標間の距離計算
    return math.hypot(a.x - b.x, a.y - b.y)

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def palm_width(hand_lm):
    """手幅の基準: Index MCP(5) - Pinky MCP(17)。取れなければ Wrist(0) - Middle MCP(9)。"""
    if hand_lm and len(hand_lm) > 17:
        return dist2d_lm(hand_lm[5], hand_lm[17])
    return dist2d_lm(hand_lm[0], hand_lm[9])

# =========================
# MediaPipe 初期化
# =========================
# ★★★ mp_pose を削除 ★★★
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ★★★ mp_pose.Pose の初期化を削除 ★★★
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    hist_guard  = deque(maxlen=HOLD_FRAMES)
    hist_attack = deque(maxlen=HOLD_FRAMES)
    hist_charge = deque(maxlen=HOLD_FRAMES)
    hist_hands_detected = deque(maxlen=HOLD_FRAMES) # Guard判定に必要

    while True:
        ok, frame = cap.read()
        if not ok:
            print("カメラから映像を取得できません。")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        # res_pose = pose.process(img) を削除
        res_hands = hands.process(img)
        img.flags.writeable = True

        out = frame.copy()
        H, W = out.shape[:2]

        # ========== 判定変数初期化 ==========
        guard_frame  = False
        attack_frame = False
        charge_frame = False
        hands_detected_frame = False
        
        mean_norm_tip_dist = None 
        norm_pinky_tip_dist = None 
        mean_norm_non_pinky_tip_dist = None # Attack判定に必要

        # ========== Hands: Charge/Attack 判定に必要な値の計算 ==========
        if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
            left_lm = right_lm = None
            for hand_lm, handed in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                if handed.classification[0].label == 'Left':
                    left_lm = hand_lm.landmark
                else:
                    right_lm = hand_lm.landmark

                if DRAW_DEBUG:
                    mp_draw.draw_landmarks(
                        out, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(80,200,80), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(80,160,220), thickness=2)
                    )

            if left_lm is not None and right_lm is not None:
                hands_detected_frame = True # ★Guard判定に利用
                tip_ids = [8, 12, 16, 20] # Index(8), Middle(12), Ring(16), Pinky(20)
                
                # 手幅を正規化基準として使用
                base_left  = palm_width(left_lm)
                base_right = palm_width(right_lm)
                base = safe_mean([base_left, base_right])
                
                if base and base > 1e-6:
                    norm_dists_all = []
                    norm_dists_non_pinky = []
                    
                    for tid in tip_ids:
                        d = dist2d_lm(left_lm[tid], right_lm[tid])
                        norm_dists_all.append(d / base) # 全てのTIPの距離
                        
                        if tid == 20: # Pinky TIP ID
                            norm_pinky_tip_dist = d / base # Pinkyの正規化距離を保存
                        else:
                            norm_dists_non_pinky.append(d / base) # Pinky以外の距離を収集

                        if DRAW_DEBUG:
                            lx, ly = to_px(left_lm[tid], W, H)
                            rx, ry = to_px(right_lm[tid], W, H)
                            cv2.line(out, (lx, ly), (rx, ry), (60, 60, 255), 2)
                            
                    # Charge判定に使う平均距離
                    mean_norm_tip_dist = sum(norm_dists_all) / len(norm_dists_all)
                    
                    # Attack判定に使うPinky以外の平均距離
                    mean_norm_non_pinky_tip_dist = sum(norm_dists_non_pinky) / len(norm_dists_non_pinky)

        # ----------------------------------------
        # Charge/Attack/Guard 判定の確定
        # ----------------------------------------
        if mean_norm_tip_dist is not None and norm_pinky_tip_dist is not None:
            
            # --- 1. Charge 判定 ---
            charge_frame = (mean_norm_tip_dist <= CHARGE_TIP_RATIO)
            
            # --- 2. Attack 判定 (小指特化型) ---
            is_pinky_close = (norm_pinky_tip_dist <= ATTACK_PINKY_CLOSE_RATIO)
            is_non_pinky_far = (mean_norm_non_pinky_tip_dist >= ATTACK_NON_PINKY_FAR_RATIO)
            
            attack_frame = bool(is_pinky_close and is_non_pinky_far)
            
            # Chargeが成立した場合は、Attackを強制的にFalseにする (Charge優先)
            if charge_frame:
                attack_frame = False
            
            # --- 3. Guard 判定 ---
            # Guard判定: AttackでもChargeでもない（両手が検出されていることが前提）
            if not attack_frame and not charge_frame:
                guard_frame = True # 手が検出されていて、Attack/Charge条件を満たさない場合はGuardとする

        # ========== 時系列の平滑化 ==========
        hist_guard.append(1 if guard_frame else 0)
        hist_attack.append(1 if attack_frame else 0)
        hist_charge.append(1 if charge_frame else 0)

        guard_now  = (sum(hist_guard)  == HOLD_FRAMES)
        attack_now = (sum(hist_attack) == HOLD_FRAMES)
        charge_now = (sum(hist_charge) == HOLD_FRAMES)

        # 優先度: Charge > Attack > Guard
        label = None
        if charge_now:
            label = "CHARGE"
        elif attack_now:
            label = "ATTACK"
        elif guard_now:
            label = "GUARD"
        else:
            label = "Idle"

        # ========== 表示 ==========
        if label:
            cv2.putText(out, label, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
        else:
            cv2.putText(out, "Idle", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (100, 100, 100), 4)

        # デバッグ: 生のフレーム判定を表示
        dbg = f"G:{guard_frame}  A:{attack_frame}  C:{charge_frame}"
        if norm_pinky_tip_dist is not None and DRAW_DEBUG:
            dbg += f" | Pin:{norm_pinky_tip_dist:.2f}"
        if mean_norm_non_pinky_tip_dist is not None and DRAW_DEBUG:
            dbg += f" | NonPinAvg:{mean_norm_non_pinky_tip_dist:.2f}"
            
        cv2.putText(out, dbg, (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 50), 2)

        cv2.imshow("Guard/Attack/Charge Detection (Hands Only)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
