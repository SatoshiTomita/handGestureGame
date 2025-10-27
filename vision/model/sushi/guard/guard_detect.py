# guard_detect.py
import cv2
import math
from collections import deque
import mediapipe as mp

# =========================
# パラメータ（調整ポイント）
# =========================
NEAR_FACE_RATIO = 0.9      # Guard: 鼻-手首 距離のしきい（肩幅×係数以内を“近い”とする）
HOLD_FRAMES = 3            # 何フレーム連続で成立したら確定表示するか（反応↔安定のトレードオフ）

# Attack（かめはめ波）
WRISTS_CLOSE_RATIO = 0.60  # 両手首同士がどの程度“近い”か（肩幅×係数以下）
HANDS_FAR_RATIO   = 1.20   # 両手が顔から十分“遠い”か（肩幅×係数以上）
ARM_EXT_RATIO     = 1.30   # 肩→手首距離が長い（腕が伸びている）目安（肩幅×係数以上）

# Charge（4本指のTIPを左右で近づける）
CHARGE_TIP_RATIO  = 0.45   # 4本のTIPペア距離の平均 / 手幅 <= しきい（大きいほど緩い）

DRAW_DEBUG = True          # デバッグ描画

# =========================
# ユーティリティ
# =========================
def to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def euclid_xy(a_xy, b_xy):
    return math.hypot(a_xy[0]-b_xy[0], a_xy[1]-b_xy[1])

def dist2d_lm(a, b):
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
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(
    static_image_mode=False,
    enable_segmentation=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose, mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    hist_guard  = deque(maxlen=HOLD_FRAMES)
    hist_attack = deque(maxlen=HOLD_FRAMES)
    hist_charge = deque(maxlen=HOLD_FRAMES)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("カメラから映像を取得できません。")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res_pose  = pose.process(img)
        res_hands = hands.process(img)
        img.flags.writeable = True

        out = frame.copy()
        H, W = out.shape[:2]

        # ========== Pose ランドマーク取得と可視化 ==========
        guard_frame  = False
        attack_frame = False
        charge_frame = False

        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            try:
                nose = lm[mp_pose.PoseLandmark.NOSE]
                l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                nose_xy = to_px(nose, W, H)
                lsh_xy  = to_px(l_sh, W, H)
                rsh_xy  = to_px(r_sh, W, H)
                lwr_xy  = to_px(l_wr, W, H)
                rwr_xy  = to_px(r_wr, W, H)

                # 肩幅
                shoulder_width = euclid_xy(lsh_xy, rsh_xy) if (lsh_xy and rsh_xy) else None

                # ===== Guard（両手首が顔の近く） =====
                if shoulder_width and shoulder_width > 0:
                    dl = euclid_xy(nose_xy, lwr_xy)
                    dr = euclid_xy(nose_xy, rwr_xy)
                    near_left  = dl <= shoulder_width * NEAR_FACE_RATIO
                    near_right = dr <= shoulder_width * NEAR_FACE_RATIO
                    guard_frame = (near_left and near_right)

                # ===== Attack（かめはめ波） =====
                # 条件:
                #  A) 手首同士が近い（両手を揃える/合わせる）
                #  B) その状態で顔から遠い  もしくは  肩→手首距離が長い（腕が伸びている）
                if shoulder_width and shoulder_width > 0:
                    wrists_close = euclid_xy(lwr_xy, rwr_xy) <= shoulder_width * WRISTS_CLOSE_RATIO

                    # 顔からの距離（鼻-手首平均）
                    mean_hand_face = (euclid_xy(nose_xy, lwr_xy) + euclid_xy(nose_xy, rwr_xy)) / 2.0
                    hands_far_from_face = mean_hand_face >= shoulder_width * HANDS_FAR_RATIO

                    # 肩→手首が長い（左右平均）
                    lw = euclid_xy(lsh_xy, lwr_xy)
                    rw = euclid_xy(rsh_xy, rwr_xy)
                    mean_arm_len = (lw + rw) / 2.0
                    arms_extended = mean_arm_len >= shoulder_width * ARM_EXT_RATIO

                    attack_frame = bool(wrists_close and (hands_far_from_face or arms_extended))

                if DRAW_DEBUG:
                    mp_draw.draw_landmarks(
                        out, res_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_draw.DrawingSpec(color=(0,200,200), thickness=2)
                    )
                    # 参考線
                    cv2.line(out, lsh_xy, lwr_xy, (255,120,60), 3)
                    cv2.line(out, rsh_xy, rwr_xy, (60,120,255), 3)
                    cv2.circle(out, nose_xy, 6, (0,255,255), -1)
                    cv2.circle(out, lwr_xy, 6, (255,255,0), -1)
                    cv2.circle(out, rwr_xy, 6, (255,255,0), -1)
            except Exception:
                pass

        # ========== Hands: Charge 判定（4本指 TIP を左右で“結ぶ”） ==========
        # 指先ペア: Index(8), Middle(12), Ring(16), Pinky(20)
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
                tip_ids = [8, 12, 16, 20]
                # 平均手幅（左右の手幅の平均）
                base_left  = palm_width(left_lm)
                base_right = palm_width(right_lm)
                base = safe_mean([base_left, base_right])
                if base and base > 1e-6:
                    # 4本の同名指TIP間の平均距離（正規化）
                    norm_dists = []
                    for tid in tip_ids:
                        d = dist2d_lm(left_lm[tid], right_lm[tid])
                        norm_dists.append(d / base)
                        if DRAW_DEBUG:
                            lx, ly = to_px(left_lm[tid], W, H)
                            rx, ry = to_px(right_lm[tid], W, H)
                            cv2.line(out, (lx, ly), (rx, ry), (60, 60, 255), 2)
                    mean_norm = sum(norm_dists) / len(norm_dists)
                    charge_frame = (mean_norm <= CHARGE_TIP_RATIO)

        # ========== 時系列の平滑化 ==========
        hist_guard.append(1 if guard_frame else 0)
        hist_attack.append(1 if attack_frame else 0)
        hist_charge.append(1 if charge_frame else 0)

        guard_now  = (len(hist_guard)  == HOLD_FRAMES and sum(hist_guard)  == HOLD_FRAMES)
        attack_now = (len(hist_attack) == HOLD_FRAMES and sum(hist_attack) == HOLD_FRAMES)
        charge_now = (len(hist_charge) == HOLD_FRAMES and sum(hist_charge) == HOLD_FRAMES)

        # 優先度: Charge > Attack > Guard
        label = None
        if charge_now:
            label = "CHARGE"
        elif attack_now:
            label = "ATTACK"
        elif guard_now:
            label = "GUARD"

        # ========== 表示 ==========
        if label:
            cv2.putText(out, label, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
        else:
            cv2.putText(out, "Idle", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (100, 100, 100), 4)

        # デバッグ: 生のフレーム判定を表示
        dbg = f"G:{guard_frame}  A:{attack_frame}  C:{charge_frame}"
        cv2.putText(out, dbg, (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 50), 2)

        cv2.imshow("Guard/Attack/Charge Detection (Pose + Hands)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
