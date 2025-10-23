import cv2
import math
import time
from collections import deque
import mediapipe as mp

# ========== ユーティリティ ==========
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

# ========== パラメータ ==========
# 鼻から手首までの距離が「肩幅 * この係数」以下なら顔の前にあるとみなす
NEAR_FACE_RATIO = 0.65
# 何フレーム連続で条件を満たしたら「ガード」とするか
HOLD_FRAMES = 6
# 判定の平滑化バッファ
history = deque(maxlen=HOLD_FRAMES)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # カメラIDは環境に応じて
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(
    static_image_mode=False,
    enable_segmentation=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    last_state = False  # 直近の表示状態（True=ガード表示中）
    while True:
        ok, frame = cap.read()
        if not ok:
            print("カメラから映像を取得できません。")
            break

        # MediaPipe はRGB想定
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = pose.process(img)
        img.flags.writeable = True
        out = frame.copy()

        H, W = out.shape[:2]
        is_guard_frame = False

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # 必要ランドマーク（存在チェック）
            try:
                nose = lm[mp_pose.PoseLandmark.NOSE]
                l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            except Exception:
                nose = l_sh = r_sh = l_wr = r_wr = None

            if all(p is not None for p in [nose, l_sh, r_sh, l_wr, r_wr]):
                # ピクセル座標へ
                nose_xy = to_px(nose, W, H)
                lsh_xy  = to_px(l_sh, W, H)
                rsh_xy  = to_px(r_sh, W, H)
                lwr_xy  = to_px(l_wr, W, H)
                rwr_xy  = to_px(r_wr, W, H)

                # 肩幅（ピクセル）
                shoulder_width = euclid(lsh_xy, rsh_xy)
                if shoulder_width > 0:
                    # 鼻-手首距離
                    dl = euclid(nose_xy, lwr_xy)
                    dr = euclid(nose_xy, rwr_xy)

                    # 手首が顔の前（鼻の近く）にある？
                    near_left  = (dl <= shoulder_width * NEAR_FACE_RATIO)
                    near_right = (dr <= shoulder_width * NEAR_FACE_RATIO)

                    # さらに「上半身の高さ」も軽くチェック（手首が胸より上にある想定）
                    chest_y = int((lsh_xy[1] + rsh_xy[1]) / 2)
                    high_left  = lwr_xy[1] <= chest_y + 0.15 * shoulder_width
                    high_right = rwr_xy[1] <= chest_y + 0.15 * shoulder_width

                    is_guard_frame = (near_left and near_right and high_left and high_right)

                # 可視化
                mp_drawing.draw_landmarks(
                    out,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,200), thickness=2)
                )

                # 顔位置の目安（鼻）
                cv2.circle(out, nose_xy, 6, (0, 255, 255), -1)
                # 手首
                cv2.circle(out, lwr_xy, 6, (255, 255, 0), -1)
                cv2.circle(out, rwr_xy, 6, (255, 255, 0), -1)

        # 判定の平滑化（HOLD_FRAMES連続でTrueならガード）
        history.append(1 if is_guard_frame else 0)
        guard_now = (len(history) == HOLD_FRAMES and sum(history) == HOLD_FRAMES)

        # 表示
        if guard_now:
            cv2.putText(out, "GUARD!", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        else:
            cv2.putText(out, "Guard: OFF", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)

        # デバッグ用に現在の生判定も表示
        cv2.putText(out, f"frame_guard={is_guard_frame}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)

        cv2.imshow("Guard Detection (OpenCV + MediaPipe Pose)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
