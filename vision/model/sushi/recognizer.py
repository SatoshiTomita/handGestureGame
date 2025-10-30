# recognizer.py
import time
import math
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp

class CFG:
    HOLD_FRAMES = 2
    SAMPLING_SEC = 1.1
    NEAR_FACE_RATIO = 1.10
    WRISTS_CLOSE_RATIO = 0.80
    HANDS_FAR_RATIO = 1.10
    ARM_EXT_RATIO = 1.15
    CHARGE_TIP_RATIO = 0.55

def _to_px(landmark, w, h): return (int(landmark.x * w), int(landmark.y * h))
def _euclid_xy(a_xy, b_xy): return math.hypot(a_xy[0] - b_xy[0], a_xy[1] - b_xy[1])
def _dist2d_lm(a, b): return math.hypot(a.x - b.x, a.y - b.y)
def _safe_mean(xs): xs = [x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def _palm_width(hand_lm):
    if hand_lm and len(hand_lm) > 17: return _dist2d_lm(hand_lm[5], hand_lm[17])
    return _dist2d_lm(hand_lm[0], hand_lm[9])

def _label_from_flags(guard_now, attack_now, charge_now):
    if charge_now: return "CHARGE"
    if attack_now: return "ATTACK"
    if guard_now:  return "GUARD"
    return None

class SushiRecognizer:
    """ユーザーの手を CHARGE / ATTACK / GUARD のどれかに認識するモジュール。"""
    def __init__(self, draw_debug: bool = True, cfg: CFG = CFG()):
        self.cfg = cfg
        self.draw_debug = draw_debug
        self._mp_pose  = mp.solutions.pose
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self.pose = self._mp_pose.Pose(
            static_image_mode=False, enable_segmentation=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.hands = self._mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

    def close(self):
        if self.pose:  self.pose.close()
        if self.hands: self.hands.close()

    def process_frame(self, frame) -> Tuple[bool, bool, bool, Optional[str]]:
        """1フレームから (guard, attack, charge, label or None) を返す。描画なし。"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res_pose  = self.pose.process(img)
        res_hands = self.hands.process(img)
        img.flags.writeable = True

        H, W = frame.shape[:2]
        guard = attack = charge = False

        # ---- Pose → Guard / Attack ----
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            try:
                nose = lm[self._mp_pose.PoseLandmark.NOSE]
                l_sh = lm[self._mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = lm[self._mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wr = lm[self._mp_pose.PoseLandmark.LEFT_WRIST]
                r_wr = lm[self._mp_pose.PoseLandmark.RIGHT_WRIST]
                nose_xy = _to_px(nose, W, H)
                lsh_xy  = _to_px(l_sh, W, H)
                rsh_xy  = _to_px(r_sh, W, H)
                lwr_xy  = _to_px(l_wr, W, H)
                rwr_xy  = _to_px(r_wr, W, H)
                shoulder_width = _euclid_xy(lsh_xy, rsh_xy) if (lsh_xy and rsh_xy) else None

                if shoulder_width and shoulder_width > 0:
                    # Guard
                    dl = _euclid_xy(nose_xy, lwr_xy)
                    dr = _euclid_xy(nose_xy, rwr_xy)
                    near_left  = dl <= shoulder_width * self.cfg.NEAR_FACE_RATIO
                    near_right = dr <= shoulder_width * self.cfg.NEAR_FACE_RATIO
                    guard = (near_left and near_right)
                    # Attack
                    wrists_close = _euclid_xy(lwr_xy, rwr_xy) <= shoulder_width * self.cfg.WRISTS_CLOSE_RATIO
                    mean_hand_face = (_euclid_xy(nose_xy, lwr_xy) + _euclid_xy(nose_xy, rwr_xy)) / 2.0
                    hands_far_from_face = mean_hand_face >= shoulder_width * self.cfg.HANDS_FAR_RATIO
                    lw = _euclid_xy(lsh_xy, lwr_xy)
                    rw = _euclid_xy(rsh_xy, rwr_xy)
                    mean_arm_len = (lw + rw) / 2.0
                    arms_extended = mean_arm_len >= shoulder_width * self.cfg.ARM_EXT_RATIO
                    attack = bool(wrists_close and (hands_far_from_face or arms_extended))
            except Exception:
                pass

        # ---- Hands → Charge ----
        if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
            left_lm = right_lm = None
            for hand_lm, handed in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                if handed.classification[0].label == "Left": left_lm = hand_lm.landmark
                else: right_lm = hand_lm.landmark
            if left_lm is not None and right_lm is not None:
                tip_ids = [8, 12, 16, 20]
                base = _safe_mean([_palm_width(left_lm), _palm_width(right_lm)])
                if base and base > 1e-6:
                    norm_dists = [ _dist2d_lm(left_lm[tid], right_lm[tid]) / base for tid in tip_ids ]
                    charge = (sum(norm_dists)/len(norm_dists)) <= self.cfg.CHARGE_TIP_RATIO

        hand_xy = None

        if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
            L_xy = R_xy = None
            for hand_lm, handed in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                wrist = hand_lm.landmark[0] 
                xy = _to_px(wrist, W, H)
                if handed.classification[0].label == "Left":
                    L_xy = xy
                else:
                    R_xy = xy
            if L_xy and R_xy:
                hand_xy = ((L_xy[0] + R_xy[0]) // 2, (L_xy[1] + R_xy[1]) // 2)  # 両手の中点
            elif L_xy:
                hand_xy = L_xy
            elif R_xy:
                hand_xy = R_xy

        # 2) Hands が無ければ Pose の手首でフォールバック
        if hand_xy is None and res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            try:
                l_wr = lm[self._mp_pose.PoseLandmark.LEFT_WRIST]
                r_wr = lm[self._mp_pose.PoseLandmark.RIGHT_WRIST]
                L_xy = _to_px(l_wr, W, H)
                R_xy = _to_px(r_wr, W, H)
                if L_xy and R_xy:
                    hand_xy = ((L_xy[0] + R_xy[0]) // 2, (L_xy[1] + R_xy[1]) // 2)
                elif L_xy:
                    hand_xy = L_xy
                elif R_xy:
                    hand_xy = R_xy
            except Exception:
                pass
        info = {"hand_xy": hand_xy}
        return guard, attack, charge, info

    def sample_label(self, cap) -> str:
        """短時間（CFG.SAMPLING_SEC）で安定ラベルを返す。'__quit__' でユーザー終了。"""
        t_end = time.time() + self.cfg.SAMPLING_SEC
        hist_g = deque(maxlen=self.cfg.HOLD_FRAMES)
        hist_a = deque(maxlen=self.cfg.HOLD_FRAMES)
        hist_c = deque(maxlen=self.cfg.HOLD_FRAMES)
        decided_label = None
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            g, a, c, _ = self.process_frame(frame)
            hist_g.append(1 if g else 0)
            hist_a.append(1 if a else 0)
            hist_c.append(1 if c else 0)
            guard_now  = len(hist_g)==self.cfg.HOLD_FRAMES and sum(hist_g)==self.cfg.HOLD_FRAMES
            attack_now = len(hist_a)==self.cfg.HOLD_FRAMES and sum(hist_a)==self.cfg.HOLD_FRAMES
            charge_now = len(hist_c)==self.cfg.HOLD_FRAMES and sum(hist_c)==self.cfg.HOLD_FRAMES
            label = _label_from_flags(guard_now, attack_now, charge_now)
            if label is not None:
                decided_label = label
            if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
                return "__quit__"
        return decided_label
