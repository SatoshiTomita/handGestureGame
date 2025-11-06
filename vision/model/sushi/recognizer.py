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
    # unko.pyの認識モデルパラメータ
    ATTACK_PINKY_CLOSE_RATIO = 0.40  # 小指TIP間距離 / 手幅 <= しきい（小指が近い）- 緩和
    ATTACK_NON_PINKY_FAR_RATIO = 0.60  # 他3本TIPの平均距離 / 手幅 >= しきい（他が遠い）- 緩和
    CHARGE_TIP_RATIO = 0.45  # 4本のTIPペア距離の平均 / 手幅 <= しきい（近い）

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
        """1フレームから (guard, attack, charge, label or None) を返す。unko.pyの認識モデルロジックを使用。"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res_hands = self.hands.process(img)
        img.flags.writeable = True

        H, W = frame.shape[:2]
        guard = attack = charge = False
        hands_detected = False

        # unko.pyの認識モデルロジックを使用
        mean_norm_tip_dist = None
        norm_pinky_tip_dist = None
        mean_norm_non_pinky_tip_dist = None

        # ---- Hands → Charge/Attack/Guard判定 ----
        if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
            left_lm = right_lm = None
            for hand_lm, handed in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                if handed.classification[0].label == "Left":
                    left_lm = hand_lm.landmark
                else:
                    right_lm = hand_lm.landmark

            if left_lm is not None and right_lm is not None:
                hands_detected = True
                tip_ids = [8, 12, 16, 20]  # Index(8), Middle(12), Ring(16), Pinky(20)

                # 手幅を正規化基準として使用
                base_left = _palm_width(left_lm)
                base_right = _palm_width(right_lm)
                base = _safe_mean([base_left, base_right])

                if base and base > 1e-6:
                    norm_dists_all = []
                    norm_dists_non_pinky = []

                    for tid in tip_ids:
                        d = _dist2d_lm(left_lm[tid], right_lm[tid])
                        norm_dists_all.append(d / base)

                        if tid == 20:  # Pinky TIP ID
                            norm_pinky_tip_dist = d / base
                        else:
                            norm_dists_non_pinky.append(d / base)

                    # Charge判定に使う平均距離
                    mean_norm_tip_dist = sum(norm_dists_all) / len(norm_dists_all)

                    # Attack判定に使うPinky以外の平均距離
                    if norm_dists_non_pinky:
                        mean_norm_non_pinky_tip_dist = sum(norm_dists_non_pinky) / len(norm_dists_non_pinky)

        # Charge/Attack/Guard判定の確定
        if mean_norm_tip_dist is not None and norm_pinky_tip_dist is not None:
            # 1. Charge判定
            charge = (mean_norm_tip_dist <= self.cfg.CHARGE_TIP_RATIO)

            # 2. Attack判定（小指特化型、改善版）
            # 小指が近いことを主条件とし、他の指との相対的な距離差を考慮
            is_pinky_close = (norm_pinky_tip_dist <= self.cfg.ATTACK_PINKY_CLOSE_RATIO)
            is_non_pinky_far = (mean_norm_non_pinky_tip_dist is not None and 
                               mean_norm_non_pinky_tip_dist >= self.cfg.ATTACK_NON_PINKY_FAR_RATIO)
            
            # 小指が他の指より明らかに近い場合もAttackと判定（相対的な差を考慮）
            if is_pinky_close and mean_norm_non_pinky_tip_dist is not None:
                # 小指の距離が他の指の平均距離より十分に小さい場合
                pinky_vs_others_diff = mean_norm_non_pinky_tip_dist - norm_pinky_tip_dist
                is_pinky_relatively_close = (pinky_vs_others_diff >= 0.15)  # 相対的な差が0.15以上
                attack = bool(is_pinky_close and (is_non_pinky_far or is_pinky_relatively_close))
            else:
                attack = bool(is_pinky_close and is_non_pinky_far)

            # Chargeが成立した場合は、Attackを強制的にFalseにする（Charge優先）
            if charge:
                attack = False

            # 3. Guard判定
            # Guard判定: AttackでもChargeでもない（両手が検出されていることが前提）
            if not attack and not charge and hands_detected:
                guard = True

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

        # hand_xyの計算はHandsのみを使用（Poseは使用しない）
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
