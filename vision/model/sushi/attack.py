import cv2
import mediapipe as mp
import math

# MediaPipe Handsと描画ユーティリティを初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- ユーティリティ関数（変更なし） ---

def dist(a, b):
    """2点間のユークリッド距離を計算する（XY座標のみ）"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def fingers_up(hand_landmarks):
    """指が上がっている数（グー/パーの判定に使用）をざっくりと返す"""
    lm = hand_landmarks.landmark
    # 8, 12, 16, 20のY座標が、その付け根(i-2)のY座標より小さい（画面上で上にある）場合に指が上がっていると判定
    return sum(1 for i in [8, 12, 16, 20] if lm[i].y < lm[i-2].y)

def classify_pose(lms1, lms2, img_w, img_h):
    """両手のランドマークからATTACKポーズのみを識別する"""
    
    # 両手の手首（ランドマーク0）の座標とZ値を取得
    w1 = (lms1.landmark[0].x * img_w, lms1.landmark[0].y * img_h, lms1.landmark[0].z)
    w2 = (lms2.landmark[0].x * img_w, lms2.landmark[0].y * img_h, lms2.landmark[0].z)

    # 両手の手首間のXY平面上の距離
    xy_dist = dist(w1, w2)
    # 両手の手首間のZ軸（奥行き）の差 (ATTACK判別のコア要素)
    z_diff = abs(w1[2] - w2[2])

    # 1: 両手が近いかどうかの判定（画像幅の25%未満）
    close = xy_dist < img_w * 0.25

    # 指の開いている数
    f1 = fingers_up(lms1)
    f2 = fingers_up(lms2)
    open_ratio = (f1 + f2) / 8  # 4本指 * 2手で合計8本に対する比率

    # --- ATTACKポーズの条件のみをチェック ---
    
    # ATTACK: 両手が近い AND 指の開きが多い（パー）AND 前後差が大きい（突き出し）
    if close and open_ratio > 0.7 and z_diff > 0.1:
        return "ATTACK"  
        
    else:
        # ATTACKの条件を満たさない場合は全て UNKNOWN
        return "UNKNOWN"

# --- メイン処理 (VideoCapture(1)に修正) ---

def main():
    # ★★★ 変更点: カメラのインデックスを 1 に設定 ★★★
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        # インデックス1のカメラが開けない場合はエラーメッセージを表示
        print("エラー: カメラ (インデックス 1) を開けませんでした。")
        print("0 や 2 など、他のインデックスを試してください。")
        return

    # MediaPipe Handsの設定
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("フレームを読み込めませんでした。終了します。")
                break
            
            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            
            img_h, img_w = img.shape[:2]
            label = "NO HANDS"

            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                lms1, lms2 = res.multi_hand_landmarks[0], res.multi_hand_landmarks[1]
                
                mp_drawing.draw_landmarks(img, lms1, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, lms2, mp_hands.HAND_CONNECTIONS)
                
                label = classify_pose(lms1, lms2, img_w, img_h)

            # 結果を画像に表示
            color = (0, 255, 255) if label == "UNKNOWN" or label == "NO HANDS" else (255, 0, 0)
            
            cv2.putText(img, f"POSE: {label}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
            
            cv2.imshow("ATTACK Detector (Camera 1)", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()