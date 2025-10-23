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
    """両手のランドマークからCHARGEポーズのみを識別する"""
    
    # 両手の手首（ランドマーク0）の座標を取得
    w1 = (lms1.landmark[0].x * img_w, lms1.landmark[0].y * img_h, lms1.landmark[0].z)
    w2 = (lms2.landmark[0].x * img_w, lms2.landmark[0].y * img_h, lms2.landmark[0].z)

    # 両手の手首間のXY平面上の距離
    xy_dist = dist(w1, w2)

    # 1: 両手が近いかどうかの判定（画像幅の25%未満）
    close = xy_dist < img_w * 0.25

    # 指の開いている数
    f1 = fingers_up(lms1)
    f2 = fingers_up(lms2)
    open_ratio = (f1 + f2) / 8  # 4本指 * 2手で合計8本に対する比率

    # --- CHARGEポーズの条件のみをチェック ---
    
    # CHARGE: 両手が近くて、指の開きが少ない（グー）
    if close and open_ratio < 0.2: 
        return "CHARGE" 
        
    else:
        # CHARGEの条件を満たさない場合は全て UNKNOWN
        return "UNKNOWN"

# --- メイン処理（変更なし） ---

def main():
    # カメラのインデックスは環境に応じて 0, 1, 2 などを試してください
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。インデックス番号を確認してください。")
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
                # 検出された2つの手を取得
                lms1, lms2 = res.multi_hand_landmarks[0], res.multi_hand_landmarks[1]
                
                # ランドマークを描画
                mp_drawing.draw_landmarks(img, lms1, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, lms2, mp_hands.HAND_CONNECTIONS)
                
                # ポーズを分類
                label = classify_pose(lms1, lms2, img_w, img_h)

            # 結果を画像に表示
            # CHARGEのときだけ色を変えるなどすると分かりやすいです
            color = (0, 255, 255) if label == "UNKNOWN" or label == "NO HANDS" else (0, 0, 255)
            
            cv2.putText(img, f"POSE: {label}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
            
            # 映像を表示
            cv2.imshow("CHARGE Detector", img)
            
            # 'q'キーが押されたらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # 終了処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()