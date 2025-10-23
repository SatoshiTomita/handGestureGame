import cv2
import mediapipe as mp
import math

# MediaPipe Handsと描画ユーティリティを初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- ユーティリティ関数 ---

def dist(a, b):
    """2点間のユークリッド距離を計算する（XY座標のみ）"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def fingers_up(hand_landmarks):
    """指が上がっている数（グー/パーの判定に使用）をざっくりと返す"""
    lm = hand_landmarks.landmark
    # 8(人差し指先端), 12(中指先端), 16(薬指先端), 20(小指先端)のY座標が、
    # その付け根(i-2)のY座標より小さい（画面上では上にある）場合に指が上がっていると判定
    # 親指(4)は省略し、4本指の判定のみを行う
    return sum(1 for i in [8, 12, 16, 20] if lm[i].y < lm[i-2].y)

def classify_pose(lms1, lms2, img_w, img_h):
    """両手のランドマークからジェスチャーを分類する"""
    
    # 両手の手首（ランドマーク0）の座標とZ値を取得 (Z値は相対的な奥行き)
    w1 = (lms1.landmark[0].x * img_w, lms1.landmark[0].y * img_h, lms1.landmark[0].z)
    w2 = (lms2.landmark[0].x * img_w, lms2.landmark[0].y * img_h, lms2.landmark[0].z)

    # 両手の手首間のXY平面上の距離
    xy_dist = dist(w1, w2)
    # 両手の手首間のZ軸（奥行き）の差
    z_diff = abs(w1[2] - w2[2])

    # 1: 両手が近いかどうかの判定（画像幅の25%未満）
    close = xy_dist < img_w * 0.25

    # 指の本数（開いてる数）
    f1 = fingers_up(lms1)
    f2 = fingers_up(lms2)
    open_ratio = (f1 + f2) / 8  # 4本指 * 2手で合計8本に対する比率

    # 横方向の広がり具合
    horizontal_gap = abs(w1[0] - w2[0])

    # --- 条件分岐によるポーズ分類 ---
    
    # 1. CHARGE: 両手が近くて、指の開きが少ない（グー）
    if close and open_ratio < 0.2: 
        return "CHARGE" 
        
    # 2. BARRIER: 両手が横に広くて、指の開きが多い（パー）
    elif horizontal_gap > img_w * 0.45 and open_ratio > 0.7:
        return "BARRIER"  
        
    # 3. ATTACK: 両手が近くて、指の開きが多く、前後に差がある（突き出し）
    elif close and open_ratio > 0.7 and z_diff > 0.1: # z_diffの閾値を少し厳しく調整
        return "ATTACK"  
        
    else:
        return "UNKNOWN"

# --- メイン処理 ---

def main():
    # 0番目のカメラ（内蔵カメラまたはiPhoneの連係カメラ）をキャプチャ
    # ※iPhoneカメラが起動する場合は、cv2.VideoCapture(1)などを試してください
    cap = cv2.VideoCapture(1)

    # カメラが開けたかチェック
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
            
            # 画像を左右反転（鏡のようにするため）
            img = cv2.flip(frame, 1)
            # MediaPipe用にBGRをRGBに変換
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # MediaPipeで処理を実行
            res = hands.process(rgb)
            
            # 画像の高さと幅を取得
            img_h, img_w = img.shape[:2]
            label = "NO HANDS"

            # ランドマーク描画とポーズ分類
            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                # 検出された2つの手を取得
                lms1, lms2 = res.multi_hand_landmarks[0], res.multi_hand_landmarks[1]
                
                # ランドマークを描画
                mp_drawing.draw_landmarks(img, lms1, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, lms2, mp_hands.HAND_CONNECTIONS)
                
                # ポーズを分類
                label = classify_pose(lms1, lms2, img_w, img_h)

            # 結果を画像に表示
            cv2.putText(img, f"POSE: {label}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            
            # 映像を表示
            cv2.imshow("Gesture Detector", img)
            
            # 'q'キーが押されたらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # 終了処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()