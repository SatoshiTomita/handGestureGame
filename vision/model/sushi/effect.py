# effect.py
import numpy as np
import cv2

def load_image(path: str, height: int | None = None):
    """アルファ付きPNG/WEBP等を読み込む（4ch対応）。height 指定で縦を等比リサイズ。
    白背景を透過させる処理も行う。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    # アルファチャンネルがない場合、白背景を透過させる
    if img.shape[2] == 3:  # BGR画像
        # 白背景を検出：BGRの全てのチャンネルが高い値（200以上）であるピクセルを白とみなす
        # より確実に白背景を除去するため、各チャンネルを個別にチェック
        b, g, r = cv2.split(img)
        # BGRの全てのチャンネルが200以上であるピクセルを白背景として判定
        white_mask = (b >= 200) & (g >= 200) & (r >= 200)
        # 白背景を透明にするため、マスクを反転してアルファチャンネルとして使用
        alpha = (~white_mask).astype(np.uint8) * 255
        # 4チャンネル画像に変換（BGR + Alpha）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = alpha
    
    if height:
        H, W = img.shape[:2]
        scale = height / float(H)
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    return img

def overlay_rgba(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int):
    """dst(BGR) に src(BGRA/ BGR) を (x,y) 左上でアルファブレンド"""
    sh, sw = src_bgra.shape[:2]
    dh, dw = dst_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x+sw), min(dh, y+sh)
    if x0 >= x1 or y0 >= y1:
        return
    src_roi = src_bgra[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    dst_roi = dst_bgr[y0:y1, x0:x1]
    if src_roi.shape[2] == 4:
        src_rgb = src_roi[:, :, :3]
        alpha   = src_roi[:, :, 3:4] / 255.0
        dst_roi[:] = (alpha * src_rgb + (1.0 - alpha) * dst_roi).astype(np.uint8)
    else:
        dst_roi[:] = src_roi

def overlay_icon_anchored(panel_bgr: np.ndarray, icon_bgra: np.ndarray,
                          anchor: str = "left", y: int = 80, margin: int = 20):
    """左/右アンカーで良い位置にアイコンを重ねる"""
    H, W = panel_bgr.shape[:2]
    h, w = icon_bgra.shape[:2]
    x = (W - margin - w) if anchor == "right" else margin
    overlay_rgba(panel_bgr, icon_bgra, x, max(0, y - h // 2))

def overlay_icon_center(dst_bgr: np.ndarray, icon_bgra: np.ndarray, cx: int, cy: int):
    """アイコン中心を (cx, cy) にしてアルファ合成"""
    h, w = icon_bgra.shape[:2]
    x = int(cx - w // 2)
    y = int(cy - h // 2)
    overlay_rgba(dst_bgr, icon_bgra, x, y)