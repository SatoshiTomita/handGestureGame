#efftct.py
import numpy as np
import cv2
def load_png(path: str, height: int | None = None):
    """Load a PNG (with alpha) and optionally resize by height keeping aspect ratio.
    Returns None if load fails.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if height is not None and img.shape[0] != height:
        H, W = img.shape[:2]
        new_w = int(W * (height / float(H)))
        img = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    return img

def overlay_icon_anchored(panel_bgr, icon_rgba, anchor: str = "left", y: int = 90, margin: int = 20):
    """Overlay RGBA icon onto a BGR panel with alpha blending.
    anchor: "left" or "right" aligns horizontally; y gives top-left y coordinate.
    """
    if icon_rgba is None:
        return
    if icon_rgba.shape[2] == 3:
        # No alpha channel; treat as opaque
        alpha = None
    else:
        alpha = icon_rgba[:, :, 3] / 255.0
    icon_bgr = icon_rgba[:, :, :3] if icon_rgba.shape[2] >= 3 else icon_rgba

    H, W = panel_bgr.shape[:2]
    h, w = icon_bgr.shape[:2]
    if anchor == "right":
        x = max(0, W - w - margin)
    else:
        x = margin
    y = max(0, min(y, H - h))

    roi = panel_bgr[y:y+h, x:x+w]
    if roi.shape[0] != h or roi.shape[1] != w:
        return
    if alpha is None:
        panel_bgr[y:y+h, x:x+w] = icon_bgr
        return
    # Alpha blend
    for c in range(3):
        roi[:, :, c] = (alpha * icon_bgr[:, :, c] + (1.0 - alpha) * roi[:, :, c]).astype(roi.dtype)
    panel_bgr[y:y+h, x:x+w] = roi


def load_png(path: str, height: int | None = None):
    """アルファ付きPNGを読み込む。height指定で縦サイズを等比リサイズ"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"PNG not found: {path}")
    if height:
        H, W = img.shape[:2]
        scale = height / float(H)
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    return img

def overlay_rgba(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int):
    """dst(BGR) に src(BGRA) を (x,y) 左上でアルファブレンド"""
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