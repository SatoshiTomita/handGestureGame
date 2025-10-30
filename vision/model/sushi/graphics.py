# graphics.py
import cv2

def resize_keep_ar(img, h):
    H, W = img.shape[:2]
    new_w = int(W * (h / H))
    return cv2.resize(img, (new_w, h))

def put_label_top(img, text, y_ratio=0.12, scale=1.2, color=(0,255,255), thick=3):
    H, W = img.shape[:2]
    (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2
    y = int(H * y_ratio)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def overlay_center_text(img, text, y, scale=1.6, color=(0,0,0), thick=5):
    H, W = img.shape[:2]
    (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def concat_side_by_side(f1, f2):
    h = 480
    f1r = resize_keep_ar(f1, h)
    f2r = resize_keep_ar(f2, h)
    concat = cv2.hconcat([f1r, f2r])
    Hc, Wc = concat.shape[:2]
    cv2.line(concat, (Wc//2, 0), (Wc//2, Hc), (0,0,0), 4)
    return concat, f1r, f2r

def put_pose_label(img, pose: str | None, decided: bool = False):
    """Display current/decided pose on the player's panel.
    pose: "CHARGE" | "ATTACK" | "GUARD" | None
    decided: when True, uses a highlight color.
    """
    text = f"[{pose}]" if pose else "[... ]"
    color = (0,200,0) if decided else (120,120,120)
    put_label_top(img, text, y_ratio=0.18, scale=0.9, color=color, thick=2)

POSE_COLOR = {
    "ATTACK": (50, 50, 255),
    "GUARD": (60, 180, 75),
    "CHARGE": (255, 180, 0),
    None: (120,120,120),
}

def draw_vs_bar(img, p1_pose: str|None, p2_pose: str|None, y=320, scale=1.0):
    """画面中央に VS 表示。例: 'P1: ATTACK  vs  P2: GUARD'"""
    H, W = img.shape[:2]
    base = f"P1: {p1_pose or '...'}   vs   P2: {p2_pose or '...'}"
    (tw,_),_ = cv2.getTextSize(base, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    x0 = (W - tw)//2
    # まず全体の影
    cv2.putText(img, base, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    # パートごとに色付け
    x = x0
    def draw_part(txt, color):
        nonlocal x
        (w,_),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3, cv2.LINE_AA)
        x += w

    draw_part("P1: ", (230,230,230))
    draw_part(f"{p1_pose or '...'}", POSE_COLOR.get(p1_pose, (120,120,120)))
    draw_part("   vs   ", (230,230,230))
    draw_part("P2: ", (230,230,230))
    draw_part(f"{p2_pose or '...'}", POSE_COLOR.get(p2_pose, (120,120,120)))

def draw_pose_history(img, history: list[tuple[str|None, str|None]], max_items=5):
    """
    履歴を下部に表示。history は [(p1_pose, p2_pose), ...] 時系列。
    """
    H, W = img.shape[:2]
    y = H - 20
    items = history[-max_items:]
    # 左（P1履歴）
    x1 = 20
    for (p1, _) in items:
        label = p1 or "..."
        cv2.putText(img, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    POSE_COLOR.get(p1, (120,120,120)), 2, cv2.LINE_AA)
        (w,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x1 += w + 14

    # 右（P2履歴）
    x2 = W - 20
    for (_, p2) in reversed(items):
        label = p2 or "..."
        (w,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x2 -= (w + 14)
        cv2.putText(img, label, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    POSE_COLOR.get(p2, (120,120,120)), 2, cv2.LINE_AA)