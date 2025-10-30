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


def draw_charge_icons(
    img,
    count: int,
    max_count: int = 5,
    anchor: str = "left",   
    y: int = 36,         
    radius: int = 10,     
    gap: int = 10           
):
    """
    チャージ個数を丸アイコンで表示。count が満たされている個数を塗りつぶし、
    残りは外枠のみ表示する。
    """
    H, W = img.shape[:2]
    color_fill = POSE_COLOR.get("CHARGE", (255, 180, 0)) 
    color_stroke = (40, 40, 40)

    total_w = max_count * (2*radius) + (max_count - 1) * gap

    if anchor == "right":
        x0 = W - 20 - total_w 
    else:
        x0 = 20               

    for i in range(max_count):
        cx = x0 + i * (2*radius + gap)
        cy = y

        filled = (i < max(0, int(count)))
        # 影
        cv2.circle(img, (cx+1, cy+1), radius, (0,0,0), -1, lineType=cv2.LINE_AA)
        # 本体
        if filled:
            cv2.circle(img, (cx, cy), radius, color_fill, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (cx, cy), radius, color_stroke, 2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img, (cx, cy), radius, (230,230,230), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (cx, cy), radius, color_stroke, 2, lineType=cv2.LINE_AA)