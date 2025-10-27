# referee.py
from typing import Tuple, Optional

Move = Optional[str]  # "CHARGE" | "ATTACK" | "DEFENSE" | None

def judge_cclemon(user_move: Move, cpu_move: Move,
                  user_power: int, cpu_power: int) -> Tuple[str, int, int]:
    """
    寿司じゃんけん判定。
    returns: (outcome, new_user_power, new_cpu_power)
    outcome: "win" | "lose" | "draw-continue" | "no-move"
    ルール:
      - CHARGE: +1
      - ATTACK: -1（相手がCHARGEなら勝ち）
      - DEFENSE: 変化なし
      - パワー0でATTACKは不可 → DEFENSEへフォールバック
    """
    if user_move is None and cpu_move is None:
        return "no-move", user_power, cpu_power

    if user_move == "ATTACK" and user_power <= 0:
        user_move = "DEFENSE"
    if cpu_move == "ATTACK" and cpu_power <= 0:
        cpu_move = "DEFENSE"

    if user_move == "CHARGE": user_power += 1
    elif user_move == "ATTACK": user_power -= 1

    if cpu_move == "CHARGE": cpu_power += 1
    elif cpu_move == "ATTACK": cpu_power -= 1

    if user_move == "ATTACK" and cpu_move == "CHARGE":
        return "win", user_power, cpu_power
    if cpu_move == "ATTACK" and user_move == "CHARGE":
        return "lose", user_power, cpu_power

    return "draw-continue", user_power, cpu_power
