# fsm.py
import time
from dataclasses import dataclass

READY_MS = 1300
GO_MS    = 500
RESULT_MS = 600

@dataclass
class Phase:
    name: str
    until: float

class RoundFSM:
    """
    IDLE -> READY -> GO -> SAMPLE -> SHOW_RESULT -> NEXT(or GAME_END)
    時間経過で進める。sleepしない。
    """
    def __init__(self, sampling_sec: float, best_of: int = 3):
        self.best_of = best_of
        self.target_score = (best_of + 1) // 2
        self.sampling_sec = sampling_sec
        self.reset_match()

    def reset_match(self):
        self.score_L = 0
        self.score_R = 0
        self.phase = Phase("IDLE", 0.0)
        self.outcome_msg = ""

    def start_round(self):
        now = time.monotonic()
        self.phase = Phase("READY", now + READY_MS/1000.0)

    def advance(self):
        now = time.monotonic()
        if now < self.phase.until:
            return  # まだフェーズ中

        if self.phase.name == "READY":
            self.phase = Phase("GO", now + GO_MS/1000.0)
        elif self.phase.name == "GO":
            self.phase = Phase("SAMPLE", now + self.sampling_sec)
        elif self.phase.name == "SAMPLE":
            self.phase = Phase("SHOW_RESULT", now + RESULT_MS/1000.0)
        elif self.phase.name == "SHOW_RESULT":
            if self.score_L >= self.target_score or self.score_R >= self.target_score:
                self.phase = Phase("GAME_END", now + 86400)
            else:
                self.phase = Phase("IDLE", now + 86400)
        # IDLE/GAME_END は外部入力でのみ遷移

    def set_result_and_scores(self, outcome: str):
        self.outcome_msg = outcome
        if outcome == "win":  self.score_L += 1
        if outcome == "lose": self.score_R += 1
