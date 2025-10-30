# tts_async.py
import threading, queue, time

class TTSWorker:
    def __init__(self, speak_fn, timeout=0.9):
        self.q = queue.Queue()
        self.timeout = timeout
        self.speak_fn = speak_fn
        self._stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def say(self, text):
        # キューに積むだけ（メインは止まらない）
        if text:
            self.q.put(text)

    def close(self):
        self._stop = True
        self.q.put(None)
        self.th.join(timeout=0.5)

    def _loop(self):
        while not self._stop:
            text = self.q.get()
            if text is None: break
            done = {"ok": False}

            def _run():
                try:
                    self.speak_fn(text)
                    done["ok"] = True
                except Exception:
                    pass
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(self.timeout)  # 再生が遅延してもここで打ち切る
            # タイムアウトでもメインには影響させない
