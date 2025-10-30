# camera_stream.py
import cv2, threading, time

class CameraStream:
    def __init__(self, device=0, width=960, height=540, api=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(device, api)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self._stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))} open failed")
        self.th.start()
        return self

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.ok, self.frame = True, f
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return (self.ok, self.frame.copy() if self.frame is not None else None)

    def release(self):
        self._stop = True
        self.th.join(timeout=0.3)
        self.cap.release()
