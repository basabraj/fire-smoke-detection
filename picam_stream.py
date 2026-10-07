import time
import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO

PORT = 4205
WIDTH, HEIGHT = (640,480)

# Confidence Scores
FIRE_CONF = 0.65
SMOKE_CONF = 0.60
MODEL_CONF_FLOOR = min(FIRE_CONF, SMOKE_CONF) 

CLASS_MAP = {0: 'fire', 1: 'smoke'}  # empirically verified, see above
CLASS_CONF = {0: FIRE_CONF, 1: SMOKE_CONF}  # keyed by the same id as CLASS_MAP
COLORS = {'fire': (0, 0, 139), 'smoke': (139, 0, 0)}  # BGR: fire=deep red, smoke=deep blue

model = YOLO('yolo_smoke_fire.pt')
model.to('cpu')
app = Flask(__name__)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
))
picam2.start()
time.sleep(1)  


def gen_frames():
    while True:
        frame = picam2.capture_array()

        results = model(frame, conf=MODEL_CONF_FLOOR)[0]
        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            if conf < CLASS_CONF.get(cls_id, MODEL_CONF_FLOOR):
                continue  # below this class's own floor — skip before drawing/logging
            cls_name = CLASS_MAP.get(cls_id, f"class{cls_id}")
            print(f"[DETECT] {cls_name} conf={conf:.4f}")  # so exact scores show up in the terminal, not just on-screen
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            color = COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return (
        '<html><body style="margin:0;background:#111;display:flex;'
        'justify-content:center;align-items:center;min-height:100vh">'
        f'<img src="/video_feed" style="width:{WIDTH}px;max-width:100%">'
        '</body></html>'
    )


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print(f"Home-model Pi camera test running. Open http://<pi-ip>:{PORT}/ — Ctrl+C to quit.")
    app.run(host='0.0.0.0', port=PORT, debug=False)
