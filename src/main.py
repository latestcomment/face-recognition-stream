from flask import Flask, Response, render_template
import cv2

import time
import queue
import threading
from collections import deque

from modules.process_frame import process_frame
from modules.stream import Stream

from modules import camera_config

app = Flask(__name__)

OUTPUT_QUEUE = queue.Queue(maxsize=1)

def process_frame_thread():
    prev_time = time.time()
    fps_history = deque(maxlen=30)

    stream = Stream(camera_config.get("SOURCE"))
    stream.start()

    while not stream.stopped:
        frame = stream.read()

        # Calculate FPS
        curr_time = time.time()
        fps_history.append(1.0 / (curr_time - prev_time))
        prev_time = curr_time
        fps = sum(fps_history) / len(fps_history)

        processed = process_frame(frame)

        cv2.putText(processed, f"FPS: {fps:.2f}",
                    (10, processed.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', processed)
        encoded_frame = buffer.tobytes()

        if OUTPUT_QUEUE.full():
            OUTPUT_QUEUE.get_nowait()
        OUTPUT_QUEUE.put(encoded_frame)

def main():
    threading.Thread(target=process_frame_thread, daemon=True).start()
    while True:
        frame = OUTPUT_QUEUE.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)