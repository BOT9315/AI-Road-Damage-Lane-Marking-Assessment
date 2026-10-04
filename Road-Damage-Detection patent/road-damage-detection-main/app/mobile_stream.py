import cv2
import numpy as np
from flask import Flask, render_template_string, request, Response
import threading


app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

HTML_MOBILE_SENDER = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>VisionRoad Live Broadcast</title>
    <style>
        body { background: #0E1117; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
        video { width: 100%; max-width: 480px; border-radius: 12px; border: 2px solid #58A6FF; }
        .btn { background: #238636; color: white; border: none; padding: 14px 28px; font-size: 18px; border-radius: 8px; font-weight: bold; margin-top: 15px; cursor: pointer; }
        .status { margin-top: 10px; color: #3FB950; font-weight: bold; }
    </style>
</head>
<body>
    <h2>🚗 VisionRoad Live Cam</h2>
    <p>Live Streaming to Laptop AI Engine</p>
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <br>
    <button id="startBtn" class="btn" onclick="startStream()">Start Broadcasting</button>
    <p id="status" class="status"></p>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const status = document.getElementById('status');
        let streaming = false;

        async function startStream() {
            try {
                // Request rear camera for road inspection
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } }
                });
                video.srcObject = stream;
                document.getElementById('startBtn').style.display = 'none';
                status.innerText = "🔴 Live Broadcasting Active";
                streaming = true;
                sendFrames();
            } catch (err) {
                alert("Camera Access Error: " + err);
            }
        }

        function sendFrames() {
            if (!streaming) return;
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob(blob => {
                if (blob) {
                    fetch('/upload_frame', {
                        method: 'POST',
                        body: blob
                    }).then(() => {
                        setTimeout(sendFrames, 100); // Send ~10 FPS for low latency
                    }).catch(() => {
                        setTimeout(sendFrames, 200);
                    });
                }
            }, 'image/jpeg', 0.6);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_MOBILE_SENDER)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame
    img_bytes = request.data
    if img_bytes:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with frame_lock:
            latest_frame = frame
    return ('', 204)

def start_server(port=5000):
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False, use_reloader=False)

def get_current_frame():
    global latest_frame
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None

if __name__ == '__main__':
    start_server()