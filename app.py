from flask import Flask, render_template_string, jsonify
import subprocess
import os
import signal
import time

app = Flask(__name__)
process = None

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ALPR Remote</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; text-align: center; background: #121212; color: white; padding: 15px; }
        .btn { padding: 20px; width: 45%; font-size: 20px; color: white; border: none; border-radius: 12px; cursor: pointer; margin-bottom: 20px;}
        .start { background-color: #2ecc71; }
        .stop { background-color: #e74c3c; }
        .status-bar { margin: 15px 0; font-size: 1.2em; color: #aaa; }
        img { width: 100%; max-width: 600px; border: 3px solid #333; border-radius: 8px; }
        .log-container { margin-top: 20px; text-align: left; background: #222; padding: 10px; border-radius: 8px; font-family: monospace; }
        .log-entry { border-bottom: 1px solid #444; padding: 5px 0; font-size: 0.9em; }
    </style>
</head>
<body>
    <h2>ALPR Controller</h2>
    <div class="status-bar">Status: <span id="status-text">STOPPED</span></div>
    
    <button class="btn start" onclick="sendCommand('start')">START</button>
    <button class="btn stop" onclick="sendCommand('stop')">STOP</button>
    
    <div id="video-container">
        <img id="live-img" src="/static/latest.jpg" alt="Stream Stopped">
    </div>

    <div class="log-container" id="log-box">
        <div style="color: #2ecc71; margin-bottom: 10px;">Recent Detections:</div>
        <div id="log-content">Waiting for data...</div>
    </div>

    <script>
        let refreshInterval = null;

        function startImageRefresh() {
            if (!refreshInterval) {
                refreshInterval = setInterval(function() {
                    var img = document.getElementById('live-img');
                    img.src = "/static/latest.jpg?t=" + new Date().getTime();
                    updateLogs();
                }, 1000);
            }
        }

        function stopImageRefresh() {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }

        function updateLogs() {
            fetch('/get_logs').then(res => res.json()).then(data => {
                const content = document.getElementById('log-content');
                content.innerHTML = data.logs.map(line => `<div class="log-entry">${line}</div>`).join('');
            });
        }

        function sendCommand(action) {
            document.getElementById('status-text').innerText = "...";
            fetch('/' + action, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('status-text').innerText = data.status;
                if (data.status === "RUNNING") startImageRefresh();
                else stopImageRefresh();
            });
        }

        window.onload = () => { sendCommand('check'); };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/get_logs')
def get_logs():
    # Reads the last 5 lines from the text file
    try:
        with open("detected_plates.txt", "r") as f:
            lines = f.readlines()
            return jsonify({"logs": lines[-5:][::-1]}) # Last 5, reversed
    except:
        return jsonify({"logs": ["No logs found yet."]})

@app.route('/check', methods=['POST'])
def check():
    status = "RUNNING" if process and process.poll() is None else "STOPPED"
    return jsonify({"status": status})

@app.route('/start', methods=['POST'])
def start_script():
    global process
    if process is None or process.poll() is not None:
        process = subprocess.Popen(['uv', 'run', 'python', 'detection.py'], start_new_session=True)
    return jsonify({"status": "RUNNING"})

@app.route('/stop', methods=['POST'])
def stop_script():
    global process
    if process and process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
        process = None
    return jsonify({"status": "STOPPED"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
