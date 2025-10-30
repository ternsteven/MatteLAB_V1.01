# main_auto_port.py (auto-select port like your '副本', then open browser when ready)
import threading
import time
import webbrowser
import socket
import os

from src.birefnet_app.ui_gradio import create_interface
from src.birefnet_app.settings import ensure_dirs

HOST = "0.0.0.0"  # listen on all interfaces
PORT = None       # let Gradio pick a free port (>=7860)
PORT_RANGE = (7860, 7890)

def wait_for_open(host: str, port: int, timeout: float = 90.0, interval: float = 0.5) -> bool:
    import socket
    import time
    deadline = time.time() + timeout
    # Try connecting via localhost regardless of HOST binding
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=interval):
                return True
        except OSError:
            time.sleep(interval)
    return False

def find_listening_port(candidates):
    # Probe candidates and return the first one that accepts TCP
    for p in candidates:
        if wait_for_open("127.0.0.1", p, timeout=0.1, interval=0.05):
            return p
    return None

if __name__ == "__main__":
    ensure_dirs()
    demo = create_interface()

    # Launch Gradio without specifying a port; it will choose a free one.
    t = threading.Thread(
        target=lambda: demo.launch(
            server_name=HOST,
            server_port=PORT,   # auto-pick
            share=False,
            inbrowser=False,    # open manually when ready
            show_error=True,
            quiet=True
        ),
        daemon=True
    )
    t.start()

    # Try to detect the chosen port by probing common range (7860-7890)
    candidates = list(range(PORT_RANGE[0], PORT_RANGE[1] + 1))
    detected = None
    # Allow some warm-up time before probing
    time.sleep(2.0)
    for _ in range(120):  # up to ~60 seconds (120 * 0.5)
        detected = find_listening_port(candidates)
        if detected is not None:
            break
        time.sleep(0.5)

    if detected is not None:
        url = f"http://127.0.0.1:{detected}"
        print(f"✅ 自动检测端口: {detected}，正在打开浏览器… {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️ 打开浏览器失败：{e}\n请手动访问 {url}")
    else:
        print("⚠️ 未能自动检测端口，请查看控制台日志或手动访问上方 Gradio 输出的本地 URL。")

    t.join()


