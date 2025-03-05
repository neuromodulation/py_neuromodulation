import threading
import time
import logging
import requests

from .app_utils import ansi_color, ansi_reset

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import webview

DEV = True

VITE_URL = "http://localhost:54321"
FASTAPI_URL = "http://localhost:50001"
APP_URL = VITE_URL if DEV else FASTAPI_URL

USER_AGENT = "PyNmWebView"


class WebViewWindow:
    def __init__(self, debug: bool = False) -> None:
        import webview

        self.debug = debug
        self.api = WebViewWindowApi()

        self.window = webview.create_window(
            title="PyNeuromodulation GUI",
            url=APP_URL,
            min_size=(1200, 800),
            frameless=True,
            resizable=True,
            easy_drag=False,
            js_api=self.api,
        )

        self.api.register_window(self.window)
        # Customize PyWebView logging format
        color = ansi_color(color="CYAN", styles=["BOLD"])
        logger = logging.getLogger("pywebview")
        formatter = logging.Formatter(
            f"{color}[PyWebView %(levelname)s (%(asctime)s)]:{ansi_reset} %(message)s",
            datefmt="%H:%M:%S",
        )
        logger.handlers[0].setFormatter(formatter)

    def start(self):
        import webview

        # Set timer to load SPA after a delay
        if DEV:
            self.wait_for_vite_server()

        webview.start(debug=self.debug, user_agent=USER_AGENT)

    def wait_for_vite_server(self):
        while True:
            if self.is_vite_server_running():
                break
            time.sleep(0.1)  # Wait for 1 second before checking again

    def is_vite_server_running(self):
        try:
            response = requests.get(VITE_URL, timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # Register event handlers
    def register_event_handler(self, event_type, handler):
        # https://pywebview.flowrl.com/guide/api.html#window-events
        match event_type:
            case "closed":
                self.window.events.closed += handler
            case "closing":
                self.window.events.closing += handler
            case "loaded":
                self.window.events.loaded += handler
            case "minimized":
                self.window.events.minimized += handler
            case "maximized":
                self.window.events.maximized += handler
            case "resized":
                self.window.events.resized += handler
            case "restore":
                self.window.events.restore += handler
            case "shown":
                self.window.events.shown += handler


# API class implementing all the methods available in the PyWebView Window object
# API Reference: https://pywebview.flowrl.com/guide/api.html#webview-window
class WebViewWindowApi:
    def __init__(self):
        self._window: "webview.Window"
        self.is_resizing = False
        self.start_x = 0
        self.start_y = 0
        self.start_width = 0
        self.start_height = 0

    # Function to store the reference to the PyWevView window
    def register_window(self, window: "webview.Window"):
        self._window = window

    # Functions to handle window resizing
    def start_resize(self, start_x, start_y):
        self.is_resizing = True
        self.start_x = start_x
        self.start_y = start_y
        self.start_width, self.start_height = self.get_size()
        threading.Thread(target=self._resize_loop).start()

    def stop_resize(self):
        self.is_resizing = False

    def update_resize(self, current_x, current_y):
        if self.is_resizing:
            dx = current_x - self.start_x
            dy = current_y - self.start_y
            new_width = max(self.start_width + dx, 200)  # Minimum width
            new_height = max(self.start_height + dy, 200)  # Minimum height
            self.set_size(int(new_width), int(new_height))

    def _resize_loop(self):
        while self.is_resizing:
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    # All API methods from the PyWebView docs
    def close_window(self):
        self._window.destroy()

    def maximize_window(self):
        self._window.maximize()

    def minimize_window(self):
        self._window.minimize()

    def restore_window(self):
        self._window.restore()

    def toggle_fullscreen(self):
        self._window.toggle_fullscreen()

    def set_title(self, title: str):
        self._window.title = title

    def get_position(self):
        return (self._window.x, self._window.y)

    def set_position(self, x: int, y: int):
        self._window.move(x, y)

    def get_size(self):
        return (self._window.width, self._window.height)

    def set_size(self, width: int, height: int):
        self._window.resize(width, height)

    def set_on_top(self, on_top: bool):
        self._window.on_top = on_top

    def show(self):
        self._window.show()

    def hide(self):
        self._window.hide()

    def create_file_dialog(
        self,
        dialog_type: int = 10,  # webview.OPEN_DIALOG,
        directory="",
        allow_multiple=False,
        save_filename="",
        file_types=(),
    ):
        return self._window.create_file_dialog(
            dialog_type, directory, allow_multiple, save_filename, file_types
        )

    def create_confirmation_dialog(self, title, message):
        return self._window.create_confirmation_dialog(title, message)

    def load_url(self, url):
        self._window.load_url(url)

    def load_html(self, content, base_uri: str):
        self._window.load_html(content, base_uri)

    def load_css(self, css):
        self._window.load_css(css)

    def evaluate_js(self, script, callback=None):
        return self._window.evaluate_js(script, callback)

    def get_current_url(self):
        return self._window.get_current_url()

    def get_elements(self, selector):
        return self._window.get_elements(selector)
