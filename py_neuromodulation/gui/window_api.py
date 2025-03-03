import webview
import threading
import time


# API class implementing all the methods available in the PyWebView Window object
# API Reference: https://pywebview.flowrl.com/guide/api.html#webview-window
class WindowAPI:
    def __init__(self):
        self.window = None
        self.is_resizing = False
        self.start_x = 0
        self.start_y = 0
        self.start_width = 0
        self.start_height = 0

    # Function to store the reference to the PyWevView window
    def register_window(self, window: webview.Window):
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
        dialog_type=webview.OPEN_DIALOG,
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
