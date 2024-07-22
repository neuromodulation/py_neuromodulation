import webview
from window_api import WindowAPI


class AppWindow:
    def create_window(self) -> None:
        api = WindowAPI()

        window = webview.create_window(
            title="PyNeuromodulation GUI",
            url="http://localhost:5173",
            min_size=(1200, 800),
            frameless=True,
            resizable=True,
            easy_drag=False,
            js_api=api,
        )

        api.register_window(window)

        webview.start(
            debug=True,
            ssl=True,
        )
