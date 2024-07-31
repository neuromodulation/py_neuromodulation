from app_window import WebViewWindow
from app_manager import AppManager
import time

if __name__ == "__main__":
    app_manager = AppManager(debug=True)
    app_manager.run_app()

    logger = app_manager.logger

    # HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

    logger.info("Starting PyWebView window...")
    window = WebViewWindow()  # Only works from main thread
    window.window.events.closed += app_manager.terminate_app
    window.start()

    while True:
        if app_manager.shutdown_complete:
            break
        time.sleep(0.1)

    logger.info("All processes cleaned up. Exiting...")
