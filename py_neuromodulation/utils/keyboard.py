import asyncio
import sys
from typing import Callable

if sys.platform.startswith("win"):
    import msvcrt
else:
    import termios
    import tty


class KeyboardListener:
    def __init__(self, event_callback: tuple[str, Callable] | None = None):
        self.callbacks = {}
        self.running = False

        if event_callback is not None:
            self.on_press(*event_callback)

    def on_press(self, key, callback):
        self.callbacks[key] = callback

    async def _windows_listener(self):
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8").lower()
                if key in self.callbacks:
                    await self.callbacks[key]()
            await asyncio.sleep(0.01)

    async def _unix_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                key = sys.stdin.read(1).lower()
                if key in self.callbacks:
                    await self.callbacks[key]()
                await asyncio.sleep(0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def start(self):
        self.running = True
        if sys.platform.startswith("win"):
            await self._windows_listener()
        else:
            await self._unix_listener()

    def stop(self):
        self.running = False
