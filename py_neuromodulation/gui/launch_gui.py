import asyncio
import threading
from app_window import AppWindow


async def read_stream(stream, prefix):
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f"{prefix}: {line.decode().strip()}")


async def run_process(name, program, **kwargs):
    process = await asyncio.create_subprocess_shell(
        program,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    await asyncio.gather(
        read_stream(process.stdout, f"{name} [stdout]"),
        read_stream(process.stderr, f"{name} [stderr]"),
    )
    print(f"{name} exited with {process.returncode}")


async def run_async_tasks():
    await asyncio.gather(
        # Run Flask in Python unbuffered mode, otherwise output is not shown properly
        run_process("Flask", "python -u py_neuromodulation/gui/launch_backend.py"),
        run_process("Vite", "bun run dev", cwd="gui_dev"),
    )


def run_async_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_async_tasks())


if __name__ == "__main__":
    # Start the asyncio tasks in a new event loop
    async_thread = threading.Thread(target=run_async_loop, daemon=True)
    async_thread.start()

    # Launch the PyWebView window in the main thread because it is blocking
    window = AppWindow()
    window.create_window()

    async_thread.join()
