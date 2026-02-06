import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_app_dir():
    """ Get the directory of the executable or script """
    if getattr(sys, 'frozen', False):
        # Bundled executable
        exe_dir = os.path.dirname(sys.executable)
        # On macOS, the executable is inside Omokage.app/Contents/MacOS/
        # We want to save configs/output next to Omokage.app
        if sys.platform == 'darwin' and ".app/Contents/MacOS" in exe_dir:
            return os.path.abspath(os.path.join(exe_dir, "../../.."))
        return exe_dir
    else:
        # Normal script
        return os.path.dirname(os.path.abspath(__file__))

