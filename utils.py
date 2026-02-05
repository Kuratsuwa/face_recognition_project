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
        return os.path.dirname(sys.executable)
    else:
        # Normal script
        return os.path.dirname(os.path.abspath(__file__))

