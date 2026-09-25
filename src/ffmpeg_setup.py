"""
Locate an ffmpeg binary and make it discoverable on PATH.

Whisper (and moviepy) both shell out to a real `ffmpeg` executable, so simply
having ffmpeg buried inside a Python package is not enough. This module finds
the copy that ships with imageio-ffmpeg and prepends its folder to PATH, which
means the project runs without a separate system-wide ffmpeg install.
"""

import os
import shutil
from typing import Optional

_done = False


def find_ffmpeg(name: str = "ffmpeg") -> Optional[str]:
    """Return a path to ffmpeg, or None if it cannot be found."""
    found = shutil.which(name)
    if found:
        return found

    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if name == "ffmpeg" and path and os.path.exists(path):
            return path
    except Exception:
        pass

    return None


def ensure_ffmpeg_on_path() -> Optional[str]:
    """
    Make sure `ffmpeg` is resolvable by name for subprocess calls.

    Returns the resolved ffmpeg path, or None if none is available.
    Idempotent, and safe to call from multiple modules.
    """
    global _done

    exe = find_ffmpeg("ffmpeg")
    if exe is None:
        return None

    if not _done:
        folder = os.path.dirname(exe)
        # A Windows binary needs ffmpeg.exe; other tools may look for ffmpeg.
        current = os.environ.get("PATH", "")
        parts = [p for p in current.split(os.pathsep) if p]
        if folder not in parts:
            os.environ["PATH"] = folder + os.pathsep + os.pathsep.join(parts)

        # Some libraries resolve the binary as "ffmpeg" without the extension.
        if not shutil.which("ffmpeg"):
            link = os.path.join(folder, "ffmpeg")
            try:
                if not os.path.exists(link):
                    os.symlink(exe, link)
            except (OSError, NotImplementedError, AttributeError):
                # Symlinks may be unavailable; the .exe path is enough for
                # callers that use find_ffmpeg() directly.
                pass

        _done = True

    return exe
