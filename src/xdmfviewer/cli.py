"""Command-line entrypoint for xdmfviewer."""

from __future__ import annotations

import sys
from contextlib import suppress

from .version import __version__


class BootstrapSplash:
    """Show a lightweight native splash before Qt finishes loading."""

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("xdmfviewer")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        if sys.platform == "win32":
            with suppress(Exception):
                self._root.attributes("-toolwindow", True)

        width = 420
        height = 160
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        x_pos = max(0, (screen_width - width) // 2)
        y_pos = max(0, (screen_height - height) // 2)
        self._root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        self._root.configure(bg="#f4f6f8")

        container = tk.Frame(self._root, bg="#f4f6f8", highlightthickness=1, highlightbackground="#d9dee5")
        container.pack(fill="both", expand=True, padx=1, pady=1)

        title = tk.Label(
            container,
            text="xdmfviewer",
            bg="#f4f6f8",
            fg="#1f2937",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(pady=(28, 6))

        subtitle = tk.Label(
            container,
            text="Loading XDMF viewer...",
            bg="#f4f6f8",
            fg="#465566",
            font=("Segoe UI", 10),
        )
        subtitle.pack()

        version = tk.Label(
            container,
            text=f"Version {__version__}",
            bg="#f4f6f8",
            fg="#6b7280",
            font=("Segoe UI", 9),
        )
        version.pack(pady=(10, 0))

        self._root.update_idletasks()
        self._root.update()

    def finish(self, _window: object) -> None:
        """Close the splash once the main window is ready."""
        self.close()

    def close(self) -> None:
        """Destroy the splash window."""
        if getattr(self, "_root", None) is None:
            return
        with suppress(Exception):
            self._root.destroy()
        self._root = None


def _create_bootstrap_splash() -> BootstrapSplash | None:
    try:
        return BootstrapSplash()
    except Exception:
        return None


def main() -> None:
    """Run the GUI application from the console script."""
    splash = _create_bootstrap_splash()

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from .app import run as gui_run

    raise SystemExit(gui_run(app, splash))


if __name__ == "__main__":
    main()
