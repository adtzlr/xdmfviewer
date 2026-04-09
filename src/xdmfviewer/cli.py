"""Command-line entrypoint for xdmfviewer."""

from __future__ import annotations

import sys

try:
    from qtpy.QtGui import QColor, QFont, QPainter, QPixmap
    from qtpy.QtWidgets import QApplication, QSplashScreen
    from qtpy.QtCore import Qt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This application requires a Qt binding. Install with: pip install PySide6"
    ) from exc

from .version import __version__


def _create_splash_pixmap() -> QPixmap:
    """Create the splash screen artwork."""
    pixmap = QPixmap(640, 360)
    pixmap.fill(QColor("#f4f6f8"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    painter.setPen(QColor("#d9dee5"))
    painter.setBrush(QColor("#ffffff"))
    painter.drawRoundedRect(24, 24, 592, 312, 18, 18)

    painter.setPen(QColor("#1f2937"))
    title_font = QFont()
    title_font.setPointSize(30)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.drawText(pixmap.rect().adjusted(0, -45, 0, -20), Qt.AlignCenter, "xdmfviewer")

    painter.setPen(QColor("#465566"))
    subtitle_font = QFont()
    subtitle_font.setPointSize(11)
    painter.setFont(subtitle_font)
    painter.drawText(
        pixmap.rect().adjusted(0, 30, 0, 40),
        Qt.AlignHCenter | Qt.AlignTop,
        "Loading XDMF viewer...",
    )

    painter.setPen(QColor("#6b7280"))
    version_font = QFont()
    version_font.setPointSize(10)
    painter.setFont(version_font)
    painter.drawText(
        pixmap.rect().adjusted(0, 55, 0, 95),
        Qt.AlignHCenter | Qt.AlignTop,
        f"Version {__version__}",
    )

    painter.end()
    return pixmap


def _show_splash(app: QApplication) -> QSplashScreen:
    """Show a centered splash screen before importing the heavy GUI module."""
    splash = QSplashScreen(_create_splash_pixmap())
    screen = app.primaryScreen()
    if screen is not None:
        geometry = screen.availableGeometry()
        splash.move(geometry.center() - splash.rect().center())
    splash.show()
    app.processEvents()
    return splash


def main() -> None:
    """Run the GUI application from the console script."""
    app = QApplication(sys.argv)
    splash = _show_splash(app)

    from .app import run as gui_run

    raise SystemExit(gui_run(app, splash))


if __name__ == "__main__":
    main()
