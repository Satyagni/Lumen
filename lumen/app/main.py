import sys
import faulthandler
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from lumen.ui.main_window import MainWindow
from lumen.core.logger import logger
from lumen.core.services.theme_service import theme_service

def start_app():
    """Initializes the QApplication loop and renders the MainWindow shell."""
    faulthandler.enable()
    logger.info("Initializing QApplication context.")
    
    # Configure High DPI scaling defaults for crisp rendering on high-res displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Lumen")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Lumen Biological")

    # Apply style theme (load stylesheet)
    logger.info("Applying stylesheet theme from ThemeService.")
    theme_service.apply_theme()

    # Launch MainWindow
    logger.info("Constructing MainWindow layout frame.")
    window = MainWindow()
    window.show()

    logger.info("Entering PyQt application main loop.")
    sys.exit(app.exec())

if __name__ == "__main__":
    start_app()
