import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from lumen.core.constants import THEMES_DIR
from lumen.core.config import config
from lumen.core.logger import logger
from lumen.workflows.state import state

class ThemeService:
    """Manages application stylesheet rendering, switching, and database persistence."""

    def __init__(self):
        self._current_theme = config.theme
        state.current_theme = self._current_theme
        logger.info("ThemeService initialized. Starting theme: %s", self._current_theme)

    def apply_theme(self, theme_name: str = None) -> bool:
        """Loads and applies QSS stylesheet to the global QApplication instance."""
        app = QApplication.instance()
        if not app:
            logger.error("ThemeService: Cannot apply theme, QApplication instance is None.")
            return False

        if theme_name is None:
            theme_name = self._current_theme

        qss_file = THEMES_DIR / f"{theme_name}.qss"
        if not qss_file.exists():
            logger.error("ThemeService: Stylesheet file not found at: %s", qss_file)
            # Try to fall back to whatever is available or default
            return False

        try:
            with open(qss_file, "r", encoding="utf-8") as f:
                qss_style = f.read()
                app.setStyleSheet(qss_style)
                
            self._current_theme = theme_name
            config.theme = theme_name
            state.current_theme = theme_name
            logger.info("ThemeService: Applied theme '%s'", theme_name)
            return True
        except Exception as e:
            logger.error("ThemeService: Failed to read QSS file: %s", e)
            return False

    def toggle_theme(self):
        """Toggles between dark and light modes, saving changes dynamically."""
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self.apply_theme(new_theme)

    @property
    def current_theme(self) -> str:
        return self._current_theme

# Global theme service instance
theme_service = ThemeService()
