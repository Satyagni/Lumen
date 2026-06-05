from pathlib import Path
import lumen.storage.database as db_module
from lumen.core.logger import logger

class AppConfig:
    """Convenience wrapper around database settings for window geometry and preferences."""

    @property
    def theme(self) -> str:
        """Returns the current theme selection ('dark' or 'light'). Defaults to 'dark'."""
        if db_module.db:
            return db_module.db.get_setting("theme", "dark")
        return "dark"

    @theme.setter
    def theme(self, val: str):
        if db_module.db:
            db_module.db.set_setting("theme", val)

    @property
    def window_geometry(self) -> tuple:
        """Returns (width, height, x, y) for the window dimensions. Defaults to standard resolution."""
        if not db_module.db:
            return (1280, 800, -1, -1)
        
        w = int(db_module.db.get_setting("window_width", "1280"))
        h = int(db_module.db.get_setting("window_height", "800"))
        x = int(db_module.db.get_setting("window_x", "-1"))
        y = int(db_module.db.get_setting("window_y", "-1"))
        return (w, h, x, y)

    def save_window_geometry(self, w: int, h: int, x: int, y: int):
        """Saves current window size and coordinates."""
        if db_module.db:
            db_module.db.set_setting("window_width", w)
            db_module.db.set_setting("window_height", h)
            db_module.db.set_setting("window_x", x)
            db_module.db.set_setting("window_y", y)

    @property
    def last_directory(self) -> str:
        """Returns the last successfully opened file explorer path."""
        if db_module.db:
            return db_module.db.get_setting("last_directory", "")
        return ""

    @last_directory.setter
    def last_directory(self, path: str):
        if db_module.db:
            db_module.db.set_setting("last_directory", path)

    @property
    def sidebar_collapsed(self) -> bool:
        """Returns whether the sidebar was collapsed. Defaults to False (Expanded)."""
        if db_module.db:
            return db_module.db.get_setting("sidebar_collapsed", "False") == "True"
        return False

    @sidebar_collapsed.setter
    def sidebar_collapsed(self, val: bool):
        if db_module.db:
            db_module.db.set_setting("sidebar_collapsed", "True" if val else "False")

    @property
    def backend_preference(self) -> str:
        """Returns the saved compute backend preference. Defaults to 'Auto'."""
        if db_module.db:
            return db_module.db.get_setting("backend_preference", "Auto")
        return "Auto"

    @backend_preference.setter
    def backend_preference(self, val: str):
        if db_module.db:
            db_module.db.set_setting("backend_preference", val)

    @property
    def recent_files(self) -> list:
        """Returns the list of recently uploaded files (list of dicts)."""
        import json
        if db_module.db:
            data = db_module.db.get_setting("recent_files", "[]")
            try:
                return json.loads(data)
            except Exception:
                return []
        return []

    def add_recent_file(self, path: str, workflow_id: str):
        """Adds a path to the recent files list in database."""
        import json
        if not path:
            return
        
        # Normalize path
        path = path.replace('\\', '/')
        
        recents = self.recent_files
        # Remove if path already exists to move it to the top
        recents = [r for r in recents if r.get("path") != path]
        
        # Insert at the beginning
        recents.insert(0, {"path": path, "workflow_id": workflow_id})
        
        # Limit to 10 entries
        recents = recents[:10]
        
        if db_module.db:
            db_module.db.set_setting("recent_files", json.dumps(recents))

    def clear_recent_files(self):
        """Clears all recent files from the database."""
        if db_module.db:
            db_module.db.set_setting("recent_files", "[]")

    @property
    def segmentation_model(self) -> str:
        """Returns the saved segmentation model preference. Defaults to 'Auto'."""
        if db_module.db:
            return db_module.db.get_setting("segmentation_model", "Auto")
        return "Auto"

    @segmentation_model.setter
    def segmentation_model(self, val: str):
        if db_module.db:
            db_module.db.set_setting("segmentation_model", val)

# Instantiate global config wrapper
config = AppConfig()

