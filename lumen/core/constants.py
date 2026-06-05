import os
from pathlib import Path

# Application Metadata
APP_NAME = "Lumen"
APP_VERSION = "0.2.0"
APP_SUBTITLE = "AI-Assisted Biological Image Analysis"

# Directory Structure
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
LUMEN_DIR = WORKSPACE_DIR / "lumen"
LOGS_DIR = WORKSPACE_DIR / "logs"

# Assets and Themes
ASSETS_DIR = LUMEN_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
THEMES_DIR = LUMEN_DIR / "ui" / "themes"

# File Validation
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# Database Settings
DB_FILE = WORKSPACE_DIR / "lumen.db"

# Sidebar Dimensions
SIDEBAR_EXPANDED_WIDTH = 250
SIDEBAR_COLLAPSED_WIDTH = 76
ANIMATION_DURATION_MS = 250
