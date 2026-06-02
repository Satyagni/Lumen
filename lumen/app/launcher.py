import os
import sys
from pathlib import Path

# 1. Dependency Preflight Check
try:
    import PySide6
except ImportError:
    print("\n" + "=" * 60)
    print(" ERROR: Required dependencies are missing.")
    print(" Please make sure PySide6 is installed.")
    print(" Run the following command to install dependencies:")
    print("\n     pip install -r requirements.txt")
    print("=" * 60 + "\n")
    sys.exit(1)

# 2. Modify sys.path to resolve imports from workspace root
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# 3. Bootstrap core systems
from lumen.core.constants import LOGS_DIR, DB_FILE
from lumen.core.logger import setup_logger

# Initialize structured logging
logger = setup_logger(LOGS_DIR)
logger.info("Lumen launcher started.")

from lumen.storage.database import init_db
from lumen.core.services.theme_service import theme_service

def bootstrap_application():
    """Initializes workspace directories, files, database configurations, and imports main.py."""
    logger.info("Initializing SQLite settings schema.")
    # Initialize SQLite database
    init_db(DB_FILE)
    
    # Initialize styling themes
    logger.info("Bootstrapping theme service configurations.")
    theme_service.apply_theme()

    # Import and execute app/main.py
    logger.info("Launching PyQt main application execution loop.")
    from lumen.app.main import start_app
    start_app()

if __name__ == "__main__":
    bootstrap_application()
