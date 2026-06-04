from lumen.core.logger import logger
from lumen.workflows.state import state

class NavigationService:
    """Decoupled navigation service matching page name routes to UI stack selectors."""

    # Map name identifier keys to stack indexes
    PAGE_MAP = {
        "home": 0,
        "upload": 1,
        "analysis": 2,
        "results": 3,
        "settings": 4,
        "batch_progress": 5,
        "batch_explorer": 6
    }

    def __init__(self):
        logger.info("NavigationService initialized.")

    def navigate_to(self, page_name: str) -> bool:
        """Triggers application layout routing to named page."""
        if page_name not in self.PAGE_MAP:
            logger.error("NavigationService: Unknown page routing request: '%s'", page_name)
            return False

        # Intercept upload page request if there is a running/paused/completed batch
        from lumen.processing.batch_manager import batch_manager
        if page_name == "upload" and batch_manager.lifecycle_state in ("RUNNING", "PAUSED", "COMPLETED"):
            logger.info("NavigationService: Redirecting 'upload' request to 'batch_progress' due to active batch state (%s)", batch_manager.lifecycle_state)
            page_name = "batch_progress"

        # Intercept transitions away from the analysis page if state.is_dirty is True
        if state.current_page == "analysis" and page_name != "analysis" and state.is_dirty:
            state.revert_to_last_committed_state()

        logger.debug("NavigationService: Redirecting to target: '%s'", page_name)
        state.current_page = page_name
        return True

    def get_index(self, page_name: str) -> int:
        """Resolves string page name to integer widget index."""
        return self.PAGE_MAP.get(page_name, 0)

    def get_page_name(self, index: int) -> str:
        """Resolves integer widget index back to string page name."""
        for name, idx in self.PAGE_MAP.items():
            if idx == index:
                return name
        return "home"

# Global Navigation Service instance
navigation_service = NavigationService()
