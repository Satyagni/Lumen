from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
from PySide6.QtCore import Slot, Qt
from lumen.core.config import config
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.ui.sidebar import SidebarWidget
from lumen.ui.navbar import NavbarWidget
from lumen.core.services.navigation_service import navigation_service

# Import pages
from lumen.pages.home_page import HomePage
from lumen.pages.upload_page import UploadPage
from lumen.pages.analysis_page import AnalysisPage
from lumen.pages.results_page import ResultsPage
from lumen.pages.settings_page import SettingsPage
from lumen.pages.batch_progress_page import BatchProgressPage
from lumen.pages.batch_explorer_page import BatchResultsExplorerPage

class MainWindow(QMainWindow):
    """The main desktop shell. Coordinates layout frame assembly and view stack changes."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumen")
        self.setObjectName("MainWindow")

        self._setup_layout()
        self._load_geometry()
        self._init_connections()

        # Route to home page by default on start
        navigation_service.navigate_to("home")

    def _setup_layout(self):
        # 1. Main Central Widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main Layout: Sidebar on Left, Content Container on Right
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 2. Add Collapsible Sidebar
        self.sidebar = SidebarWidget(self)
        self.main_layout.addWidget(self.sidebar)

        # 3. Right Side Workspace Container (Vertical)
        self.workspace_container = QWidget(self)
        self.workspace_container.setObjectName("WorkspaceContainer")
        self.workspace_layout = QVBoxLayout(self.workspace_container)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(0)
        self.main_layout.addWidget(self.workspace_container)

        # 4. Add Top Navbar header
        self.navbar = NavbarWidget(self)
        self.workspace_layout.addWidget(self.navbar)

        # 5. Add Stacked Widget for Page Views
        self.page_stack = QStackedWidget(self)
        self.page_stack.setObjectName("PageStack")
        self.workspace_layout.addWidget(self.page_stack)

        # 6. Initialize and Insert Pages matching NavigationService maps
        self.home_page = HomePage(self)
        self.upload_page = UploadPage(self)
        self.analysis_page = AnalysisPage(self)
        self.results_page = ResultsPage(self)
        self.settings_page = SettingsPage(self)
        self.batch_progress_page = BatchProgressPage(self)
        self.batch_explorer_page = BatchResultsExplorerPage(self)

        self.page_stack.addWidget(self.home_page)       # Index 0
        self.page_stack.addWidget(self.upload_page)     # Index 1
        self.page_stack.addWidget(self.analysis_page)   # Index 2
        self.page_stack.addWidget(self.results_page)    # Index 3
        self.page_stack.addWidget(self.settings_page)   # Index 4
        self.page_stack.addWidget(self.batch_progress_page) # Index 5
        self.page_stack.addWidget(self.batch_explorer_page) # Index 6

    def _load_geometry(self):
        """Loads and applies window dimensions saved in the SQLite configuration database."""
        w, h, x, y = config.window_geometry
        
        # Clamp geometry to primary screen's available space to prevent window clipping
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(w, avail.width())
            h = min(h, avail.height())

        # Set minimum size for general usability
        self.setMinimumSize(1024, 700)

        self.resize(w, h)
        if x >= 0 and y >= 0:
            self.move(x, y)
        else:
            logger.debug("MainWindow: Position coordinates fallback to center.")

    def _init_connections(self):
        # Listen to state changes to switch active widget index
        state.page_changed.connect(self._on_page_changed)

    @Slot(str)
    def _on_page_changed(self, page_name: str):
        """Switches stacked widgets index according to routing state."""
        target_idx = navigation_service.get_index(page_name)
        self.page_stack.setCurrentIndex(target_idx)
        logger.debug("MainWindow: Workspace stacked widget shifted to index %d", target_idx)

        # Force a geometry layout refresh during page transitions
        active_widget = self.page_stack.currentWidget()
        if active_widget:
            def force_layout_refresh(widget):
                if not widget:
                    return
                lay = widget.layout
                layout = lay() if callable(lay) else lay
                if layout:
                    layout.invalidate()
                    layout.activate()
                from PySide6.QtWidgets import QWidget
                for child in widget.findChildren(QWidget):
                    child.updateGeometry()
                    child_lay = child.layout
                    child_layout = child_lay() if callable(child_lay) else child_lay
                    if child_layout:
                        child_layout.invalidate()
                        child_layout.activate()
            
            force_layout_refresh(active_widget)
            active_widget.updateGeometry()
            active_widget.update()
            
        self.page_stack.updateGeometry()
        self.page_stack.update()

    def closeEvent(self, event):
        """Saves current window coordinates on closure to preserve geometry settings."""
        if state.is_dirty:
            state.revert_to_last_committed_state()
            event.accept()

        try:
            geometry = self.geometry()
            config.save_window_geometry(
                geometry.width(),
                geometry.height(),
                geometry.x(),
                geometry.y()
            )
            logger.info("MainWindow: Saved window size configuration.")
        except Exception as e:
            logger.error("MainWindow: Failed to write window geometry settings: %s", e)
        
        super().closeEvent(event)
