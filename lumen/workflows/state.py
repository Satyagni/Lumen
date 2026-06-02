from PySide6.QtCore import QObject, Signal
from lumen.core.logger import logger

class AppState(QObject):
    """Centralized, signal-driven state manager for Lumen."""

    # Qt Signals for cross-component communication
    image_loaded = Signal(str)            # Emitted when an image is loaded (path)
    theme_changed = Signal(str)           # Emitted when theme switches ("dark" | "light")
    page_changed = Signal(str)            # Emitted on screen navigation (page_name)
    workflow_selected = Signal(str)       # Emitted when workflow configuration changes
    analysis_started = Signal()           # Emitted when mock analysis is triggered
    analysis_completed = Signal(dict)     # Emitted when analysis completes with mock data
    sidebar_toggled = Signal(bool)        # Emitted when sidebar expands/collapses
    backend_changed = Signal(str)         # Emitted when GPU/CPU backend status updates
    backend_preference_changed = Signal(str) # Emitted when backend preference changes
    
    # Phase 2D Viewer Control and State Reset Signals
    mask_opacity_changed = Signal(int)
    show_original_changed = Signal(bool)
    show_overlay_changed = Signal(bool)
    quality_mode_changed = Signal(str)
    
    # Phase 3A Batch Analysis Signals
    batch_started = Signal(int)                          # Emitted when batch starts (total_images)
    batch_progress_updated = Signal(int, int, str)       # Emitted on image processed (completed, failed, current_image_name)
    batch_finished = Signal(int, int, str)               # Emitted when batch completes (completed, failed, results_dir)
    batch_cancelled = Signal()                           # Emitted when batch is cancelled

    # Segmentation Method Signals
    segmentation_method_changed = Signal(str)

    def __init__(self):
        super().__init__()
        # Internal state store
        self._current_image_path = None
        self._current_workflow = None
        self._current_theme = "dark"
        self._current_page = "home"
        self._current_backend = "CPU"
        self._sidebar_collapsed = False
        self._analysis_results = None
        
        # Phase 2D Transient Viewer Settings
        self._mask_opacity = 40
        self._show_original_image = True
        self._show_segmentation_overlay = True
        self._quality_mode = "Balanced"
        
        # Phase 3A Batch Analysis Variables
        self._is_batch_active = False
        self._batch_progress = 0
        self._batch_status = ""
        self._batch_results_dir = ""

        # Segmentation Settings
        self._segmentation_method = "AI Segmentation"

    # Getters and Setters with Logging and Signaling

    @property
    def current_image_path(self) -> str:
        return self._current_image_path

    @current_image_path.setter
    def current_image_path(self, path: str):
        if self._current_image_path != path:
            if path:
                self.reset_analysis_session()
            self._current_image_path = path
            logger.info("AppState: image path updated: %s", path)
            self.image_loaded.emit(path or "")

    @property
    def current_workflow(self) -> str:
        return self._current_workflow

    @current_workflow.setter
    def current_workflow(self, workflow_name: str):
        if self._current_workflow != workflow_name:
            self._current_workflow = workflow_name
            logger.info("AppState: workflow selected: %s", workflow_name)
            self.workflow_selected.emit(workflow_name or "")

    @property
    def current_theme(self) -> str:
        return self._current_theme

    @current_theme.setter
    def current_theme(self, theme: str):
        if self._current_theme != theme:
            self._current_theme = theme
            logger.info("AppState: theme changed to %s", theme)
            self.theme_changed.emit(theme)

    @property
    def current_page(self) -> str:
        return self._current_page

    @current_page.setter
    def current_page(self, page: str):
        if self._current_page != page:
            self._current_page = page
            logger.info("AppState: active page routing: %s", page)
            self.page_changed.emit(page)

    @property
    def current_backend(self) -> str:
        return self._current_backend

    @current_backend.setter
    def current_backend(self, backend: str):
        if self._current_backend != backend:
            self._current_backend = backend
            logger.info("AppState: backend active changed to: %s", backend)
            self.backend_changed.emit(backend)

    @property
    def backend_preference(self) -> str:
        from lumen.core.config import config
        return config.backend_preference

    @backend_preference.setter
    def backend_preference(self, val: str):
        from lumen.core.config import config
        if config.backend_preference != val:
            config.backend_preference = val
            logger.info("AppState: backend preference updated to: %s", val)
            self.backend_preference_changed.emit(val)

    @property
    def sidebar_collapsed(self) -> bool:
        return self._sidebar_collapsed

    @sidebar_collapsed.setter
    def sidebar_collapsed(self, collapsed: bool):
        if self._sidebar_collapsed != collapsed:
            self._sidebar_collapsed = collapsed
            logger.debug("AppState: sidebar collapsed toggled to: %s", collapsed)
            self.sidebar_toggled.emit(collapsed)

    @property
    def analysis_results(self) -> dict:
        return self._analysis_results

    @analysis_results.setter
    def analysis_results(self, results: dict):
        self._analysis_results = results
        logger.info("AppState: analysis results populated.")
        self.analysis_completed.emit(results or {})

    @property
    def current_image_metadata(self) -> dict:
        """Retrieves active metadata directory from ImageManager."""
        from lumen.processing.image_manager import image_manager
        return image_manager.get_metadata()

    def reset_session(self):
        """Clears current active image caches, analysis results, and active workflow selections."""
        from lumen.processing.image_manager import image_manager
        image_manager.clear_cache()
        self._current_image_path = None
        self._current_workflow = None
        self._analysis_results = None
        
        # Broadcast changes to reload layouts
        self.image_loaded.emit("")
        self.workflow_selected.emit("")
        logger.info("AppState: Central session reset.")

    # Phase 2D: Transient Viewer Settings Properties
    @property
    def mask_opacity(self) -> int:
        return self._mask_opacity

    @mask_opacity.setter
    def mask_opacity(self, val: int):
        if self._mask_opacity != val:
            self._mask_opacity = val
            logger.debug("AppState: mask opacity updated to: %d", val)
            self.mask_opacity_changed.emit(val)

    @property
    def show_original_image(self) -> bool:
        return self._show_original_image

    @show_original_image.setter
    def show_original_image(self, val: bool):
        if self._show_original_image != val:
            self._show_original_image = val
            logger.debug("AppState: show_original_image updated to: %s", val)
            self.show_original_changed.emit(val)

    @property
    def show_segmentation_overlay(self) -> bool:
        return self._show_segmentation_overlay

    @show_segmentation_overlay.setter
    def show_segmentation_overlay(self, val: bool):
        if self._show_segmentation_overlay != val:
            self._show_segmentation_overlay = val
            logger.debug("AppState: show_segmentation_overlay updated to: %s", val)
            self.show_overlay_changed.emit(val)

    @property
    def quality_mode(self) -> str:
        return self._quality_mode

    @quality_mode.setter
    def quality_mode(self, val: str):
        if self._quality_mode != val:
            self._quality_mode = val
            logger.debug("AppState: quality_mode updated to: %s", val)
            self.quality_mode_changed.emit(val)

    @property
    def is_batch_active(self) -> bool:
        return self._is_batch_active

    @is_batch_active.setter
    def is_batch_active(self, val: bool):
        if self._is_batch_active != val:
            self._is_batch_active = val
            logger.info("AppState: is_batch_active changed to: %s", val)

    @property
    def batch_progress(self) -> int:
        return self._batch_progress

    @batch_progress.setter
    def batch_progress(self, val: int):
        if self._batch_progress != val:
            self._batch_progress = val

    @property
    def batch_status(self) -> str:
        return self._batch_status

    @batch_status.setter
    def batch_status(self, val: str):
        if self._batch_status != val:
            self._batch_status = val

    @property
    def batch_results_dir(self) -> str:
        return self._batch_results_dir

    @batch_results_dir.setter
    def batch_results_dir(self, val: str):
        if self._batch_results_dir != val:
            self._batch_results_dir = val

    @property
    def segmentation_method(self) -> str:
        return self._segmentation_method

    @segmentation_method.setter
    def segmentation_method(self, val: str):
        if self._segmentation_method != val:
            self._segmentation_method = val
            logger.info("AppState: segmentation_method updated: %s", val)
            self.segmentation_method_changed.emit(val)

    def reset_analysis_session(self):
        """Resets all transient analysis configurations to defaults for a new image."""
        logger.info("AppState: Resetting transient analysis session variables.")
        self._quality_mode = "Balanced"
        self._mask_opacity = 40
        self._show_original_image = True
        self._show_segmentation_overlay = True
        self._analysis_results = None
        
        self._segmentation_method = "AI Segmentation"
        
        self.quality_mode_changed.emit("Balanced")
        self.mask_opacity_changed.emit(40)
        self.show_original_changed.emit(True)
        self.show_overlay_changed.emit(True)
        self.segmentation_method_changed.emit("AI Segmentation")
        self.analysis_completed.emit({})

# Global instance of AppState
state = AppState()
