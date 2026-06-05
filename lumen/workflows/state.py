from typing import Optional, Any
from PySide6.QtCore import QObject, Signal
from lumen.core.logger import logger

class AnalysisSession:
    """Stores persistent state for a specific microscopy analysis session."""
    def __init__(self, image_path: str, origin_type: str = "single", batch_origin_context: Optional[str] = None):
        self.image_path = image_path
        self.origin_type = origin_type
        self.batch_origin_context = batch_origin_context
        self.dirty = False
        self.analysis_results = None
        self.committed_results = None
        self.quality_mode = "Balanced"
        self.mask_opacity = 40
        self.show_original_image = True
        self.show_segmentation_overlay = True
        self.segmentation_method = "AI Segmentation"
        self.segmentation_model = "Auto"
        self.current_workflow = None
        self.viewer_state = None  # Dict of {transform, h_scroll, v_scroll, initial_fit_scale, zoom_touched}

class BatchResultSession:
    """Stores persistent state for a specific batch explorer session."""
    def __init__(self, batch_results_dir: str):
        self.batch_results_dir = batch_results_dir
        self.records = []
        self.manifest_data = {}
        self.selected_filename = None
        self.search_text = ""
        self.sort_by = "Alphabetical"
        self.show_original_image = True
        self.show_segmentation_overlay = True
        self.mask_opacity = 40
        self.viewer_state = None  # Dict of {transform, h_scroll, v_scroll, initial_fit_scale, zoom_touched}

class WorkspaceManager(QObject):
    """Coordinates the lifecycle of active page sessions and context invalidation."""
    
    def __init__(self):
        super().__init__()
        self._analysis_sessions = {}  # dict of (image_path, origin_type) -> AnalysisSession
        self._batch_sessions = {}     # dict of batch_results_dir -> BatchResultSession
        self._active_analysis_path = None
        self._active_analysis_origin = None
        self._active_batch_dir = None

    def _normalize_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return path.replace('\\', '/')

    def get_analysis_session(self, image_path: str, origin_type: Optional[str] = None) -> Optional[AnalysisSession]:
        """Gets the analysis session for the given image path and origin type."""
        norm_path = self._normalize_path(image_path)
        if not norm_path:
            return None
        if origin_type is None:
            if norm_path == self._active_analysis_path and self._active_analysis_origin is not None:
                origin_type = self._active_analysis_origin
            else:
                for (p, o), sess in self._analysis_sessions.items():
                    if p == norm_path:
                        origin_type = o
                        break
                if origin_type is None:
                    origin_type = "single"
        session = self._analysis_sessions.get((norm_path, origin_type))
        if session:
            self._active_analysis_path = norm_path
            self._active_analysis_origin = origin_type
        return session

    def start_analysis_session(self, image_path: str, origin_type: str = "single", batch_origin_context: Optional[str] = None) -> AnalysisSession:
        """Gets or starts a session for the given image path and origin type."""
        norm_path = self._normalize_path(image_path)
        key = (norm_path, origin_type)
        if key not in self._analysis_sessions:
            self._analysis_sessions[key] = AnalysisSession(image_path, origin_type, batch_origin_context)
            logger.info("WorkspaceManager: Started fresh AnalysisSession for %s with origin %s", norm_path, origin_type)
        self._active_analysis_path = norm_path
        self._active_analysis_origin = origin_type
        return self._analysis_sessions[key]

    def reset_analysis_session(self, image_path: str = None, origin_type: Optional[str] = None):
        """Clears the specified analysis session, or all if none provided."""
        if image_path:
            norm_path = self._normalize_path(image_path)
            if origin_type is not None:
                self._analysis_sessions.pop((norm_path, origin_type), None)
                if self._active_analysis_path == norm_path and self._active_analysis_origin == origin_type:
                    self._active_analysis_path = None
                    self._active_analysis_origin = None
            else:
                keys_to_remove = [k for k in self._analysis_sessions.keys() if k[0] == norm_path]
                for k in keys_to_remove:
                    self._analysis_sessions.pop(k, None)
                if self._active_analysis_path == norm_path:
                    self._active_analysis_path = None
                    self._active_analysis_origin = None
            logger.info("WorkspaceManager: Cleared AnalysisSession for %s", norm_path)
        else:
            self._analysis_sessions.clear()
            self._active_analysis_path = None
            self._active_analysis_origin = None
            logger.info("WorkspaceManager: All AnalysisSessions cleared.")

    def get_batch_session(self, batch_results_dir: str) -> Optional[BatchResultSession]:
        """Gets the batch session for the given batch directory."""
        norm_path = self._normalize_path(batch_results_dir)
        return self._batch_sessions.get(norm_path)

    def start_batch_session(self, batch_results_dir: str) -> BatchResultSession:
        """Gets or starts a session for the given batch directory."""
        norm_path = self._normalize_path(batch_results_dir)
        if norm_path not in self._batch_sessions:
            self._batch_sessions[norm_path] = BatchResultSession(batch_results_dir)
            logger.info("WorkspaceManager: Started fresh BatchResultSession for %s", norm_path)
        self._active_batch_dir = norm_path
        return self._batch_sessions[norm_path]

    def reset_batch_session(self, batch_results_dir: str = None):
        """Clears the specified batch session, or all if none provided."""
        if batch_results_dir:
            norm_path = self._normalize_path(batch_results_dir)
            self._batch_sessions.pop(norm_path, None)
            if self._active_batch_dir == norm_path:
                self._active_batch_dir = None
            logger.info("WorkspaceManager: Cleared BatchResultSession for %s", norm_path)
        else:
            self._batch_sessions.clear()
            self._active_batch_dir = None
            logger.info("WorkspaceManager: All BatchResultSessions cleared.")


class AppState(QObject):
    """Centralized, signal-driven state manager for Lumen."""

    # Qt Signals for cross-component communication
    image_loaded = Signal(str)            # Emitted when an image is loaded (path)
    theme_changed = Signal(str)           # Emitted when theme switches ("dark" | "light")
    page_changed = Signal(str)            # Emitted on screen navigation (page_name)
    workflow_selected = Signal(str)       # Emitted when workflow configuration changes
    analysis_started = Signal()           # Emitted when mock analysis is triggered
    analysis_completed = Signal(dict)     # Emitted when analysis completes with mock data
    manual_mask_saved = Signal(str)           # Emitted when manual mask changes are saved (image_path)
    analysis_results_updated = Signal(dict)   # Emitted when analysis results are updated manually (results)
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
    segmentation_model_changed = Signal(str)
    dirty_state_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        # Workspace Session Manager
        self.workspace_manager = WorkspaceManager()
        
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
            if self.is_dirty:
                self.revert_to_last_committed_state()
            if path:
                self.reset_analysis_session()
            self._current_image_path = path
            logger.info("AppState: image path updated: %s", path)
            self.image_loaded.emit(path or "")

    @property
    def is_dirty(self) -> bool:
        if self._current_image_path:
            session = self.workspace_manager.get_analysis_session(self._current_image_path, self.workspace_manager._active_analysis_origin)
            if session:
                return getattr(session, "dirty", False)
        return False

    @is_dirty.setter
    def is_dirty(self, val: bool):
        if self._current_image_path:
            session = self.workspace_manager.get_analysis_session(self._current_image_path, self.workspace_manager._active_analysis_origin)
            if session:
                old_val = getattr(session, "dirty", False)
                if old_val != val:
                    session.dirty = val
                    logger.info("AppState: dirty state updated to: %s", val)
                    self.dirty_state_changed.emit(val)

    @property
    def current_origin_type(self) -> str:
        if self._current_image_path:
            session = self.workspace_manager.get_analysis_session(self._current_image_path, self.workspace_manager._active_analysis_origin)
            if session:
                return getattr(session, "origin_type", "single")
        return "single"

    @property
    def current_batch_origin_context(self) -> Optional[str]:
        if self._current_image_path:
            session = self.workspace_manager.get_analysis_session(self._current_image_path, self.workspace_manager._active_analysis_origin)
            if session:
                return getattr(session, "batch_origin_context", None)
        return None

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

    @property
    def segmentation_model(self) -> str:
        from lumen.core.config import config
        return config.segmentation_model

    @segmentation_model.setter
    def segmentation_model(self, val: str):
        from lumen.core.config import config
        if config.segmentation_model != val:
            config.segmentation_model = val
            logger.info("AppState: segmentation model preference updated to: %s", val)
            self.segmentation_model_changed.emit(val)

    def revert_to_last_committed_state(self):
        """Reverts the active analysis session to its last committed results."""
        if self._current_image_path:
            session = self.workspace_manager.get_analysis_session(self._current_image_path, self.workspace_manager._active_analysis_origin)
            if session and session.dirty:
                committed = session.committed_results
                session.analysis_results = committed
                self._analysis_results = committed
                session.dirty = False
                logger.info("AppState: Reverted to last committed state automatically.")
                self.dirty_state_changed.emit(False)
                self.analysis_results_updated.emit(committed or {})
                self.manual_mask_saved.emit(self._current_image_path)

    def reset_analysis_session(self):
        """Resets all transient analysis configurations to defaults for a new image."""
        logger.info("AppState: Resetting transient analysis session variables.")
        self._quality_mode = "Balanced"
        self._mask_opacity = 40
        self._show_original_image = True
        self._show_segmentation_overlay = True
        self._analysis_results = None
        
        self._segmentation_method = "AI Segmentation"
        
        # Clear workspace manager session as well
        self.workspace_manager.reset_analysis_session()
        
        self.quality_mode_changed.emit("Balanced")
        self.mask_opacity_changed.emit(40)
        self.show_original_changed.emit(True)
        self.show_overlay_changed.emit(True)
        self.segmentation_method_changed.emit("AI Segmentation")
        self.analysis_completed.emit({})

# Global instance of AppState
state = AppState()
