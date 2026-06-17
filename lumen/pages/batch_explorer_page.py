import os
import csv
import time
import numpy as np
import tifffile
import PIL.Image
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QComboBox, QPushButton, QSlider, QCheckBox,
    QScrollArea, QFrame, QFileDialog, QMessageBox, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage, QCursor

from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.processing.batch_manager import batch_manager
from lumen.core.services.navigation_service import navigation_service
from lumen.core.services.theme_service import theme_service
from lumen.pages.analysis_page import InteractiveImageViewer

def load_microscopy_pixmap(file_path: str) -> QPixmap:
    """Loads a raw high-depth or standard microscopy image from file path with display normalization."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        raw_arr = tifffile.imread(file_path)
    else:
        with PIL.Image.open(file_path) as pil_img:
            raw_arr = np.asarray(pil_img)
            
    if raw_arr is None or raw_arr.size == 0:
        return QPixmap()

    # Apply 1/99 percentile display normalization
    if raw_arr.ndim == 2:
        p1 = np.percentile(raw_arr, 1)
        p99 = np.percentile(raw_arr, 99)
        if p99 <= p1:
            p1 = np.min(raw_arr)
            p99 = np.max(raw_arr)
        if p99 <= p1:
            display_arr = np.zeros_like(raw_arr, dtype=np.uint8)
        else:
            display_arr = (np.clip((raw_arr - p1) / (p99 - p1), 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        channels = raw_arr.shape[2]
        display_arr = np.zeros_like(raw_arr, dtype=np.uint8)
        for c in range(channels):
            channel = raw_arr[..., c]
            p1 = np.percentile(channel, 1)
            p99 = np.percentile(channel, 99)
            if p99 <= p1:
                p1 = np.min(channel)
                p99 = np.max(channel)
            if p99 > p1:
                display_arr[..., c] = (np.clip((channel - p1) / (p99 - p1), 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                display_arr[..., c] = 0
                
    h, w = display_arr.shape[:2]
    if display_arr.ndim == 2:
        qimg = QImage(display_arr.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
    else:
        c = display_arr.shape[2]
        if c == 1:
            qimg = QImage(display_arr[..., 0].tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
        elif c == 3:
            qimg = QImage(display_arr.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
        elif c == 4:
            qimg = QImage(display_arr.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()
        else:
            qimg = QImage()
            
    return QPixmap.fromImage(qimg)


class BatchPlaceholderWidget(QFrame):
    """Elegant placeholder shown when no batch is loaded or active."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BatchPlaceholder")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        # Scientific Icon
        self.icon_lbl = QLabel("📂")
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_lbl)

        # Title
        self.title_lbl = QLabel("No Batch Loaded")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_lbl)

        # Description
        self.desc_lbl = QLabel("No batch loaded. Run a batch analysis to browse results.")
        self.desc_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.desc_lbl)
        
        self.sync_theme(theme_service.current_theme)

    def sync_theme(self, theme_name: str):
        if theme_name == "light":
            self.setStyleSheet("""
                #BatchPlaceholder {
                    background-color: #FFFFFF;
                    border: 1px dashed #D1D5DB;
                    border-radius: 8px;
                }
            """)
            self.icon_lbl.setStyleSheet("font-size: 40px; color: #4F46E5; margin-bottom: 4px; background: transparent;")
            self.title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #111827; background: transparent;")
            self.desc_lbl.setStyleSheet("font-size: 12px; color: #4B5563; background: transparent;")
        else:
            self.setStyleSheet("""
                #BatchPlaceholder {
                    background-color: #131317;
                    border: 1px dashed #2B2B35;
                    border-radius: 8px;
                }
            """)
            self.icon_lbl.setStyleSheet("font-size: 40px; color: #6366F1; margin-bottom: 4px; background: transparent;")
            self.title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF; background: transparent;")
            self.desc_lbl.setStyleSheet("font-size: 12px; color: #9CA3AF; background: transparent;")


class BatchResultsExplorerPage(QWidget):
    """Integrated workspace for completed batch analysis review."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_batch_dir = None
        self.batch_dir = None
        self.records = []
        self.manifest_data = {}
        self._explorer_transaction_active = False
        self._pending_search_repopulate = False
        self._pending_session_save = False

        # Debounce timer for search queries
        from PySide6.QtCore import QTimer
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(250) # 250ms debounce delay

        self._setup_ui()
        self._init_connections()
        self._sync_theme()

    def _setup_ui(self):
        self.page_layout = QVBoxLayout(self)
        self.page_layout.setObjectName("PageVerticalLayout")
        self.page_layout.setContentsMargins(20, 20, 20, 20)
        self.page_layout.setSpacing(12)

        from lumen.ui.workspace_switcher import WorkspaceSwitcher
        self.workspace_switcher = WorkspaceSwitcher("batch")
        self.page_layout.addWidget(self.workspace_switcher)

        self.main_layout = QHBoxLayout()
        self.main_layout.setObjectName("PageContainer")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(16)

        # ----------------------------------------------------
        # 1. Left Panel: Image Navigator
        # ----------------------------------------------------
        self.left_panel = QFrame()
        self.left_panel.setObjectName("ExplorerLeftPanel")
        self.left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # Header Title
        left_header = QLabel("Batch Explorer")
        left_header.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        left_layout.addWidget(left_header)

        # Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search filename...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: #131317;
                border: 1px solid #2B2B35;
                border-radius: 4px;
                padding: 6px 10px;
                color: #FFFFFF;
                font-size: 11px;
            }
        """)
        left_layout.addWidget(self.search_bar)

        # Sort Dropdown
        sort_layout = QHBoxLayout()
        sort_lbl = QLabel("Sort by:")
        sort_lbl.setStyleSheet("font-size: 10px; color: #9CA3AF; text-transform: uppercase; font-weight: bold;")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Alphabetical", "Cell Count", "Processing Time", "Status"])
        self.sort_combo.setCursor(QCursor(Qt.PointingHandCursor))
        self.sort_combo.setStyleSheet("""
            QComboBox {
                background-color: #131317;
                border: 1px solid #2B2B35;
                border-radius: 4px;
                padding: 4px 8px;
                color: #FFFFFF;
                font-size: 11px;
            }
        """)
        sort_layout.addWidget(sort_lbl)
        sort_layout.addWidget(self.sort_combo, 1)
        left_layout.addLayout(sort_layout)

        # Navigator List Widget
        self.navigator_list = QListWidget()
        self.navigator_list.setStyleSheet("""
            QListWidget {
                background-color: #131317;
                border: 1px solid #2B2B35;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                border-bottom: 1px solid #1C1C22;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #312E81;
            }
        """)
        left_layout.addWidget(self.navigator_list, 1)

        # Navigation Buttons (Previous/Next)
        nav_btns_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.prev_btn.setProperty("class", "SecondaryButton")
        self.next_btn = QPushButton("Next")
        self.next_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.next_btn.setProperty("class", "SecondaryButton")
        nav_btns_layout.addWidget(self.prev_btn)
        nav_btns_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_btns_layout)

        self.main_layout.addWidget(self.left_panel)

        # ----------------------------------------------------
        # 2. Center Panel: Reused InteractiveImageViewer
        # ----------------------------------------------------
        self.center_panel = QFrame()
        center_layout = QVBoxLayout(self.center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)

        # Viewer Canvas
        self.image_viewer = InteractiveImageViewer(self)
        center_layout.addWidget(self.image_viewer, 1)

        # Viewer Control Options Bar
        self.viewer_controls_bar = QFrame()
        self.viewer_controls_bar.setObjectName("ViewerControlsBar")
        self.viewer_controls_bar.setStyleSheet("""
            #ViewerControlsBar {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 6px;
                padding: 6px 12px;
            }
        """)
        controls_layout = QHBoxLayout(self.viewer_controls_bar)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(16)

        self.show_original_chk = QCheckBox("Show Original Image")
        self.show_original_chk.setChecked(True)
        self.show_original_chk.setCursor(Qt.PointingHandCursor)
        self.show_original_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")

        self.show_overlay_chk = QCheckBox("Show Segmentation Overlay")
        self.show_overlay_chk.setChecked(True)
        self.show_overlay_chk.setCursor(Qt.PointingHandCursor)
        self.show_overlay_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")

        controls_layout.addWidget(self.show_original_chk)
        controls_layout.addWidget(self.show_overlay_chk)

        sep = QLabel("|")
        sep.setStyleSheet("color: #2B2B35;")
        controls_layout.addWidget(sep)

        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(6)
        slider_label = QLabel("Mask Opacity:")
        slider_label.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.setCursor(Qt.PointingHandCursor)
        
        self.opacity_val_lbl = QLabel("40%")
        self.opacity_val_lbl.setStyleSheet("font-size: 11px; color: #E5E7EB; min-width: 30px;")
        
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.opacity_slider)
        slider_layout.addWidget(self.opacity_val_lbl)
        controls_layout.addLayout(slider_layout)

        controls_layout.addStretch(1)

        # Reset View button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.reset_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #131317;
                border: 1px solid #2B2B35;
                border-radius: 4px;
                padding: 4px 10px;
                color: #E5E7EB;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #24242B;
            }
        """)
        controls_layout.addWidget(self.reset_view_btn)

        center_layout.addWidget(self.viewer_controls_bar)
        self.main_layout.addWidget(self.center_panel, 1)

        # ----------------------------------------------------
        # 3. Right Panel: Results & Metadata
        # ----------------------------------------------------
        self.right_panel = QFrame()
        self.right_panel.setObjectName("ExplorerRightPanel")
        self.right_panel.setFixedWidth(260)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)

        # Header Title
        right_header = QLabel("Quantification Detail")
        right_header.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        right_layout.addWidget(right_header)

        # Scroll Area for Metadata
        self.meta_scroll = QScrollArea()
        self.meta_scroll.setWidgetResizable(True)
        self.meta_scroll.setStyleSheet("border: none; background: transparent;")
        
        self.meta_container = QWidget()
        self.meta_container.setStyleSheet("background: transparent;")
        self.meta_grid = QGridLayout(self.meta_container)
        self.meta_grid.setContentsMargins(0, 0, 0, 0)
        self.meta_grid.setSpacing(12)
        
        self.meta_scroll.setWidget(self.meta_container)
        right_layout.addWidget(self.meta_scroll, 1)

        # Action Buttons (Open in Analysis Workspace, Return to Upload)
        self.open_analysis_btn = QPushButton("Open in Analysis Workspace")
        self.open_analysis_btn.setObjectName("OpenAnalysisWorkspaceButton")
        self.open_analysis_btn.setProperty("class", "PrimaryButton")
        self.open_analysis_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.open_analysis_btn.setEnabled(False)
        right_layout.addWidget(self.open_analysis_btn)

        self.back_upload_btn = QPushButton("Return to Upload")
        self.back_upload_btn.setProperty("class", "SecondaryButton")
        self.back_upload_btn.setCursor(QCursor(Qt.PointingHandCursor))
        right_layout.addWidget(self.back_upload_btn)

        self.main_layout.addWidget(self.right_panel)

        self._placeholder = BatchPlaceholderWidget(self)
        self.main_layout.addWidget(self._placeholder, 1)

        self.page_layout.addLayout(self.main_layout, 1)

    def _init_connections(self):
        # State signals
        state.page_changed.connect(self._on_page_changed)
        state.theme_changed.connect(self._sync_theme)

        # Search and sorting (debounced)
        self.search_bar.textChanged.connect(self._on_search_text_changed)
        self.search_timer.timeout.connect(self._on_search_changed)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)

        # Selection handler
        self.navigator_list.currentItemChanged.connect(self._on_selection_changed)

        # Navigation shortcuts
        self.prev_btn.clicked.connect(self._on_prev_clicked)
        self.next_btn.clicked.connect(self._on_next_clicked)

        # Visual controls
        self.show_original_chk.toggled.connect(self._on_show_original_changed)
        self.show_overlay_chk.toggled.connect(self._on_show_overlay_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        self.reset_view_btn.clicked.connect(self._on_reset_view_clicked)

        # Action gateways
        self.open_analysis_btn.clicked.connect(self._on_open_analysis_clicked)
        self.back_upload_btn.clicked.connect(self._on_back_upload_clicked)

        # Double click to open analysis
        self.navigator_list.itemDoubleClicked.connect(self._on_open_analysis_clicked)

        # Invalidate batch cache when a new batch starts or finishes, or when a manual mask is saved
        batch_manager.batch_started.connect(self._invalidate_batch_cache)
        batch_manager.batch_finished.connect(self._invalidate_batch_cache)
        state.manual_mask_saved.connect(self._invalidate_batch_cache)

        # Initial boot check
        self._load_from_state()

    @Slot(str)
    def _on_page_changed(self, page_name: str):
        if page_name == "batch_explorer":
            self._load_from_state()
        else:
            self._save_to_session()

    @Slot()
    def _invalidate_batch_cache(self, *args, **kwargs):
        logger.info("BatchExplorer: Invalidating loaded batch directory cache due to batch state update.")
        self._loaded_batch_dir = None
        results_dir = None
        for arg in args:
            if isinstance(arg, str):
                if os.path.isdir(arg):
                    results_dir = arg
                    break
                elif os.path.isfile(arg):
                    anal_sess = state.workspace_manager.get_analysis_session(arg)
                    if anal_sess and anal_sess.batch_origin_context:
                        results_dir = anal_sess.batch_origin_context
                        break
        if not results_dir:
            if state.is_batch_active:
                results_dir = batch_manager.output_dir
        if not results_dir:
            results_dir = state.batch_results_dir
        if results_dir:
            state.workspace_manager.reset_batch_session(results_dir)

    def _ui_matches_session(self, session) -> bool:
        if not session:
            return False
        
        # If records are empty but session has them, they don't match
        if not self.records and session.records:
            return False
            
        # Check widgets
        if self.search_bar.text() != session.search_text:
            return False
        if self.sort_combo.currentText() != session.sort_by:
            return False
        if self.opacity_slider.value() != session.mask_opacity:
            return False
            
        # Check if the selected item matches session's selected filename
        curr_item = self.navigator_list.currentItem()
        curr_filename = None
        if curr_item:
            rec = curr_item.data(Qt.UserRole)
            curr_filename = rec.get("image_name") if rec else None
            
        if curr_filename != session.selected_filename:
            return False
            
        return True

    def _reset_explorer_state(self):
        logger.info("BatchExplorer: Performing full state reset and UI reconstruction.")
        # 1. Clear navigator list safely
        self.navigator_list.blockSignals(True)
        self.navigator_list.clearSelection()
        self.navigator_list.clear()
        self.navigator_list.blockSignals(False)

        # 2. Reset records and manifest
        self.records = []
        self.manifest_data = {}

        # 3. Clear image viewer
        self.image_viewer.clear()
        self.image_viewer.set_analysis_results(None)

        # 4. Explicitly rebuild the metadata panel widgets and container
        if hasattr(self, "meta_container") and self.meta_container is not None:
            # Delete child widgets safely first
            if hasattr(self, "meta_grid") and self.meta_grid is not None:
                for i in reversed(range(self.meta_grid.count())):
                    item = self.meta_grid.itemAt(i)
                    if item:
                        w = item.widget()
                        if w is not None:
                            w.setParent(None)
                            w.deleteLater()
                self.meta_grid.deleteLater()
                self.meta_grid = None
            self.meta_container.setParent(None)
            self.meta_container.deleteLater()
            self.meta_container = None

        # Reconstruct fresh metadata container & layout
        self.meta_container = QWidget()
        self.meta_container.setStyleSheet("background: transparent;")
        self.meta_grid = QGridLayout(self.meta_container)
        self.meta_grid.setContentsMargins(0, 0, 0, 0)
        self.meta_grid.setSpacing(12)
        self.meta_scroll.setWidget(self.meta_container)

        # 5. Reset cached references & UI state
        self._loaded_batch_dir = None
        self._temp_restore_viewer_state = None
        self.open_analysis_btn.setEnabled(False)

    def _load_from_state(self):
        results_dir = state.batch_results_dir
        if not results_dir or not os.path.exists(results_dir):
            self.left_panel.setVisible(False)
            self.center_panel.setVisible(False)
            self.right_panel.setVisible(False)
            self._placeholder.setVisible(True)
            self._loaded_batch_dir = None
            return

        session = state.workspace_manager.get_batch_session(results_dir)
        if (hasattr(self, "_loaded_batch_dir") and 
            self._loaded_batch_dir == results_dir and 
            self._ui_matches_session(session)):
            logger.info("BatchExplorer: Batch directory %s already loaded and matches session. Skipping load.", results_dir)
            return

        # Perform full explorer invalidation and UI reconstruction before load
        self._reset_explorer_state()

        self._placeholder.setVisible(False)
        self.left_panel.setVisible(True)
        self.center_panel.setVisible(True)
        self.right_panel.setVisible(True)

        logger.info("BatchExplorer: Loading batch results from %s", results_dir)
        self.batch_dir = Path(results_dir)
        
        session = state.workspace_manager.get_batch_session(results_dir)
        if session:
            logger.info("BatchExplorer: Restoring from persistent batch session state.")
            self._restore_from_session(session)
        else:
            state.workspace_manager.start_batch_session(results_dir)
            # Determine files and settings
            summary_csv = self.batch_dir / "batch_summary.csv"
            manifest_json = self.batch_dir / "run_manifest.json"
            
            self.records = []
            self.manifest_data = {}
            
            # Load run manifest
            if manifest_json.exists():
                try:
                    import json
                    with open(manifest_json, mode="r", encoding="utf-8") as f:
                        self.manifest_data = json.load(f)
                except Exception as e:
                    logger.error("BatchExplorer: Failed to load manifest: %s", e)

            # Load summary CSV
            if summary_csv.exists():
                try:
                    with open(summary_csv, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            self.records.append(row)
                except Exception as e:
                    logger.error("BatchExplorer: Failed to load CSV summary: %s", e)
            
            # Sync edited status from manifest into records
            if self.manifest_data and "images" in self.manifest_data:
                edited_images = {img.get("image_name") for img in self.manifest_data["images"] if img.get("edited") in [True, "True", "true", 1, "1"]}
                for rec in self.records:
                    if rec.get("image_name") in edited_images:
                        rec["edited"] = True
            
            # Clear search and reset list
            self.search_bar.blockSignals(True)
            self.search_bar.clear()
            self.search_bar.blockSignals(False)
            
            self._populate_list(select_default=True)

        self._loaded_batch_dir = results_dir

    def _save_to_session(self):
        if getattr(self, "_explorer_transaction_active", False):
            self._pending_session_save = True
            return
        results_dir = state.batch_results_dir
        if not results_dir:
            return
            
        session = state.workspace_manager.start_batch_session(results_dir)
        session.records = self.records
        session.manifest_data = self.manifest_data
        session.search_text = self.search_bar.text()
        session.sort_by = self.sort_combo.currentText()
        session.show_original_image = self.show_original_chk.isChecked()
        session.show_segmentation_overlay = self.show_overlay_chk.isChecked()
        session.mask_opacity = self.opacity_slider.value()
        
        curr_item = self.navigator_list.currentItem()
        if curr_item:
            rec = curr_item.data(Qt.UserRole)
            session.selected_filename = rec.get("image_name") if rec else None
        else:
            session.selected_filename = None
            
        v = self.image_viewer
        if v.pixmap_item and not v.pixmap_item.pixmap().isNull():
            session.viewer_state = {
                "transform": v.transform(),
                "h_scroll": v.horizontalScrollBar().value(),
                "v_scroll": v.verticalScrollBar().value(),
                "initial_fit_scale": v._initial_fit_scale,
                "zoom_touched": v._zoom_touched
            }
        else:
            session.viewer_state = None
        logger.info("BatchExplorer: Saved batch session to workspace manager.")

    def _restore_from_session(self, session):
        self.records = session.records
        self.manifest_data = session.manifest_data
        if session.batch_results_dir:
            self.batch_dir = Path(session.batch_results_dir)
        
        # Refresh modified records from disk to ensure we load the latest committed metrics
        if session.batch_results_dir and os.path.exists(session.batch_results_dir):
            summary_csv = Path(session.batch_results_dir) / "batch_summary.csv"
            manifest_json = Path(session.batch_results_dir) / "run_manifest.json"
            
            disk_records = {}
            if summary_csv.exists():
                try:
                    import csv
                    with open(summary_csv, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get("image_name"):
                                disk_records[row["image_name"]] = row
                except Exception as e:
                    logger.error("BatchExplorer: Failed to load fresh CSV summary during restore: %s", e)
                    
            disk_manifest_images = {}
            if manifest_json.exists():
                try:
                    import json
                    with open(manifest_json, mode="r", encoding="utf-8") as f:
                        disk_mdata = json.load(f)
                    if "images" in disk_mdata:
                        for img in disk_mdata["images"]:
                            if img.get("image_name"):
                                disk_manifest_images[img["image_name"]] = img
                except Exception as e:
                    logger.error("BatchExplorer: Failed to load fresh manifest during restore: %s", e)

            # Update in-memory session records with latest disk metrics for edited items
            if self.records:
                for rec in self.records:
                    image_name = rec.get("image_name")
                    is_edited_on_disk = False
                    if image_name in disk_records:
                        is_edited_on_disk = disk_records[image_name].get("edited") in [True, "True", "true", 1, "1"]
                    is_edited_in_session = rec.get("edited") in [True, "True", "true", 1, "1"]
                    
                    if is_edited_on_disk or is_edited_in_session:
                        rec["edited"] = True
                        if image_name in disk_records:
                            rec.update(disk_records[image_name])
                            rec["edited"] = True

            # Update manifest_data with latest disk metrics for edited items
            if self.manifest_data and "images" in self.manifest_data:
                for img in self.manifest_data["images"]:
                    image_name = img.get("image_name")
                    is_edited_on_disk = False
                    if image_name in disk_manifest_images:
                        is_edited_on_disk = disk_manifest_images[image_name].get("edited") in [True, "True", "true", 1, "1"]
                    is_edited_in_session = img.get("edited") in [True, "True", "true", 1, "1"]
                    
                    if is_edited_on_disk or is_edited_in_session:
                        img["edited"] = True
                        if image_name in disk_manifest_images:
                            img.update(disk_manifest_images[image_name])
                            img["edited"] = True
        
        # Block signals to prevent premature triggers
        self.navigator_list.blockSignals(True)
        self.search_bar.blockSignals(True)
        self.sort_combo.blockSignals(True)
        self.show_original_chk.blockSignals(True)
        self.show_overlay_chk.blockSignals(True)
        self.opacity_slider.blockSignals(True)
        
        self.search_bar.setText(session.search_text)
        self.sort_combo.setCurrentText(session.sort_by)
        self.show_original_chk.setChecked(session.show_original_image)
        self.show_overlay_chk.setChecked(session.show_segmentation_overlay)
        self.opacity_slider.setValue(session.mask_opacity)
        self.opacity_val_lbl.setText(f"{session.mask_opacity}%")
        
        self.navigator_list.blockSignals(False)
        self.search_bar.blockSignals(False)
        self.sort_combo.blockSignals(False)
        self.show_original_chk.blockSignals(False)
        self.show_overlay_chk.blockSignals(False)
        self.opacity_slider.blockSignals(False)
        
        # Populate navigator list with filters applied, but do NOT select first item by default
        self._populate_list(select_default=False)
        
        # Set temp viewer state for the selection change trigger
        self._temp_restore_viewer_state = session.viewer_state
        
        # Select the saved item
        if session.selected_filename:
            found_item = None
            for idx in range(self.navigator_list.count()):
                item = self.navigator_list.item(idx)
                rec = item.data(Qt.UserRole)
                if rec and rec.get("image_name") == session.selected_filename:
                    found_item = item
                    break
            if found_item:
                self.navigator_list.setCurrentItem(found_item)
                self.navigator_list.scrollToItem(found_item)
            else:
                if self.navigator_list.count() > 0:
                    self.navigator_list.setCurrentRow(0)
                else:
                    self._on_selection_changed(None, None)
        else:
            if self.navigator_list.count() > 0:
                self.navigator_list.setCurrentRow(0)
            else:
                self._on_selection_changed(None, None)

    def _on_search_text_changed(self, text: str):
        if getattr(self, "_explorer_transaction_active", False):
            self._pending_search_repopulate = True
            return
        self.search_timer.start()

    def _on_search_changed(self):
        if getattr(self, "_explorer_transaction_active", False):
            self._pending_search_repopulate = True
            return
        self._populate_list()
        self._save_to_session()

    def _on_sort_changed(self, text: str):
        if getattr(self, "_explorer_transaction_active", False):
            return
        self._populate_list()
        self._save_to_session()

    def _on_show_original_changed(self, checked: bool):
        if getattr(self, "_explorer_transaction_active", False):
            return
        self.image_viewer.set_show_original(checked)
        self._save_to_session()

    def _on_show_overlay_changed(self, checked: bool):
        if getattr(self, "_explorer_transaction_active", False):
            return
        self.image_viewer.set_show_overlay(checked)
        self._save_to_session()

    def _populate_list(self, select_default=True):
        if getattr(self, "_explorer_transaction_active", False):
            return
        self._explorer_transaction_active = True
        self._pending_search_repopulate = False
        self._pending_session_save = False
        
        # 1. Save current selection before clearing
        selected_name = None
        curr_item = self.navigator_list.currentItem()
        if curr_item:
            rec = curr_item.data(Qt.UserRole)
            selected_name = rec.get("image_name") if rec else None

        from PySide6.QtCore import QSignalBlocker
        blocker = QSignalBlocker(self.navigator_list)
        sel_model = self.navigator_list.selectionModel()
        sel_blocker = QSignalBlocker(sel_model) if sel_model else None

        target_item = None

        try:
            self.navigator_list.clear()
            self.image_viewer.clear()
            self.image_viewer.set_analysis_results(None)
            self._clear_metadata_panel()
            self.open_analysis_btn.setEnabled(False)

            search_text = self.search_bar.text().lower()
            sort_by = self.sort_combo.currentText()

            # Sort copy of records list
            sorted_records = list(self.records)
            if sort_by == "Alphabetical":
                sorted_records.sort(key=lambda x: x.get("image_name", "").lower())
            elif sort_by == "Cell Count":
                sorted_records.sort(key=lambda x: int(x.get("cell_count") or 0), reverse=True)
            elif sort_by == "Processing Time":
                sorted_records.sort(key=lambda x: float(x.get("processing_time_s") or 0.0), reverse=True)
            elif sort_by == "Status":
                sorted_records.sort(key=lambda x: x.get("status", ""))

            theme = theme_service.current_theme
            
            for rec in sorted_records:
                if not rec:
                    continue
                filename = rec.get("image_name", "")
                if not filename:
                    continue
                if search_text and search_text not in filename.lower():
                    continue

                item = QListWidgetItem()
                item.setData(Qt.UserRole, rec)

                # Custom list item row widget
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(8, 6, 8, 6)
                row_layout.setSpacing(8)

                is_edited = rec.get("edited") in [True, "True", "true", 1, "1"]
                display_name = f"{filename} ✏ Modified" if is_edited else filename
                name_lbl = QLabel(display_name)
                if theme == "light":
                    name_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #111827;")
                else:
                    name_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #FFFFFF;")

                status = rec.get("status", "SUCCESS")
                status_lbl = QLabel()
                if "SUCCESS" in status:
                    status_lbl.setText("✅ SUCCESS")
                    status_lbl.setStyleSheet("font-size: 9px; color: #34D399; background: #065F46; padding: 2px 6px; border-radius: 4px; font-weight: bold;")
                elif "SKIP" in status:
                    status_lbl.setText("⏭ SKIPPED")
                    status_lbl.setStyleSheet("font-size: 9px; color: #F59E0B; background: #78350F; padding: 2px 6px; border-radius: 4px; font-weight: bold;")
                else:
                    status_lbl.setText("⚠ FAILED")
                    status_lbl.setStyleSheet("font-size: 9px; color: #F87171; background: #991B1B; padding: 2px 6px; border-radius: 4px; font-weight: bold;")

                row_layout.addWidget(name_lbl)
                row_layout.addStretch(1)
                row_layout.addWidget(status_lbl)

                self.navigator_list.addItem(item)
                self.navigator_list.setItemWidget(item, row_widget)
                
                # Check if this is the item to restore selection to
                if selected_name and filename == selected_name:
                    target_item = item
        finally:
            self._explorer_transaction_active = False
            if sel_blocker:
                sel_blocker.unblock()
            blocker.unblock()

        # 2. Restore selection or select default
        if select_default:
            if target_item and self.navigator_list.row(target_item) >= 0:
                self.navigator_list.setCurrentItem(target_item)
            elif self.navigator_list.count() > 0:
                self.navigator_list.setCurrentRow(0)
            else:
                # Clear viewer and metadata explicitly if no items match
                self._on_selection_changed(None, None)

        # 3. Process deferred tasks
        if getattr(self, "_pending_search_repopulate", False):
            self._pending_search_repopulate = False
            self.search_timer.start()
        elif getattr(self, "_pending_session_save", False):
            self._pending_session_save = False
            self._save_to_session()

    def _on_selection_changed(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        if getattr(self, "_explorer_transaction_active", False):
            return
        if not current_item:
            self._clear_metadata_panel()
            self.image_viewer.clear()
            self.image_viewer.set_analysis_results(None)
            self.open_analysis_btn.setEnabled(False)
            return

        record = current_item.data(Qt.UserRole)
        if not record:
            self._clear_metadata_panel()
            self.image_viewer.clear()
            self.image_viewer.set_analysis_results(None)
            self.open_analysis_btn.setEnabled(False)
            return

        self._load_record_details(record)
        self._save_to_session()

    def _load_record_details(self, record: dict):
        if getattr(self, "_explorer_transaction_active", False):
            return
        self._clear_metadata_panel()
        self.image_viewer.clear()
        self.image_viewer.set_analysis_results(None)

        image_name = record["image_name"]
        status = record.get("status", "SUCCESS")
        
        # 1. Update Right Metadata Grid Panel
        self._render_metadata_fields(record)

        # Enable Open in Analysis Workspace button only for non-failed runs
        is_failed = "FAILED" in status
        self.open_analysis_btn.setEnabled(not is_failed)

        if is_failed:
            logger.warning("BatchExplorer: Selected record is in FAILED state.")
            return

        if not self.batch_dir:
            logger.warning("BatchExplorer: batch_dir is None, skipping record details loading.")
            return

        # 2. Asynchronously / Lazily load images and overlays
        # Resolve raw microscopy path
        raw_image_path = self.batch_dir.parent / image_name
        if not raw_image_path.exists():
            # Recursive scan
            for root, dirs, files in os.walk(str(self.batch_dir.parent)):
                if "batch_results" in root:
                    continue
                if image_name in files:
                    raw_image_path = Path(root) / image_name
                    break

        # Load raw labels TIFF
        labels_path = self.batch_dir / image_name / f"{image_name}_labels_raw.tif"
        masks = None
        if labels_path.exists():
            try:
                masks = tifffile.imread(str(labels_path))
            except Exception as e:
                logger.error("BatchExplorer: Failed to load raw masks: %s", e)

        # Load metrics CSV to reconstruct cell_metrics dictionary
        cell_metrics = {}
        csv_path = self.batch_dir / image_name / f"{image_name}_cell_metrics.csv"
        if csv_path.exists():
            try:
                with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 5:
                            cell_id = int(row[0])
                            area_px = int(row[1])
                            diam = float(row[2])
                            cx = float(row[3])
                            cy = float(row[4])
                            diam_est = float(row[5]) if len(row) > 5 else diam
                            cell_metrics[cell_id] = {
                                "area_px": area_px,
                                "diameter_px": diam,
                                "centroid": (cx, cy),
                                "diameter_estimate": diam_est
                            }
            except Exception as e:
                logger.error("BatchExplorer: Failed to parse cell CSV: %s", e)

        # Reconstruct results_dict context
        results_context = {
            "masks": masks,
            "cell_metrics": cell_metrics
        }
        self.image_viewer.set_analysis_results(results_context)

        # Load background Pixmap (with overlay fallback if raw is missing)
        pixmap = QPixmap()
        try:
            if raw_image_path.exists():
                pixmap = load_microscopy_pixmap(str(raw_image_path))
        except Exception as e:
            logger.error("BatchExplorer: Normalization load failed: %s", e)
            
        if pixmap.isNull():
            # Fallback to preview PNG
            preview_png_path = self.batch_dir / image_name / f"{image_name}_overlay_preview.png"
            if preview_png_path.exists():
                pixmap = QPixmap(str(preview_png_path))
                logger.info("BatchExplorer: Fallback preview overlay PNG displayed.")

        # Check if we have a temporary viewer state to restore
        restore_state = getattr(self, "_temp_restore_viewer_state", None)
        if hasattr(self, "_temp_restore_viewer_state"):
            del self._temp_restore_viewer_state

        # Set Display
        if not pixmap.isNull():
            self.image_viewer.set_image(pixmap, restore_state=restore_state)
            if masks is not None:
                self.image_viewer.set_masks(masks)
                # Re-apply current controls state
                self.image_viewer.set_show_original(self.show_original_chk.isChecked())
                self.image_viewer.set_show_overlay(self.show_overlay_chk.isChecked())
                self.image_viewer.set_mask_opacity(self.opacity_slider.value())

    def _render_metadata_fields(self, record: dict):
        # Guard: meta_grid may be transiently None during _reset_explorer_state()
        if self.meta_grid is None:
            logger.warning("BatchExplorer: _render_metadata_fields called with meta_grid=None, skipping.")
            return
        grid = self.meta_grid
        theme = theme_service.current_theme
        
        lbl_style = "color: #6B7280; font-size: 9px; font-weight: bold; text-transform: uppercase;"
        if theme == "light":
            val_style = "color: #111827; font-size: 11px; font-weight: bold;"
            err_style = "color: #DC2626; font-size: 11px; font-weight: bold;"
        else:
            val_style = "color: #FFFFFF; font-size: 11px; font-weight: bold;"
            err_style = "color: #F87171; font-size: 11px; font-weight: bold;"

        is_edited = record.get("edited") in [True, "True", "true", 1, "1"]
        fields = [
            ("Filename", record["image_name"]),
            ("Edited", "Yes (✏ Modified)" if is_edited else "No"),
            ("Status", record.get("status", "SUCCESS")),
            ("Workflow", record.get("workflow", "Cell Counting")),
            ("Mode Preset", record.get("segmentation_mode", "Balanced")),
            ("Model Used", record.get("model_type", "cyto")),
            ("Resolved Backend", record.get("resolved_backend", "CPU")),
            ("Runtime", f"{record.get('processing_time_s', '0.0')} s"),
            ("Cell Count", record.get("cell_count", "0"))
        ]

        row = 0
        for label, val in fields:
            lbl_widget = QLabel(f"{label}:")
            lbl_widget.setStyleSheet(lbl_style)
            
            val_widget = QLabel(str(val))
            val_widget.setWordWrap(True)
            
            if label == "Status" and "FAILED" in str(val):
                val_widget.setStyleSheet(err_style)
            else:
                val_widget.setStyleSheet(val_style)
                
            grid.addWidget(lbl_widget, row, 0)
            grid.addWidget(val_widget, row, 1)
            row += 1

    def _clear_metadata_panel(self):
        # Guard: meta_grid may be transiently None during _reset_explorer_state()
        if self.meta_grid is None:
            return
        # Delete widgets and remove from layout immediately to prevent double addition overlaps
        while self.meta_grid.count() > 0:
            item = self.meta_grid.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

    def _on_prev_clicked(self):
        curr_row = self.navigator_list.currentRow()
        if curr_row > 0:
            self.navigator_list.setCurrentRow(curr_row - 1)

    def _on_next_clicked(self):
        curr_row = self.navigator_list.currentRow()
        if curr_row < self.navigator_list.count() - 1:
            self.navigator_list.setCurrentRow(curr_row + 1)

    def _on_opacity_slider_changed(self, value: int):
        self.opacity_val_lbl.setText(f"{value}%")
        self.image_viewer.set_mask_opacity(value)

    def _on_reset_view_clicked(self):
        if not self.image_viewer.pixmap_item.pixmap().isNull():
            self.image_viewer.resetTransform()
            self.image_viewer.fitInView(self.image_viewer.pixmap_item, Qt.KeepAspectRatio)
            self.image_viewer._zoom_touched = False
            self.image_viewer.update_viewer_cursor()
            logger.debug("BatchExplorer: View reset manually.")

    def _on_open_analysis_clicked(self):
        """Redirection Gateway loading selected image, mask overlay, and properties into AnalysisPage workspace."""
        curr_item = self.navigator_list.currentItem()
        if not curr_item:
            return
            
        record = curr_item.data(Qt.UserRole)
        image_name = record["image_name"]
        
        # 1. Resolve raw microscopy image path
        raw_image_path = self.batch_dir.parent / image_name
        if not raw_image_path.exists():
            # Recursive scan
            for root, dirs, files in os.walk(str(self.batch_dir.parent)):
                if "batch_results" in root:
                    continue
                if image_name in files:
                    raw_image_path = Path(root) / image_name
                    break
                    
        if not raw_image_path.exists():
            QMessageBox.critical(
                self,
                "Microscopy Image Not Found",
                f"Could not locate the raw microscopy image file '{image_name}' on local disk for analysis staging.",
                QMessageBox.Ok
            )
            return

        # 2. Reconstruct state.analysis_results (masks + metrics)
        labels_path = self.batch_dir / image_name / f"{image_name}_labels_raw.tif"
        masks = None
        if labels_path.exists():
            try:
                masks = tifffile.imread(str(labels_path))
            except Exception as e:
                logger.error("BatchExplorer: Failed to load raw masks for analysis redirection: %s", e)
                
        cell_metrics = {}
        csv_path = self.batch_dir / image_name / f"{image_name}_cell_metrics.csv"
        if csv_path.exists():
            try:
                with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 5:
                            cell_id = int(row[0])
                            area_px = int(row[1])
                            diam = float(row[2])
                            cx = float(row[3])
                            cy = float(row[4])
                            diam_est = float(row[5]) if len(row) > 5 else diam
                            cell_metrics[cell_id] = {
                                "area_px": area_px,
                                "diameter_px": diam,
                                "centroid": (cx, cy),
                                "diameter_estimate": diam_est
                            }
            except Exception as e:
                logger.error("BatchExplorer: Failed to parse cell CSV: %s", e)

        # Load edit operation log if exists
        edit_log = []
        edit_log_path = self.batch_dir / image_name / f"{image_name}_edit_log.json"
        if edit_log_path.exists():
            try:
                import json
                with open(edit_log_path, mode="r", encoding="utf-8") as f:
                    edit_log = json.load(f)
            except Exception as e:
                logger.error("BatchExplorer: Failed to load edit log: %s", e)

        # Reconstruct Results dictionary matching Cellpose output schema
        results = {
            "masks": masks,
            "cell_metrics": cell_metrics,
            "edit_operation_log": edit_log,
            "cell_count": int(record.get("cell_count") or 0),
            "average_diameter_px": float(record.get("average_diameter_px") or 0.0),
            "mean_cell_area_px": float(record.get("mean_area_px") or 0.0),
            "median_cell_area_px": float(record.get("median_area_px") or 0.0),
            "cell_density": float(record.get("cell_density") or 0.0),
            "processing_time_s": float(record.get("processing_time_s") or 0.0),
            "used_gpu": record.get("used_gpu") == "CUDA",
            "resolved_backend": record.get("resolved_backend", "CPU"),
            "model_type": record.get("model_type", "cyto"),
            "modality": "Fluorescence Microscopy"
        }

        # 3. Write properties to global state
        # Set path first (resets session)
        state.current_image_path = str(raw_image_path)
        
        # Start/get the analysis session and tag it immutably as batch-origin
        session = state.workspace_manager.start_analysis_session(
            str(raw_image_path),
            origin_type="batch",
            batch_origin_context=str(self.batch_dir)
        )
        
        # Set workflow and quality mode (resolve user-facing name back to ID if needed)
        wf_val = record.get("workflow", "cell_counting")
        from lumen.workflows.workflow_manager import workflow_manager
        if wf_val not in workflow_manager.workflows:
            for wf_id, wf in workflow_manager.workflows.items():
                if wf.name == wf_val:
                    wf_val = wf_id
                    break
        state.current_workflow = wf_val
        state.quality_mode = record.get("segmentation_mode", "Balanced")
        # Set analysis results last so they are preserved
        state.analysis_results = results
        
        # Populate session attributes so they restore correctly during sync_state
        session.current_workflow = state.current_workflow
        session.quality_mode = state.quality_mode
        session.segmentation_method = state.segmentation_method
        session.analysis_results = state.analysis_results
        session.committed_results = results
        session.mask_opacity = state.mask_opacity
        session.show_original_image = state.show_original_image
        session.show_segmentation_overlay = state.show_segmentation_overlay
        
        # Save batch session before redirecting
        self._save_to_session()
        
        # 4. Redirect to analysis page
        navigation_service.navigate_to("analysis")

    def _on_back_upload_clicked(self):
        navigation_service.navigate_to("upload")

    @Slot(str)
    def _sync_theme(self, theme_name: str = ""):
        theme = theme_name if theme_name else theme_service.current_theme
        self.image_viewer.sync_theme(theme)
        if hasattr(self, '_placeholder'):
            self._placeholder.sync_theme(theme)
        if hasattr(self, 'workspace_switcher'):
            self.workspace_switcher.sync_theme(theme)

        # Style components matching theme
        if theme == "light":
            self.left_panel.setStyleSheet("""
                #ExplorerLeftPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                }
            """)
            self.right_panel.setStyleSheet("""
                #ExplorerRightPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                }
            """)
            self.viewer_controls_bar.setStyleSheet("""
                #ViewerControlsBar {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
            """)
            self.show_original_chk.setStyleSheet("font-size: 11px; color: #1F2937;")
            self.show_overlay_chk.setStyleSheet("font-size: 11px; color: #1F2937;")
            self.opacity_val_lbl.setStyleSheet("font-size: 11px; color: #1F2937;")
            
            # Left labels
            left_lbls = self.left_panel.findChildren(QLabel)
            for lbl in left_lbls:
                if lbl.text() != "Batch Explorer":
                    lbl.setStyleSheet("color: #4B5563; font-size: 10px; font-weight: bold;")
            left_lbls[0].setStyleSheet("font-size: 15px; font-weight: bold; color: #111827;")

            # Right labels
            right_lbls = self.right_panel.findChildren(QLabel)
            if right_lbls:
                right_lbls[0].setStyleSheet("font-size: 15px; font-weight: bold; color: #111827;")

        else:
            self.left_panel.setStyleSheet("""
                #ExplorerLeftPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                }
            """)
            self.right_panel.setStyleSheet("""
                #ExplorerRightPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                }
            """)
            self.viewer_controls_bar.setStyleSheet("""
                #ViewerControlsBar {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
            """)
            self.show_original_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")
            self.show_overlay_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")
            self.opacity_val_lbl.setStyleSheet("font-size: 11px; color: #E5E7EB;")

            left_lbls = self.left_panel.findChildren(QLabel)
            for lbl in left_lbls:
                if lbl.text() != "Batch Explorer":
                    lbl.setStyleSheet("color: #9CA3AF; font-size: 10px; font-weight: bold;")
            left_lbls[0].setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")

            right_lbls = self.right_panel.findChildren(QLabel)
            if right_lbls:
                right_lbls[0].setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")

        # Refresh list row styling
        self._populate_list()
