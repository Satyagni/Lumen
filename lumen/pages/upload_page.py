import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QFileDialog, QSizePolicy, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPixmap, QCursor, QColor
from lumen.core.logger import logger
from lumen.core.constants import ALLOWED_EXTENSIONS
from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager, ImageManager
from lumen.core.services.navigation_service import navigation_service
from lumen.core.services.theme_service import theme_service
from lumen.processing.batch_manager import batch_manager

class RecommendationCard(QFrame):
    """Clickable recommendation card displaying workflow name, details, and selection checks."""
    
    clicked = Signal(str)

    def __init__(self, workflow_id: str, name: str, desc: str, relevance: str, parent=None):
        super().__init__(parent)
        self.workflow_id = workflow_id
        self.name = name
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.selected = False
        
        self._setup_ui(name, desc, relevance)
        self.update_appearance()

    def _setup_ui(self, name: str, desc: str, relevance: str):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 12, 14, 12)
        self.layout.setSpacing(4)

        # Header: Name, Relevance badge, and selection checkmark
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")
        
        # Relevance badge (e.g. High / Moderate / Low)
        self.badge_lbl = QLabel(f"Relevance: {relevance}")
        badge_style = "font-size: 9px; padding: 2px 6px; border-radius: 4px; font-weight: bold;"
        if relevance.lower() == "high":
            badge_style += "background-color: #065F46; color: #34D399;" # Green
        elif relevance.lower() == "moderate":
            badge_style += "background-color: #78350F; color: #F59E0B;" # Yellow/Amber
        else:
            badge_style += "background-color: #374151; color: #9CA3AF;" # Grey
        self.badge_lbl.setStyleSheet(badge_style)
        
        # Checked status indicator (hidden by default)
        self.check_lbl = QLabel("✓ Selected")
        self.check_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #818CF8;")
        self.check_lbl.setVisible(False)

        header_layout.addWidget(self.name_lbl)
        header_layout.addWidget(self.badge_lbl)
        header_layout.addStretch(1)
        header_layout.addWidget(self.check_lbl)
        self.layout.addLayout(header_layout)

        # Description
        self.desc_lbl = QLabel(desc)
        self.desc_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        self.desc_lbl.setWordWrap(True)
        self.layout.addWidget(self.desc_lbl)

    def set_selected(self, is_selected: bool):
        self.selected = is_selected
        self.check_lbl.setVisible(is_selected)
        self.update_appearance()

    def update_appearance(self):
        theme = state.current_theme
        if self.selected:
            if theme == "light":
                self.setStyleSheet("""
                    RecommendationCard {
                        background-color: #EEF2F6;
                        border: 2px solid #4F46E5;
                        border-radius: 6px;
                    }
                """)
                self.name_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #111827;")
                self.desc_lbl.setStyleSheet("font-size: 11px; color: #4B5563;")
                self.check_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #4F46E5;")
            else:
                self.setStyleSheet("""
                    RecommendationCard {
                        background-color: #242432;
                        border: 2px solid #6366F1;
                        border-radius: 6px;
                    }
                """)
                self.name_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")
                self.desc_lbl.setStyleSheet("font-size: 11px; color: #E5E7EB;")
                self.check_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #818CF8;")
        else:
            if theme == "light":
                self.setStyleSheet("""
                    RecommendationCard {
                        background-color: #FFFFFF;
                        border: 1px solid #D1D5DB;
                        border-radius: 6px;
                    }
                    RecommendationCard:hover {
                        background-color: #F9FAFB;
                        border-color: #9CA3AF;
                    }
                """)
                self.name_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #111827;")
                self.desc_lbl.setStyleSheet("font-size: 11px; color: #4B5563;")
            else:
                self.setStyleSheet("""
                    RecommendationCard {
                        background-color: #1C1C22;
                        border: 1px solid #2B2B35;
                        border-radius: 6px;
                    }
                    RecommendationCard:hover {
                        background-color: #25252F;
                        border-color: #3E3E4C;
                    }
                """)
                self.name_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")
                self.desc_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.workflow_id)
        super().mousePressEvent(event)


class DropZoneWidget(QFrame):
    """Clickable and drag-and-drop frame for loading local image files."""
    
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZoneFrame")
        self.setAcceptDrops(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setProperty("dragActive", "false")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 40, 32, 40)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel("📥")
        self.icon_label.setStyleSheet("font-size: 40px; color: #6366F1;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("Drag & drop biological image here, or click to browse")
        self.text_label.setObjectName("DropZoneLabel")
        self.text_label.setStyleSheet("font-weight: 500; font-size: 13px;")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.formats_label = QLabel(f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS).upper()}")
        self.formats_label.setStyleSheet("font-size: 10px; color: #6B7280;")
        self.formats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.formats_label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Filter only files
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if os.path.isfile(path):
                    event.acceptProposedAction()
                    self.setProperty("dragActive", "true")
                    self.refresh_style()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", "false")
        self.refresh_style()

    def dropEvent(self, event):
        self.setProperty("dragActive", "false")
        self.refresh_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self.file_dropped.emit(file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Biological Image",
                "",
                "Microscopy Images (*.png *.jpg *.jpeg *.tiff *.tif *.czi);;All Files (*)"
            )
            if file_path:
                self.file_dropped.emit(file_path)
        super().mousePressEvent(event)

    def refresh_style(self):
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class FolderDropZoneWidget(QFrame):
    """Clickable and drag-and-drop frame for picking directory paths."""
    
    folder_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FolderDropZoneFrame")
        self.setAcceptDrops(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setProperty("dragActive", "false")
        self._setup_ui()
        self.refresh_style()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 40, 32, 40)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel("📂")
        self.icon_label.setStyleSheet("font-size: 40px; color: #6366F1; background: transparent; border: none;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("Drag & drop microscopy folder here or click to browse")
        self.text_label.setObjectName("FolderDropZoneLabel")
        self.text_label.setStyleSheet("font-weight: 500; font-size: 13px; color: #9CA3AF; background: transparent; border: none;")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.secondary_label = QLabel("Supports microscopy image datasets")
        self.secondary_label.setStyleSheet("font-size: 11px; color: #9CA3AF; font-weight: 500; background: transparent; border: none;")
        self.secondary_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.secondary_label)

        self.helper_label = QLabel("TIFF/TIF/PNG/CZI supported")
        self.helper_label.setStyleSheet("font-size: 10px; color: #6B7280; background: transparent; border: none;")
        self.helper_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.helper_label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    self.setProperty("dragActive", "true")
                    self.refresh_style()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", "false")
        self.refresh_style()

    def dropEvent(self, event):
        self.setProperty("dragActive", "false")
        self.refresh_style()
        urls = event.mimeData().urls()
        if urls:
            dir_path = urls[0].toLocalFile()
            if os.path.isdir(dir_path):
                self.folder_dropped.emit(dir_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Microscopy Images Folder", "")
            if dir_path:
                self.folder_dropped.emit(dir_path)
        super().mousePressEvent(event)

    def refresh_style(self):
        theme = theme_service.current_theme
        active = self.property("dragActive") == "true"
        if theme == "light":
            bg = "#EEF2F6" if active else "#FFFFFF"
            border_color = "#818CF8" if active else "#D1D5DB"
            self.setStyleSheet(f"#FolderDropZoneFrame {{ border: 2px dashed {border_color}; border-radius: 12px; background-color: {bg}; }}")
            self.text_label.setStyleSheet("font-weight: 500; font-size: 13px; color: #4B5563; background: transparent; border: none;")
            self.secondary_label.setStyleSheet("font-size: 11px; color: #4B5563; font-weight: 500; background: transparent; border: none;")
            self.helper_label.setStyleSheet("font-size: 10px; color: #6B7280; background: transparent; border: none;")
        else:
            bg = "#1E1B4B" if active else "#141419"
            border_color = "#818CF8" if active else "#4F46E5"
            self.setStyleSheet(f"#FolderDropZoneFrame {{ border: 2px dashed {border_color}; border-radius: 12px; background-color: {bg}; }}")
            self.text_label.setStyleSheet("font-weight: 500; font-size: 13px; color: #9CA3AF; background: transparent; border: none;")
            self.secondary_label.setStyleSheet("font-size: 11px; color: #9CA3AF; font-weight: 500; background: transparent; border: none;")
            self.helper_label.setStyleSheet("font-size: 10px; color: #6B7280; background: transparent; border: none;")


class UploadPage(QWidget):
    """File selection, summary reporting, biological classification, and workflow recommendation view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rec_cards = []
        self.active_tab = "single"  # "single" or "batch"
        self.selected_batch_dir = ""
        self.local_image_manager = ImageManager()
        self.staged_workflow_id = None
        
        self._setup_ui()
        self._init_connections()
        self._sync_theme()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setObjectName("PageContainer")
        self.main_layout.setContentsMargins(24, 24, 24, 24)
        self.main_layout.setSpacing(16)

        # Page Header
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self.title = QLabel("Import Image")
        self.title.setObjectName("PageTitle")

        self.subtitle = QLabel("Load microscopy or sample photograph files (.png, .jpg, .tiff)")
        self.subtitle.setObjectName("PageSubtitle")

        header_layout.addWidget(self.title)
        header_layout.addWidget(self.subtitle)
        self.main_layout.addWidget(header_frame)

        # Divider line
        self.line_sep = QFrame()
        self.line_sep.setFrameShape(QFrame.HLine)
        self.line_sep.setFrameShadow(QFrame.Sunken)
        self.line_sep.setStyleSheet("background-color: #2B2B35; max-height: 1px; border: none;")
        self.main_layout.addWidget(self.line_sep)

        # Tabs bar (Segmented Control Mode Switcher)
        self.tabs_frame = QFrame()
        self.tabs_frame.setObjectName("TabsContainer")
        self.tabs_frame.setStyleSheet("""
            #TabsContainer {
                background-color: #16161A;
                border: 1px solid #2B2B35;
                border-radius: 6px;
                padding: 2px;
                max-height: 38px;
            }
        """)
        self.tabs_layout = QHBoxLayout(self.tabs_frame)
        self.tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_layout.setSpacing(4)

        self.tab_single_btn = QPushButton("Single Image Analysis")
        self.tab_single_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        self.tab_batch_btn = QPushButton("Batch Folder Analysis")
        self.tab_batch_btn.setCursor(QCursor(Qt.PointingHandCursor))

        self.tabs_layout.addWidget(self.tab_single_btn)
        self.tabs_layout.addWidget(self.tab_batch_btn)
        self.main_layout.addWidget(self.tabs_frame)

        # ====================================================================
        # 1. SINGLE IMAGE VIEW CONTAINER
        # ====================================================================
        self.single_view_container = QWidget()
        single_layout = QVBoxLayout(self.single_view_container)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(16)

        # Drag and Drop Input
        self.drop_zone = DropZoneWidget(self)
        single_layout.addWidget(self.drop_zone)

        # Uploaded Image Summary Container (Horizontal Split)
        self.summary_card = QFrame()
        self.summary_card.setObjectName("SummaryCard")
        summary_layout = QHBoxLayout(self.summary_card)
        summary_layout.setSpacing(20)

        # Large scaled thumbnail preview
        self.thumb_container = QFrame()
        self.thumb_container.setFixedWidth(140)
        thumb_vbox = QVBoxLayout(self.thumb_container)
        thumb_vbox.setContentsMargins(0, 0, 0, 0)
        thumb_vbox.setSpacing(6)
        thumb_vbox.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(140, 140)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        thumb_vbox.addWidget(self.thumb_label)

        self.contrast_lbl = QLabel("Auto Contrast Applied")
        self.contrast_lbl.setAlignment(Qt.AlignCenter)
        thumb_vbox.addWidget(self.contrast_lbl)
        summary_layout.addWidget(self.thumb_container)

        # Details Panel
        self.details_frame = QFrame()
        details_layout = QVBoxLayout(self.details_frame)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)

        details_title = QLabel("Uploaded Image Summary")
        details_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        details_layout.addWidget(details_title)

        # Labels Grid
        grid = QGridLayout()
        grid.setSpacing(8)

        fn_prefix = QLabel("Filename:")
        fn_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.fn_val = QLabel("-")
        self.fn_val.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")

        dim_prefix = QLabel("Dimensions:")
        dim_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.dim_val = QLabel("-")
        self.dim_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        ch_prefix = QLabel("Channels:")
        ch_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.ch_val = QLabel("-")
        self.ch_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        type_prefix = QLabel("Image Type:")
        type_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.type_val = QLabel("-")
        self.type_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        fmt_prefix = QLabel("Detected Format:")
        fmt_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.fmt_val = QLabel("-")
        self.fmt_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        grid.addWidget(fn_prefix, 0, 0)
        grid.addWidget(self.fn_val, 0, 1)
        grid.addWidget(dim_prefix, 1, 0)
        grid.addWidget(self.dim_val, 1, 1)
        grid.addWidget(ch_prefix, 2, 0)
        grid.addWidget(self.ch_val, 2, 1)
        grid.addWidget(type_prefix, 3, 0)
        grid.addWidget(self.type_val, 3, 1)
        grid.addWidget(fmt_prefix, 4, 0)
        grid.addWidget(self.fmt_val, 4, 1)
        details_layout.addLayout(grid)

        summary_layout.addWidget(self.details_frame, 1)
        single_layout.addWidget(self.summary_card)
        self.summary_card.setVisible(False)

        # Workflow Recommendations Panel
        self.recs_panel = QFrame()
        self.recs_panel.setObjectName("RecsPanel")
        recs_layout = QVBoxLayout(self.recs_panel)
        recs_layout.setSpacing(10)

        self.class_header = QLabel("Heuristic Modality Classification")
        self.class_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        recs_layout.addWidget(self.class_header)

        self.class_sub = QLabel("Select a recommended analysis pipeline to proceed:")
        self.class_sub.setStyleSheet("font-size: 11px; color: #9CA3AF; margin-bottom: 4px;")
        recs_layout.addWidget(self.class_sub)

        # Recommendations Vertical List
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setSpacing(8)
        recs_layout.addLayout(self.cards_layout)
        single_layout.addWidget(self.recs_panel)
        self.recs_panel.setVisible(False)

        # Single Actions
        self.single_actions_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Upload Different Image")
        self.reset_btn.setProperty("class", "SecondaryButton")
        self.reset_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.reset_btn.setVisible(False)
        self.single_actions_layout.addWidget(self.reset_btn)
        
        self.single_actions_layout.addStretch(1)

        self.proceed_btn = QPushButton("Proceed to Analysis")
        self.proceed_btn.setProperty("class", "PrimaryButton")
        self.proceed_btn.setCursor(QCursor(Qt.ArrowCursor))
        self.proceed_btn.setEnabled(False)
        self.single_actions_layout.addWidget(self.proceed_btn)
        single_layout.addLayout(self.single_actions_layout)

        self.main_layout.addWidget(self.single_view_container)

        # ====================================================================
        # 2. BATCH FOLDER VIEW CONTAINER
        # ====================================================================
        self.batch_view_container = QWidget()
        batch_layout = QVBoxLayout(self.batch_view_container)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(16)

        # Folder drag & drop Zone
        self.folder_drop_zone = FolderDropZoneWidget(self)
        batch_layout.addWidget(self.folder_drop_zone)

        # Center open existing batch results button
        self.open_existing_batch_btn = QPushButton("Open Existing Batch Results")
        self.open_existing_batch_btn.setObjectName("OpenExistingBatchButton")
        self.open_existing_batch_btn.setProperty("class", "SecondaryButton")
        self.open_existing_batch_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.open_existing_batch_btn.setFixedHeight(36)
        self.open_existing_batch_btn.setStyleSheet("margin-top: 8px;")
        batch_layout.addWidget(self.open_existing_batch_btn, 0, Qt.AlignCenter)

        # Scanned Batch Info Card
        self.batch_info_card = QFrame()
        self.batch_info_card.setObjectName("BatchInfoCard")
        self.batch_info_card.setStyleSheet("""
            #BatchInfoCard {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        batch_info_layout = QVBoxLayout(self.batch_info_card)
        batch_info_layout.setSpacing(12)

        self.batch_dir_lbl = QLabel("Folder: Not selected")
        self.batch_dir_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")
        batch_info_layout.addWidget(self.batch_dir_lbl)

        # Config Panel Grid
        config_grid = QGridLayout()
        config_grid.setSpacing(12)

        # Method Selector
        method_lbl = QLabel("Segmentation Method:")
        method_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF; font-weight: bold; text-transform: uppercase;")
        self.batch_method_combo = QComboBox()
        self.batch_method_combo.addItems(["AI Segmentation (Cellpose)"])
        self.batch_method_combo.setCurrentText("AI Segmentation (Cellpose)")
        self.batch_method_combo.setCursor(QCursor(Qt.PointingHandCursor))
        config_grid.addWidget(method_lbl, 0, 0)
        config_grid.addWidget(self.batch_method_combo, 0, 1)

        # Quality preset combo
        self.batch_quality_lbl = QLabel("Segmentation Quality preset:")
        self.batch_quality_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF; font-weight: bold; text-transform: uppercase;")
        self.batch_quality_combo = QComboBox()
        self.batch_quality_combo.addItems(["Balanced", "Fast", "Sensitive", "Precise"])
        self.batch_quality_combo.setCursor(QCursor(Qt.PointingHandCursor))
        config_grid.addWidget(self.batch_quality_lbl, 1, 0)
        config_grid.addWidget(self.batch_quality_combo, 1, 1)

        # Workflow combo
        w_lbl = QLabel("Analysis Workflow:")
        w_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF; font-weight: bold; text-transform: uppercase;")
        self.batch_workflow_combo = QComboBox()
        self.batch_workflow_combo.addItem("Cell Segmentation", "cell_counting")
        self.batch_workflow_combo.addItem("Fluorescence Analysis (Disabled - Single Image Only)", "fluorescence")
        self.batch_workflow_combo.setCursor(QCursor(Qt.PointingHandCursor))
        # Disable the Fluorescence Analysis item
        model = self.batch_workflow_combo.model()
        if model:
            item = model.item(1)
            if item:
                item.setEnabled(False)
        config_grid.addWidget(w_lbl, 2, 0)
        config_grid.addWidget(self.batch_workflow_combo, 2, 1)

        # Execution Backend combo
        self.batch_backend_lbl = QLabel("Execution Backend:")
        self.batch_backend_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF; font-weight: bold; text-transform: uppercase;")
        self.batch_backend_combo = QComboBox()
        self.batch_backend_combo.addItems(["Use Global Setting", "Auto", "CUDA (GPU)", "CPU"])
        self.batch_backend_combo.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Disable CUDA option if not available on machine
        from lumen.core.services.gpu_service import gpu_service
        if not gpu_service.is_cuda_available:
            self.batch_backend_combo.setItemText(2, "CUDA (Unavailable)")
            self.batch_backend_combo.model().item(2).setEnabled(False)
            
        config_grid.addWidget(self.batch_backend_lbl, 3, 0)
        config_grid.addWidget(self.batch_backend_combo, 3, 1)

        # Recursive search checkbox
        self.batch_recursive_chk = QCheckBox("Scan nested folders recursively")
        self.batch_recursive_chk.setCursor(QCursor(Qt.PointingHandCursor))
        self.batch_recursive_chk.setStyleSheet("font-size: 12px; color: #E5E7EB;")
        config_grid.addWidget(self.batch_recursive_chk, 4, 0, 1, 2)

        batch_info_layout.addLayout(config_grid)
        batch_layout.addWidget(self.batch_info_card)
        self.batch_info_card.setVisible(False)

        # BATCH ESTIMATE UI CARD
        self.batch_estimate_card = QFrame()
        self.batch_estimate_card.setObjectName("BatchEstimateCard")
        self.batch_estimate_card.setStyleSheet("""
            #BatchEstimateCard {
                background-color: #131317;
                border: 1px solid #312E81;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        estimate_vbox = QVBoxLayout(self.batch_estimate_card)
        estimate_vbox.setSpacing(6)

        est_header = QLabel("Batch Execution Summary")
        est_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #818CF8; text-transform: uppercase; letter-spacing: 0.5px;")
        estimate_vbox.addWidget(est_header)

        # Label displaying quick summary (Found, Estimated runtime, Backend)
        self.estimate_val = QLabel("-")
        self.estimate_val.setObjectName("estimateui")
        self.estimate_val.setStyleSheet("font-size: 13px; color: #F3F4F6; line-height: 1.5; font-family: sans-serif;")
        self.estimate_val.setWordWrap(True)
        estimate_vbox.addWidget(self.estimate_val)

        batch_layout.addWidget(self.batch_estimate_card)
        self.batch_estimate_card.setVisible(False)

        # Batch Action Buttons
        self.batch_actions_layout = QHBoxLayout()
        self.clear_batch_btn = QPushButton("Select Different Folder")
        self.clear_batch_btn.setProperty("class", "SecondaryButton")
        self.clear_batch_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_batch_btn.setVisible(False)
        self.batch_actions_layout.addWidget(self.clear_batch_btn)

        self.batch_actions_layout.addStretch(1)

        self.run_batch_btn = QPushButton("Run Batch Analysis")
        self.run_batch_btn.setProperty("class", "PrimaryButton")
        self.run_batch_btn.setCursor(QCursor(Qt.ArrowCursor))
        self.run_batch_btn.setEnabled(False)
        self.batch_actions_layout.addWidget(self.run_batch_btn)
        batch_layout.addLayout(self.batch_actions_layout)

        self.main_layout.addWidget(self.batch_view_container)
        self.batch_view_container.setVisible(False)

        # Spacing fill
        self.main_layout.addStretch(1)

    def _init_connections(self):
        # Navigation tabs connection
        self.tab_single_btn.clicked.connect(lambda: self._set_mode("single"))
        self.tab_batch_btn.clicked.connect(lambda: self._set_mode("batch"))

        # Single connections
        self.drop_zone.file_dropped.connect(self._on_file_selected)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        self.proceed_btn.clicked.connect(self._on_proceed_clicked)
        
        state.theme_changed.connect(self._sync_theme)

        # Batch connections
        self.folder_drop_zone.folder_dropped.connect(self._on_folder_selected)
        self.open_existing_batch_btn.clicked.connect(self._on_open_existing_batch_clicked)
        self.clear_batch_btn.clicked.connect(self._on_clear_batch_clicked)
        self.run_batch_btn.clicked.connect(self._on_run_batch_clicked)
        
        self.batch_method_combo.currentIndexChanged.connect(self._on_batch_method_changed)
        self.batch_quality_combo.currentIndexChanged.connect(self._update_batch_estimation)
        self.batch_workflow_combo.currentIndexChanged.connect(self._on_batch_workflow_changed)
        self.batch_backend_combo.currentIndexChanged.connect(self._update_batch_estimation)
        self.batch_recursive_chk.stateChanged.connect(self._update_batch_estimation)
        state.backend_preference_changed.connect(self._on_global_backend_changed)

    def _set_mode(self, mode: str):
        if mode == "batch":
            sess = state.workspace_manager.get_analysis_session(state.current_image_path) if state.current_image_path else None
            logger.warning(
                "TIMELINE [3. Entering Batch Upload]: state.current_workflow=%s, session.current_workflow=%s",
                state.current_workflow,
                sess.current_workflow if sess else None
            )
        self.active_tab = mode
        
        # Style active tab indicator
        theme = theme_service.current_theme
        if mode == "single":
            self.single_view_container.setVisible(True)
            self.batch_view_container.setVisible(False)
            
            # Styles for dark/light mode active/inactive state
            if theme == "light":
                self.tab_single_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: transparent; color: #4B5563; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
            else:
                self.tab_single_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: transparent; color: #9CA3AF; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
        else:
            self.single_view_container.setVisible(False)
            self.batch_view_container.setVisible(True)
            
            if theme == "light":
                self.tab_single_btn.setStyleSheet("background-color: transparent; color: #4B5563; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")
            else:
                self.tab_single_btn.setStyleSheet("background-color: transparent; color: #9CA3AF; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")

        self._sync_theme()

    def _on_file_selected(self, file_path: str):
        success, msg = self.local_image_manager.load_image(file_path, set_state=False)
        if success:
            self.staged_image_path = file_path
            self._update_ui_with_image()
        else:
            logger.error("UploadPage: Load failed: %s", msg)
            QMessageBox.critical(self, "Load Failed", f"Failed to load image file:\n{msg}")

    def _update_ui_with_image(self):
        meta = self.local_image_manager.get_metadata()
        if not meta:
            return

        self.fn_val.setText(meta["filename"])
        self.dim_val.setText(f"{meta['width']} × {meta['height']}")
        self.ch_val.setText(str(meta["channels"]))
        self.type_val.setText(meta["format"])
        self.fmt_val.setText(meta["mode"].upper())

        pixmap = self.local_image_manager.get_thumbnail(140, 140)
        if pixmap:
            self.thumb_label.setPixmap(pixmap)
            self.thumb_label.setText("")
        else:
            self.thumb_label.setText("No Preview")

        class_str = meta["classification"]
        conf_str = meta["confidence"]
        self.class_header.setText(f"Likely Type: {class_str}  (Confidence: {conf_str})")

        self.drop_zone.setVisible(False)
        self.summary_card.setVisible(True)
        self.recs_panel.setVisible(True)
        self.reset_btn.setVisible(True)

        recs = list(meta["recommended_workflows"])
        # If state.current_workflow is set (e.g. intentionally from Home card),
        # make sure it is present in the recommendation cards so it can be selected.
        if state.current_workflow and not any(wf["id"] == state.current_workflow for wf in recs):
            if state.current_workflow == "fluorescence":
                recs.append({
                    "id": "fluorescence",
                    "name": "Fluorescence Analysis",
                    "desc": "Quantify signal intensity profiles across color channels.",
                    "relevance": "High"
                })
            elif state.current_workflow == "cell_counting":
                recs.append({
                    "id": "cell_counting",
                    "name": "Cell Segmentation",
                    "desc": "Detect and segment cells or nuclei.",
                    "relevance": "High"
                })

        self._build_recommendation_cards(recs)
        
        default_wf = recs[0]["id"] if recs else None
        if recs and state.current_workflow in [wf["id"] for wf in recs]:
            default_wf = state.current_workflow
            
        if default_wf:
            self._on_card_selected(default_wf)

    def _build_recommendation_cards(self, workflows: list):
        for card in self.rec_cards:
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self.rec_cards.clear()

        for wf in workflows:
            card = RecommendationCard(wf["id"], wf["name"], wf["desc"], wf["relevance"], self)
            card.clicked.connect(self._on_card_selected)
            self.cards_layout.addWidget(card)
            self.rec_cards.append(card)

    def _on_card_selected(self, workflow_id: str):
        for card in self.rec_cards:
            card.set_selected(card.workflow_id == workflow_id)
        
        self.staged_workflow_id = workflow_id
        
        self.proceed_btn.setEnabled(True)
        self.proceed_btn.setCursor(QCursor(Qt.PointingHandCursor))

    def _on_reset_clicked(self):
        self.staged_image_path = None
        self.staged_workflow_id = None
        self.local_image_manager.clear_cache()
        self._show_input_state()

    def _show_input_state(self):
        self.drop_zone.setVisible(True)
        self.summary_card.setVisible(False)
        self.recs_panel.setVisible(False)
        self.reset_btn.setVisible(False)
        
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.setCursor(QCursor(Qt.ArrowCursor))

        for card in self.rec_cards:
            card.deleteLater()
        self.rec_cards.clear()

    # ====================================================================
    # BATCH EVENT HANDLERS
    # ====================================================================
    def _on_folder_selected(self, dir_path: str):
        if not dir_path or not os.path.exists(dir_path):
            return

        self.selected_batch_dir = dir_path
        self.folder_drop_zone.setVisible(False)
        self.open_existing_batch_btn.setVisible(False)
        self.batch_info_card.setVisible(True)
        self.clear_batch_btn.setVisible(True)
        
        self._update_batch_estimation()

    def _on_clear_batch_clicked(self):
        self.selected_batch_dir = ""
        self.folder_drop_zone.setVisible(True)
        self.open_existing_batch_btn.setVisible(True)
        self.batch_info_card.setVisible(False)
        self.batch_estimate_card.setVisible(False)
        self.clear_batch_btn.setVisible(False)
        self.run_batch_btn.setEnabled(False)
        self.run_batch_btn.setCursor(QCursor(Qt.ArrowCursor))

    def _on_open_existing_batch_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Open Completed Batch Results Folder", "")
        if dir_path:
            path = Path(dir_path)
            results_dir = path
            if not (path / "batch_summary.csv").exists() and not (path / "run_manifest.json").exists():
                if (path / "batch_results").exists():
                    results_dir = path / "batch_results"
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Batch Results",
                        "The selected folder does not contain completed batch results (missing batch_summary.csv or run_manifest.json).",
                        QMessageBox.Ok
                    )
                    return
            
            # Navigate to explorer page
            state.batch_results_dir = str(results_dir)
            navigation_service.navigate_to("batch_explorer")

    def _on_batch_method_changed(self):
        selected = self.batch_method_combo.currentText()
        is_ai = selected == "AI Segmentation (Cellpose)"

        # AI mode: show quality+backend; other extensible backends can hide or show their own panels
        self.batch_quality_lbl.setVisible(is_ai)
        self.batch_quality_combo.setVisible(is_ai)
        self.batch_backend_lbl.setVisible(is_ai)
        self.batch_backend_combo.setVisible(is_ai)

        self._update_batch_estimation()

    def _on_batch_workflow_changed(self, index: int):
        self._update_batch_estimation()

    def _update_batch_estimation(self):
        if not self.selected_batch_dir:
            self.batch_estimate_card.setVisible(False)
            self.run_batch_btn.setEnabled(False)
            return

        selected_workflow = self.batch_workflow_combo.currentData()
        if selected_workflow == "fluorescence":
            self.batch_dir_lbl.setText(f"Folder: {self.selected_batch_dir}")
            self.batch_estimate_card.setVisible(True)
            self.estimate_val.setText("Fluorescence analysis batch pipeline under development.")
            self.run_batch_btn.setEnabled(False)
            self.run_batch_btn.setCursor(QCursor(Qt.ArrowCursor))
            return

        # Prepare parameters
        selected_method = self.batch_method_combo.currentText()
        if selected_method == "AI Segmentation (Cellpose)":
            segmentation_method = "AI Segmentation"
        else:
            segmentation_method = selected_method
        
        backend_pref = self.batch_backend_combo.currentText()
        if "CUDA" in backend_pref:
            backend_pref = "CUDA (GPU)"

        params = {
            "segmentation_method": segmentation_method,
            "quality_mode": self.batch_quality_combo.currentText(),
            "workflow": self.batch_workflow_combo.currentData() or "cell_counting",
            "backend_preference": backend_pref
        }
        recursive = self.batch_recursive_chk.isChecked()
        
        num_images = batch_manager.prepare_batch(self.selected_batch_dir, params, recursive)
        
        if num_images == 0:
            self.batch_dir_lbl.setText(f"Folder: {self.selected_batch_dir} (No microscopy images found!)")
            self.batch_estimate_card.setVisible(False)
            self.run_batch_btn.setEnabled(False)
            self.run_batch_btn.setCursor(QCursor(Qt.ArrowCursor))
            return

        self.batch_dir_lbl.setText(f"Folder: {self.selected_batch_dir}")
        self.batch_estimate_card.setVisible(True)
        self.run_batch_btn.setEnabled(True)
        self.run_batch_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        est_min = batch_manager.get_estimated_runtime_minutes()
        
        from lumen.core.services.gpu_service import gpu_service
        use_gpu, resolved_name = gpu_service.resolve_execution_backend(backend_pref)
        if segmentation_method != "AI Segmentation":
            resolved_name = "CPU (Alternative)"
            backend_pref = "CPU"
            
        summary_text = (
            f"Found: {num_images} microscopy images\n\n"
            f"Estimated runtime:\n"
            f"~{est_min} minutes\n\n"
            f"Method: {segmentation_method}\n"
            f"Resolved Backend: {resolved_name}"
        )
        self.estimate_val.setText(summary_text)

    @Slot(str)
    def _on_global_backend_changed(self, preference: str):
        if self.batch_backend_combo.currentText() == "Use Global Setting":
            self._update_batch_estimation()

    def _on_run_batch_clicked(self):
        if not self.selected_batch_dir:
            return

        # Setup state variables
        state.current_workflow = self.batch_workflow_combo.currentData() or "cell_counting"
        
        selected_method = self.batch_method_combo.currentText()
        if selected_method == "AI Segmentation (Cellpose)":
            state.segmentation_method = "AI Segmentation"
        else:
            state.segmentation_method = selected_method
        
        # Switch navigation to progress screen
        navigation_service.navigate_to("batch_progress")
        
        # Kick off run
        batch_manager.start_batch()

    # ====================================================================
    # GENERAL HANDLERS
    # ====================================================================
    @Slot(str)
    def _on_state_image_loaded(self, path: str):
        if path:
            self.staged_image_path = path
            self._update_ui_with_image()
        else:
            self.staged_image_path = None
            self._show_input_state()

    @Slot(str)
    def _on_state_workflow_selected(self, workflow_id: str):
        for card in self.rec_cards:
            card.set_selected(card.workflow_id == workflow_id)

    @Slot(str)
    def _sync_theme(self, theme_name: str = ""):
        theme = theme_name if theme_name else theme_service.current_theme
        
        # Refresh custom drop zones style
        self.drop_zone.refresh_style()
        self.folder_drop_zone.refresh_style()
        
        # Styled theme properties
        if theme == "light":
            self.summary_card.setStyleSheet("""
                #SummaryCard {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.recs_panel.setStyleSheet("""
                #RecsPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            
            # Batch Frame style
            self.batch_info_card.setStyleSheet("""
                #BatchInfoCard {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.batch_estimate_card.setStyleSheet("""
                #BatchEstimateCard {
                    background-color: #F3F4F6;
                    border: 1px solid #C7D2FE;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            
            self.thumb_label.setStyleSheet("background-color: #F9FAFB; border: 1px solid #D1D5DB; border-radius: 6px;")
            self.contrast_lbl.setStyleSheet("font-size: 10px; color: #4F46E5; font-weight: bold; background: transparent;")
            self.details_frame.setStyleSheet("QLabel { color: #1F2937; }")
            self.fn_val.setStyleSheet("color: #111827; font-weight: bold; font-size: 12px;")
            self.dim_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.ch_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.type_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.fmt_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.class_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #111827;")
            self.class_sub.setStyleSheet("font-size: 11px; color: #4B5563;")
            
            self.batch_dir_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #111827;")
            self.estimate_val.setStyleSheet("font-size: 13px; color: #1F2937; line-height: 1.5; font-family: sans-serif;")
            
            # Segmented controls style light
            self.tabs_frame.setStyleSheet("""
                #TabsContainer {
                    background-color: #E5E7EB;
                    border: 1px solid #D1D5DB;
                    border-radius: 6px;
                    padding: 2px;
                    max-height: 38px;
                }
            """)
            
        else:
            self.summary_card.setStyleSheet("""
                #SummaryCard {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.recs_panel.setStyleSheet("""
                #RecsPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            
            self.batch_info_card.setStyleSheet("""
                #BatchInfoCard {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.batch_estimate_card.setStyleSheet("""
                #BatchEstimateCard {
                    background-color: #131317;
                    border: 1px solid #312E81;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            
            self.thumb_label.setStyleSheet("background-color: #0B0B0D; border: 1px solid #2B2B35; border-radius: 6px;")
            self.contrast_lbl.setStyleSheet("font-size: 10px; color: #818CF8; font-weight: bold; background: transparent;")
            self.details_frame.setStyleSheet("QLabel { color: #F3F4F6; }")
            self.fn_val.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 12px;")
            self.dim_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.ch_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.type_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.fmt_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.class_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
            self.class_sub.setStyleSheet("font-size: 11px; color: #9CA3AF;")
            
            self.batch_dir_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")
            self.estimate_val.setStyleSheet("font-size: 13px; color: #F3F4F6; line-height: 1.5; font-family: sans-serif;")
            
            self.tabs_frame.setStyleSheet("""
                #TabsContainer {
                    background-color: #16161A;
                    border: 1px solid #2B2B35;
                    border-radius: 6px;
                    padding: 2px;
                    max-height: 38px;
                }
            """)

        # Re-apply styling for active vs inactive tab buttons
        if self.active_tab == "single":
            if theme == "light":
                self.tab_single_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: transparent; color: #4B5563; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
            else:
                self.tab_single_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: transparent; color: #9CA3AF; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
        else:
            if theme == "light":
                self.tab_single_btn.setStyleSheet("background-color: transparent; color: #4B5563; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")
            else:
                self.tab_single_btn.setStyleSheet("background-color: transparent; color: #9CA3AF; border-radius: 4px; padding: 6px 16px; font-weight: 500; border: none;")
                self.tab_batch_btn.setStyleSheet("background-color: #4F46E5; color: #FFFFFF; border-radius: 4px; padding: 6px 16px; font-weight: 600; border: none;")

        for card in self.rec_cards:
            card.update_appearance()

    def _on_proceed_clicked(self):
        if hasattr(self, "staged_image_path") and self.staged_image_path:
            state.current_image_path = self.staged_image_path
            
            # Start analysis session
            state.workspace_manager.start_analysis_session(self.staged_image_path, origin_type="single")
            
            # Save to recents history
            from lumen.core.config import config
            config.add_recent_file(self.staged_image_path, self.staged_workflow_id)
            
        if hasattr(self, "staged_workflow_id") and self.staged_workflow_id:
            state.current_workflow = self.staged_workflow_id
            
        navigation_service.navigate_to("analysis")

    def showEvent(self, event):
        sess = state.workspace_manager.get_analysis_session(state.current_image_path) if state.current_image_path else None
        logger.warning(
            "TIMELINE [2. Entering Upload page]: state.current_workflow=%s, session.current_workflow=%s",
            state.current_workflow,
            sess.current_workflow if sess else None
        )
        super().showEvent(event)
        if state.current_workflow:
            idx = self.batch_workflow_combo.findData(state.current_workflow)
            if idx == 0:
                self.batch_workflow_combo.setCurrentIndex(idx)
            else:
                self.batch_workflow_combo.setCurrentIndex(0)
