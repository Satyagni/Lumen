from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QComboBox, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCursor
from lumen.core.constants import APP_VERSION
from lumen.core.services.theme_service import theme_service
from lumen.core.services.gpu_service import gpu_service
from lumen.workflows.state import state
from lumen.core.logger import logger

class SettingsPage(QWidget):
    """System configuration parameters and diagnostics details view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._init_connections()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setObjectName("PageContainer")
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # Page Header
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        title = QLabel("System Settings")
        title.setObjectName("PageTitle")

        subtitle = QLabel("Configure user preferences and view active hardware diagnostics")
        subtitle.setObjectName("PageSubtitle")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header_frame)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #2B2B35; max-height: 1px; border: none;")
        main_layout.addWidget(line)

        # 1. Theme Configuration Card
        theme_frame = QFrame()
        theme_frame.setObjectName("SettingCard")
        theme_frame.setStyleSheet("""
            #SettingCard {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        theme_layout = QVBoxLayout(theme_frame)
        theme_layout.setSpacing(8)

        theme_title = QLabel("Appearance Preferences")
        theme_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        theme_layout.addWidget(theme_title)

        theme_desc = QLabel("Switch between dark mode (microscopy-optimized) and standard light mode interfaces.")
        theme_desc.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        theme_layout.addWidget(theme_desc)

        # Selector Row
        row_layout = QHBoxLayout()
        row_layout.setSpacing(12)
        
        row_label = QLabel("Active Color Scheme:")
        row_label.setStyleSheet("font-size: 12px; color: #E5E7EB;")
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Mode", "Light Mode"])
        self.theme_combo.setCursor(QCursor(Qt.PointingHandCursor))
        # Set current selection
        current_idx = 0 if theme_service.current_theme == "dark" else 1
        self.theme_combo.setCurrentIndex(current_idx)
        
        row_layout.addWidget(row_label)
        row_layout.addWidget(self.theme_combo)
        row_layout.addStretch(1)
        theme_layout.addLayout(row_layout)
        main_layout.addWidget(theme_frame)

        # 2. Hardware Diagnostics Card
        hw_frame = QFrame()
        hw_frame.setObjectName("SettingCard")
        hw_frame.setStyleSheet("""
            #SettingCard {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        hw_layout = QVBoxLayout(hw_frame)
        hw_layout.setSpacing(10)

        hw_title = QLabel("Hardware Diagnostics")
        hw_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        hw_layout.addWidget(hw_title)

        # Status displays
        backend_lbl = QLabel(f"Active Processing Backend:  {gpu_service.backend}")
        backend_lbl.setStyleSheet("font-size: 12px; color: #E5E7EB;")
        hw_layout.addWidget(backend_lbl)

        # Detailed state label
        self.cuda_avail_lbl = QLabel()
        self.cuda_avail_lbl.setStyleSheet("font-size: 12px;")
        if gpu_service.is_cuda_available:
            self.cuda_avail_lbl.setText("Hardware Backend: CUDA GPU Acceleration Active")
            self.cuda_avail_lbl.setStyleSheet("color: #34D399; font-weight: 500;") # Green text
        else:
            self.cuda_avail_lbl.setText("Hardware Backend: CPU Mode (PyTorch GPU libraries not found or inactive)")
            self.cuda_avail_lbl.setStyleSheet("color: #F87171;") # Red/Grey text
        hw_layout.addWidget(self.cuda_avail_lbl)
        
        main_layout.addWidget(hw_frame)

        # 3. Information Card
        info_frame = QFrame()
        info_frame.setObjectName("SettingCard")
        info_frame.setStyleSheet("""
            #SettingCard {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(6)

        info_title = QLabel("System Information")
        info_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        info_layout.addWidget(info_title)

        version_lbl = QLabel(f"Lumen Shell Desktop Version: {APP_VERSION}")
        version_lbl.setStyleSheet("font-size: 12px; color: #9CA3AF;")
        
        platform_lbl = QLabel("Desktop UI Framework: PySide6 (Qt for Python)")
        platform_lbl.setStyleSheet("font-size: 12px; color: #9CA3AF;")

        info_layout.addWidget(version_lbl)
        info_layout.addWidget(platform_lbl)
        main_layout.addWidget(info_frame)

        # Spacing fill
        main_layout.addStretch(1)

    def _init_connections(self):
        # Sync changes on selection change
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selection_changed)
        # Sync state changes (in case theme toggled from navbar)
        state.theme_changed.connect(self._on_state_theme_changed)

    def _on_theme_selection_changed(self, index: int):
        target_theme = "dark" if index == 0 else "light"
        if theme_service.current_theme != target_theme:
            theme_service.apply_theme(target_theme)

    @Slot(str)
    def _on_state_theme_changed(self, theme_name: str):
        target_idx = 0 if theme_name == "dark" else 1
        # Block signals temporarily to prevent loop trigger
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(target_idx)
        self.theme_combo.blockSignals(False)
        logger.debug("Settings: Synced theme selection box.")
