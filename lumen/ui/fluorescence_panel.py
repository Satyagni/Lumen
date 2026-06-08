from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QFrame, QGridLayout
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCursor
from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager
from lumen.core.logger import logger

class FluorescencePanel(QWidget):
    """Surgical interface overlay showing channel controls mounted inside the banner frame placeholder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FluorescencePanel")
        self._setup_ui()
        self._init_connections()
        self._sync_channels()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Title/Header info
        header = QLabel("Fluorescence Workspace Controls")
        header.setStyleSheet("font-weight: bold; font-size: 12px; color: #5BE7FF;")
        layout.addWidget(header)

        # Grids of selectors
        grid = QGridLayout()
        grid.setSpacing(8)

        # 1. Viewer Channel dropdown
        viewer_lbl = QLabel("Display View:")
        viewer_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0; font-weight: 500;")
        self.viewer_combo = QComboBox()
        self.viewer_combo.setCursor(QCursor(Qt.PointingHandCursor))
        grid.addWidget(viewer_lbl, 0, 0)
        grid.addWidget(self.viewer_combo, 0, 1)

        # 2. Segmentation Channel dropdown
        seg_lbl = QLabel("Segmentation Target:")
        seg_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0; font-weight: 500;")
        self.seg_combo = QComboBox()
        self.seg_combo.setCursor(QCursor(Qt.PointingHandCursor))
        grid.addWidget(seg_lbl, 1, 0)
        grid.addWidget(self.seg_combo, 1, 1)

        layout.addLayout(grid)

        # Divider line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: rgba(255, 255, 255, 0.06); max-height: 1px; border: none;")
        layout.addWidget(line)

        # 3. Channel Mapping fields
        self.mapping_title = QLabel("Channel Name Customization")
        self.mapping_title.setStyleSheet("font-size: 10px; font-weight: 600; color: #9EA4B0; text-transform: uppercase; letter-spacing: 0.5px;")
        layout.addWidget(self.mapping_title)

        self.mapping_container = QWidget()
        self.mapping_layout = QVBoxLayout(self.mapping_container)
        self.mapping_layout.setContentsMargins(0, 0, 0, 0)
        self.mapping_layout.setSpacing(6)
        layout.addWidget(self.mapping_container)

    def _init_connections(self):
        self.viewer_combo.currentIndexChanged.connect(self._on_viewer_channel_changed)
        self.seg_combo.currentIndexChanged.connect(self._on_seg_channel_changed)
        state.channel_names_changed.connect(self._on_state_channel_names_changed)
        state.image_loaded.connect(self._on_image_loaded)

    def _sync_channels(self):
        """Loads channels from image metadata and populates dropdown lists."""
        metadata = image_manager.get_metadata()
        channels_count = metadata.get("channels", 1)

        # Block signals temporarily to prevent loop trigger
        self.viewer_combo.blockSignals(True)
        self.seg_combo.blockSignals(True)

        self.viewer_combo.clear()
        self.seg_combo.clear()

        # Clear naming inputs
        while self.mapping_layout.count() > 0:
            item = self.mapping_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get naming mapping
        names = state.channel_names
        if not names or len(names) != channels_count:
            # Re-fetch default list if mismatch
            from lumen.core.fluorescence.channels import get_default_channel_names
            names = get_default_channel_names(channels_count, metadata.get("filename", ""))
            state.channel_names = names

        if channels_count > 1:
            self.viewer_combo.addItem("Composite (All Merged)", -1)
            for idx, name in enumerate(names):
                self.viewer_combo.addItem(f"Channel {idx}: {name}", idx)
                self.seg_combo.addItem(f"Channel {idx}: {name}", idx)
                
                # Add text input fields dynamically to edit naming mapping
                row = QHBoxLayout()
                row.setSpacing(6)
                num_lbl = QLabel(f"Ch {idx}:")
                num_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0; min-width: 35px;")
                
                edit = QLineEdit(name)
                edit.setProperty("channel_idx", idx)
                edit.setStyleSheet("font-size: 11px; padding: 4px 8px;")
                edit.textChanged.connect(self._on_channel_renamed)
                
                row.addWidget(num_lbl)
                row.addWidget(edit)
                row_widget = QWidget()
                row_widget.setLayout(row)
                self.mapping_layout.addWidget(row_widget)
            
            self.mapping_title.show()
            self.mapping_container.show()
        else:
            self.viewer_combo.addItem("Channel 0: Grayscale", 0)
            self.seg_combo.addItem("Channel 0: Grayscale", 0)
            self.mapping_title.hide()
            self.mapping_container.hide()

        # Sync dropdown selected indexes
        # Set Active Viewer Channel selection index
        viewer_idx = self.viewer_combo.findData(state.active_viewer_channel)
        if viewer_idx >= 0:
            self.viewer_combo.setCurrentIndex(viewer_idx)
        else:
            self.viewer_combo.setCurrentIndex(0)

        # Set Target Segmentation Channel selection index
        seg_idx = self.seg_combo.findData(state.segmentation_channel)
        if seg_idx >= 0:
            self.seg_combo.setCurrentIndex(seg_idx)
        else:
            self.seg_combo.setCurrentIndex(0)

        self.viewer_combo.blockSignals(False)
        self.seg_combo.blockSignals(False)

    @Slot(int)
    def _on_viewer_channel_changed(self, combo_idx: int):
        val = self.viewer_combo.currentData()
        if val is not None:
            logger.info("FluorescencePanel: Switching visible display channel selection to index %s", val)
            state.active_viewer_channel = val
            image_manager.set_active_channel(val)
            
            # Broadcast image reload to repaint GraphicsView
            state.image_loaded.emit(state.current_image_path)

    @Slot(int)
    def _on_seg_channel_changed(self, combo_idx: int):
        val = self.seg_combo.currentData()
        if val is not None:
            logger.info("FluorescencePanel: Setting target segmentation channel selection to index %s", val)
            state.segmentation_channel = val

    @Slot(str)
    def _on_channel_renamed(self, text: str):
        sender = self.sender()
        if not sender:
            return
        idx = sender.property("channel_idx")
        if idx is not None:
            current_names = list(state.channel_names)
            if idx < len(current_names):
                current_names[idx] = text
                # Set silently to block double updates
                state.blockSignals(True)
                state.channel_names = current_names
                state.blockSignals(False)

    @Slot(list)
    def _on_state_channel_names_changed(self, names: list):
        # Trigger redraw of dropdown options to match
        self._sync_channels()

    @Slot(str)
    def _on_image_loaded(self, path: str):
        # Regenerate list items if new image loaded
        # Check sender to prevent reload loop on active display switch
        if self.sender() is not state:
            self._sync_channels()
