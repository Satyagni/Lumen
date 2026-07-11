from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
)
from PySide6.QtCore import Qt, Slot
from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager
from lumen.core.logger import logger

class FluorescencePanel(QWidget):
    """Dynamic interface overlay showing channel naming text fields."""

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

        # Channel Mapping fields
        self.mapping_title = QLabel("Channel Name Customization")
        self.mapping_title.setStyleSheet("font-size: 10px; font-weight: 600; color: #9EA4B0; text-transform: uppercase; letter-spacing: 0.5px;")
        layout.addWidget(self.mapping_title)

        self.mapping_container = QWidget()
        self.mapping_layout = QVBoxLayout(self.mapping_container)
        self.mapping_layout.setContentsMargins(0, 0, 0, 0)
        self.mapping_layout.setSpacing(6)
        layout.addWidget(self.mapping_container)

    def _init_connections(self):
        state.channel_names_changed.connect(self._on_state_channel_names_changed)
        state.image_loaded.connect(self._on_image_loaded)

    def _sync_channels(self):
        """Loads channels from image metadata and populates text inputs."""
        metadata = image_manager.get_metadata()
        channels_count = metadata.get("channels", 1)

        # Clear naming inputs
        while self.mapping_layout.count() > 0:
            item = self.mapping_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get naming mapping
        names = state.channel_names
        if not names or len(names) != channels_count:
            from lumen.core.fluorescence.channels import get_default_channel_names
            names = get_default_channel_names(channels_count, metadata.get("filename", ""))
            state.channel_names = names

        if channels_count > 1:
            for idx, name in enumerate(names):
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
            self.mapping_title.hide()
            self.mapping_container.hide()

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
        self._sync_channels()

    @Slot(str)
    def _on_image_loaded(self, path: str):
        if self.sender() is not state:
            self._sync_channels()
