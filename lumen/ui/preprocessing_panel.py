from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton, QFrame, QGridLayout
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCursor
from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager
from lumen.core.logger import logger

class FocusWheelSlider(QSlider):
    """QSlider subclass that only consumes mouse wheel events when focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()

class PreprocessingPanel(QWidget):
    """Surgical interface card implementing modular non-destructive preprocessing controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreprocessingPanel")
        self._setup_ui()
        self._init_connections()
        self.sync_from_state()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Header Title
        header = QLabel("Image Preprocessing")
        header.setStyleSheet("font-weight: bold; font-size: 11px; color: #818CF8; text-transform: uppercase; letter-spacing: 0.5px;")
        layout.addWidget(header)

        # Grid layout for parameters
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)

        # 1. Auto Contrast Checkbox
        self.auto_contrast_chk = QCheckBox("Auto Contrast (Percentile Stretch)")
        self.auto_contrast_chk.setCursor(QCursor(Qt.PointingHandCursor))
        self.auto_contrast_chk.setStyleSheet("font-size: 11px; color: #E5E7EB; font-weight: 500;")
        grid.addWidget(self.auto_contrast_chk, 0, 0, 1, 3)

        # 2. Percentile Low Slider
        self.p_low_lbl = QLabel("Low Cutoff:")
        self.p_low_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0;")
        self.p_low_slider = FocusWheelSlider(Qt.Horizontal)
        self.p_low_slider.setRange(0, 100)  # Represents 0.0% to 10.0% (val / 10.0)
        self.p_low_val_lbl = QLabel("1.0%")
        self.p_low_val_lbl.setStyleSheet("font-size: 11px; color: #34D399; font-weight: 600; min-width: 40px; qproperty-alignment: AlignRight;")
        grid.addWidget(self.p_low_lbl, 1, 0)
        grid.addWidget(self.p_low_slider, 1, 1)
        grid.addWidget(self.p_low_val_lbl, 1, 2)

        # 3. Percentile High Slider
        self.p_high_lbl = QLabel("High Cutoff:")
        self.p_high_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0;")
        self.p_high_slider = FocusWheelSlider(Qt.Horizontal)
        self.p_high_slider.setRange(900, 1000)  # Represents 90.0% to 100.0% (val / 10.0)
        self.p_high_val_lbl = QLabel("99.0%")
        self.p_high_val_lbl.setStyleSheet("font-size: 11px; color: #34D399; font-weight: 600; min-width: 40px; qproperty-alignment: AlignRight;")
        grid.addWidget(self.p_high_lbl, 2, 0)
        grid.addWidget(self.p_high_slider, 2, 1)
        grid.addWidget(self.p_high_val_lbl, 2, 2)

        # 4. Brightness Slider
        self.brightness_lbl = QLabel("Brightness:")
        self.brightness_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0;")
        self.brightness_slider = FocusWheelSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)  # Represents -1.0 to 1.0
        self.brightness_val_lbl = QLabel("0.00")
        self.brightness_val_lbl.setStyleSheet("font-size: 11px; color: #60A5FA; font-weight: 600; min-width: 40px; qproperty-alignment: AlignRight;")
        grid.addWidget(self.brightness_lbl, 3, 0)
        grid.addWidget(self.brightness_slider, 3, 1)
        grid.addWidget(self.brightness_val_lbl, 3, 2)

        # 5. Contrast Slider
        self.contrast_lbl = QLabel("Contrast:")
        self.contrast_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0;")
        self.contrast_slider = FocusWheelSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)  # Represents 0.1 to 3.0 (val / 100.0)
        self.contrast_val_lbl = QLabel("1.00")
        self.contrast_val_lbl.setStyleSheet("font-size: 11px; color: #F59E0B; font-weight: 600; min-width: 40px; qproperty-alignment: AlignRight;")
        grid.addWidget(self.contrast_lbl, 4, 0)
        grid.addWidget(self.contrast_slider, 4, 1)
        grid.addWidget(self.contrast_val_lbl, 4, 2)

        # 6. Gamma Slider
        self.gamma_lbl = QLabel("Gamma:")
        self.gamma_lbl.setStyleSheet("font-size: 11px; color: #9EA4B0;")
        self.gamma_slider = FocusWheelSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)  # Represents 0.1 to 3.0 (val / 100.0)
        self.gamma_val_lbl = QLabel("1.00")
        self.gamma_val_lbl.setStyleSheet("font-size: 11px; color: #A78BFA; font-weight: 600; min-width: 40px; qproperty-alignment: AlignRight;")
        grid.addWidget(self.gamma_lbl, 5, 0)
        grid.addWidget(self.gamma_slider, 5, 1)
        grid.addWidget(self.gamma_val_lbl, 5, 2)

        layout.addLayout(grid)

        # Reset button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setObjectName("ResetPreprocessButton")
        self.reset_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.reset_btn.setStyleSheet("""
            QPushButton#ResetPreprocessButton {
                background-color: transparent;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                color: #9CA3AF;
                padding: 5px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton#ResetPreprocessButton:hover {
                border-color: rgba(255, 255, 255, 0.2);
                color: #FFFFFF;
                background-color: rgba(255, 255, 255, 0.02);
            }
        """)
        layout.addWidget(self.reset_btn)

    def _init_connections(self):
        self.auto_contrast_chk.toggled.connect(self._on_auto_contrast_toggled)
        self.p_low_slider.valueChanged.connect(self._on_p_low_changed)
        self.p_high_slider.valueChanged.connect(self._on_p_high_changed)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        # Connect to state updates to keep controls synchronized
        state.preprocessing_changed.connect(self.sync_from_state)

    def sync_from_state(self):
        """Updates UI sliders and labels from current AppState."""
        self.blockSignals(True)
        
        # Auto contrast
        auto = state.preprocess_auto_contrast
        self.auto_contrast_chk.setChecked(auto)
        
        # Cutoffs (low/high)
        p_low_val = state.preprocess_percentile_low
        self.p_low_slider.setValue(int(p_low_val * 10))
        self.p_low_val_lbl.setText(f"{p_low_val:.1f}%")
        self.p_low_slider.setEnabled(auto)
        self.p_low_lbl.setEnabled(auto)
        
        p_high_val = state.preprocess_percentile_high
        self.p_high_slider.setValue(int(p_high_val * 10))
        self.p_high_val_lbl.setText(f"{p_high_val:.1f}%")
        self.p_high_slider.setEnabled(auto)
        self.p_high_lbl.setEnabled(auto)

        # Brightness
        brightness_val = state.preprocess_brightness
        self.brightness_slider.setValue(int(brightness_val * 100))
        self.brightness_val_lbl.setText(f"{brightness_val:+.2f}")

        # Contrast
        contrast_val = state.preprocess_contrast
        self.contrast_slider.setValue(int(contrast_val * 100))
        self.contrast_val_lbl.setText(f"{contrast_val:.2f}")

        # Gamma
        gamma_val = state.preprocess_gamma
        self.gamma_slider.setValue(int(gamma_val * 100))
        self.gamma_val_lbl.setText(f"{gamma_val:.2f}")

        self.blockSignals(False)

    @Slot(bool)
    def _on_auto_contrast_toggled(self, checked: bool):
        state.preprocess_auto_contrast = checked
        self.p_low_slider.setEnabled(checked)
        self.p_low_lbl.setEnabled(checked)
        self.p_high_slider.setEnabled(checked)
        self.p_high_lbl.setEnabled(checked)
        self._trigger_view_update()

    @Slot(int)
    def _on_p_low_changed(self, val: int):
        f_val = val / 10.0
        self.p_low_val_lbl.setText(f"{f_val:.1f}%")
        state.preprocess_percentile_low = f_val
        self._trigger_view_update()

    @Slot(int)
    def _on_p_high_changed(self, val: int):
        f_val = val / 10.0
        self.p_high_val_lbl.setText(f"{f_val:.1f}%")
        state.preprocess_percentile_high = f_val
        self._trigger_view_update()

    @Slot(int)
    def _on_brightness_changed(self, val: int):
        f_val = val / 100.0
        self.brightness_val_lbl.setText(f"{f_val:+.2f}")
        state.preprocess_brightness = f_val
        self._trigger_view_update()

    @Slot(int)
    def _on_contrast_changed(self, val: int):
        f_val = val / 100.0
        self.contrast_val_lbl.setText(f"{f_val:.2f}")
        state.preprocess_contrast = f_val
        self._trigger_view_update()

    @Slot(int)
    def _on_gamma_changed(self, val: int):
        f_val = val / 100.0
        self.gamma_val_lbl.setText(f"{f_val:.2f}")
        state.preprocess_gamma = f_val
        self._trigger_view_update()

    @Slot()
    def reset_to_defaults(self):
        state.blockSignals(True)
        state.preprocess_auto_contrast = True
        state.preprocess_percentile_low = 1.0
        state.preprocess_percentile_high = 99.0
        state.preprocess_brightness = 0.0
        state.preprocess_contrast = 1.0
        state.preprocess_gamma = 1.0
        state.blockSignals(False)
        
        state.preprocessing_changed.emit()
        self._trigger_view_update()

    def _trigger_view_update(self):
        """Forces ImageManager cache update and broadcasts image_loaded to repaint screen."""
        if state.current_image_path:
            # Force cache update
            image_manager.set_active_channel(state.active_viewer_channel)
            # Notify canvas to paint the new preprocessed image
            state.image_loaded.emit(state.current_image_path)
