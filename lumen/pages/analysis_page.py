import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QSizePolicy, QGridLayout, QProgressBar, QComboBox, QCheckBox, QSlider, QScrollArea, QSplitter, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QPixmap, QPainter, QCursor
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager
from lumen.processing.processing_manager import processing_manager
from lumen.workflows.workflow_manager import workflow_manager
from lumen.core.services.theme_service import theme_service
from lumen.core.services.gpu_service import gpu_service

# UI Spacing System
UI_SPACING_XS = 8   # Micro elements, compact layouts
UI_SPACING_SM = 12  # Element grouping padding
UI_SPACING_MD = 16  # Card padding, column margins
UI_SPACING_LG = 24  # Outer margins, page spacing

# Theme Colors (Default Dark Mode)
COLOR_BACKGROUND = "#0B0B0D"
COLOR_CARD = "#17171C"
COLOR_BORDER = "rgba(255, 255, 255, 0.06)"
COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#9EA4B0"
COLOR_ACCENT = "#6366F1"

class FocusWheelComboBox(QComboBox):
    """QComboBox subclass that only consumes mouse wheel events when focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()

class FocusWheelSlider(QSlider):
    """QSlider subclass that only consumes mouse wheel events when focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()

class FocusWheelSpinBox(QSpinBox):
    """QSpinBox subclass that only consumes mouse wheel events when focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()

class FocusWheelDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox subclass that only consumes mouse wheel events when focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()

class AnalysisParameterCard(QFrame):
    """Standardized scientific parameter section card layout."""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("AnalysisParameterCard")
        self.setStyleSheet(f"""
            #AnalysisParameterCard {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(UI_SPACING_SM, UI_SPACING_SM, UI_SPACING_SM, UI_SPACING_SM)
        self.layout.setSpacing(UI_SPACING_XS)
        
        self.header = QLabel(title)
        self.header.setStyleSheet("font-weight: bold; font-size: 11px; color: #5BE7FF; text-transform: uppercase; letter-spacing: 0.5px;")
        self.layout.addWidget(self.header)

class CollapsibleSection(QWidget):
    """Instant open/close panel without animation delays."""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.toggle_btn = QPushButton(f"▼ {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                font-size: 11px;
                color: #FFFFFF;
                background-color: #24242D;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 4px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #2D2D37;
            }
            QPushButton:checked {
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
        """)
        self.layout.addWidget(self.toggle_btn)
        
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        """)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(UI_SPACING_SM, UI_SPACING_SM, UI_SPACING_SM, UI_SPACING_SM)
        self.content_layout.setSpacing(UI_SPACING_XS)
        self.layout.addWidget(self.content_frame)
        
        self.toggle_btn.toggled.connect(self._on_toggle)
        
    def set_content(self, widget: QWidget):
        self.content_layout.addWidget(widget)

    def _on_toggle(self, checked: bool):
        self.content_frame.setVisible(checked)
        self.toggle_btn.setText(f"▼ {self.toggle_btn.text()[2:]}" if checked else f"▶ {self.toggle_btn.text()[2:]}")

class AnalysisPlaceholderWidget(QFrame):
    """Elegant scientific placeholder shown when no image is active in the viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AnalysisPlaceholder")
        self.setStyleSheet("""
            #AnalysisPlaceholder {
                background-color: #131317;
                border: 1px dashed #2B2B35;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        # Scientific Icon
        icon_lbl = QLabel("🔬")
        icon_lbl.setStyleSheet("font-size: 40px; color: #6366F1; margin-bottom: 4px;")
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

        # Title
        title_lbl = QLabel("No Image Loaded")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        # Description
        desc_lbl = QLabel("Upload an image or open one from Batch Explorer.")
        desc_lbl.setStyleSheet("font-size: 12px; color: #9CA3AF;")
        desc_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_lbl)

        # Small Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #2C2C35; max-height: 1px; max-width: 140px; margin: 4px 0;")
        layout.addWidget(sep, 0, Qt.AlignCenter)

        # Formats list
        formats_box = QFrame()
        formats_layout = QVBoxLayout(formats_box)
        formats_layout.setSpacing(2)
        formats_layout.setContentsMargins(0, 0, 0, 0)
        formats_layout.setAlignment(Qt.AlignCenter)

        formats_lbl = QLabel("Supported Formats:")
        formats_lbl.setStyleSheet("font-size: 9px; font-weight: bold; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;")
        formats_val = QLabel("TIFF  •  PNG  •  JPG")
        formats_val.setStyleSheet("font-size: 11px; font-weight: 600; color: #E5E7EB;")
        
        formats_lbl.setAlignment(Qt.AlignCenter)
        formats_val.setAlignment(Qt.AlignCenter)
        
        formats_layout.addWidget(formats_lbl)
        formats_layout.addWidget(formats_val)
        layout.addWidget(formats_box)

        # Recommendations list
        recs_box = QFrame()
        recs_layout = QVBoxLayout(recs_box)
        recs_layout.setSpacing(2)
        recs_layout.setContentsMargins(0, 0, 0, 0)
        recs_layout.setAlignment(Qt.AlignCenter)

        recs_lbl = QLabel("Recommended Modalities:")
        recs_lbl.setStyleSheet("font-size: 9px; font-weight: bold; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;")
        recs_val = QLabel("Fluorescence  |  Brightfield  |  Histology")
        recs_val.setStyleSheet("font-size: 11px; color: #9CA3AF;")

        recs_lbl.setAlignment(Qt.AlignCenter)
        recs_val.setAlignment(Qt.AlignCenter)

        recs_layout.addWidget(recs_lbl)
        recs_layout.addWidget(recs_val)
        layout.addWidget(recs_box)

    def sync_theme(self, theme_name: str):
        if theme_name == "light":
            self.setStyleSheet("""
                #AnalysisPlaceholder {
                    background-color: #FFFFFF;
                    border: 1px dashed #D1D5DB;
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                #AnalysisPlaceholder {
                    background-color: #131317;
                    border: 1px dashed #2B2B35;
                    border-radius: 8px;
                }
            """)


class InteractiveImageViewer(QGraphicsView):
    """Zoomable, pannable graphics canvas for image inspection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Labeled masks overlay layered directly on top of base image
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(1.0)
        self.mask_item.setOpacity(0.4) # default 40%
        self.scene.addItem(self.mask_item)

        # Click highlight item tracking
        self.highlight_item = None

        # Viewer Settings
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Panning behavior
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #000000; border: 1px solid #2B2B35; border-radius: 6px;")

        # Anchor transformation under mouse for intuitive scientific inspection zooming
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._zoom_factor = 1.15
        self._zoom_touched = False
        self._initial_fit_scale = None
        self._is_panning = False
        self._analysis_results = None
        
        # Premium Empty State Placeholder
        self._placeholder = AnalysisPlaceholderWidget(self)

        # Loading Overlay (overlay frame)
        self.loading_overlay = QFrame(self)
        self.loading_overlay.setObjectName("ViewerLoadingOverlay")
        self.loading_overlay.setStyleSheet("""
            #ViewerLoadingOverlay {
                background-color: rgba(11, 11, 13, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
            }
        """)
        overlay_layout = QVBoxLayout(self.loading_overlay)
        overlay_layout.setAlignment(Qt.AlignCenter)
        overlay_layout.setSpacing(10)
        
        self.loading_icon = QLabel("⚡")
        self.loading_icon.setStyleSheet("font-size: 32px; color: #6366F1;")
        self.loading_icon.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.loading_icon)
        
        self.loading_text = QLabel("Analyzing image...")
        self.loading_text.setStyleSheet("color: #FFFFFF; font-size: 13px; font-weight: bold;")
        self.loading_text.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.loading_text)
        
        self.loading_subtext = QLabel("GPU Active")
        self.loading_subtext.setStyleSheet("color: #9EA4B0; font-size: 11px;")
        self.loading_subtext.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.loading_subtext)
        
        self.loading_overlay.hide()

    def set_image(self, pixmap: QPixmap, restore_state: dict = None):
        """Sets canvas image and resets viewport zoom."""
        self.clear_highlight()
        if pixmap and not pixmap.isNull():
            self._placeholder.hide()
            self.pixmap_item.setPixmap(pixmap)
            # Clear existing mask overlay on loading new image
            self.mask_item.setPixmap(QPixmap())
            self.mask_item.setVisible(False)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            
            if restore_state:
                self.setTransform(restore_state["transform"])
                self.horizontalScrollBar().setValue(restore_state["h_scroll"])
                self.verticalScrollBar().setValue(restore_state["v_scroll"])
                self._initial_fit_scale = restore_state["initial_fit_scale"]
                self._zoom_touched = restore_state["zoom_touched"]
                logger.debug("ImageViewer: Displaying loaded image and restoring view state.")
            else:
                # Reset transform and fit completely to window on initial load
                self.resetTransform()
                self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                
                # Store initial scale for clamping ratios
                self._initial_fit_scale = self.transform().m11()
                self._zoom_touched = False
                logger.debug("ImageViewer: Displaying loaded image on canvas. Fit scale: %s", self._initial_fit_scale)
            self.update_viewer_cursor()
        else:
            self.clear()

    def clear(self):
        """Clears canvas items and displays placeholder."""
        self.clear_highlight()
        self.pixmap_item.setPixmap(QPixmap())
        self.mask_item.setPixmap(QPixmap())
        self.mask_item.setVisible(False)
        self.scene.setSceneRect(0, 0, 0, 0)
        self.resetTransform()
        self._initial_fit_scale = None
        self._zoom_touched = False
        self.update_viewer_cursor()
        self._placeholder.show()
        self._placeholder.setGeometry(self.rect())
        if hasattr(self, "loading_overlay") and self.loading_overlay:
            self.loading_overlay.hide()

    def set_canvas_background(self, color_hex: str):
        self.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #2B2B35; border-radius: 6px;")

    def show_loading(self, text: str, gpu_active: bool = True):
        self.loading_text.setText(text)
        self.loading_subtext.setVisible(gpu_active)
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.show()
        # Process events to force rendering overlay immediately
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def hide_loading(self):
        self.loading_overlay.hide()

    def zoom_in(self):
        current_scale = self.transform().m11()
        if self._initial_fit_scale is None:
            self._initial_fit_scale = current_scale
        factor = self._zoom_factor
        new_scale = current_scale * factor
        max_limit = self._initial_fit_scale * 20.0
        if new_scale > max_limit:
            new_scale = max_limit
        relative_factor = new_scale / current_scale
        self.scale(relative_factor, relative_factor)
        self._zoom_touched = True
        self.update_viewer_cursor()

    def zoom_out(self):
        current_scale = self.transform().m11()
        if self._initial_fit_scale is None:
            self._initial_fit_scale = current_scale
        factor = 1.0 / self._zoom_factor
        new_scale = current_scale * factor
        min_limit = self._initial_fit_scale * 0.1
        if new_scale < min_limit:
            new_scale = min_limit
        relative_factor = new_scale / current_scale
        self.scale(relative_factor, relative_factor)
        self._zoom_touched = True
        self.update_viewer_cursor()

    def fit_screen(self):
        if not self.pixmap_item.pixmap().isNull():
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._zoom_touched = False
            self.update_viewer_cursor()

    def set_masks(self, masks_arr: np.ndarray):
        """Generates a high-contrast colored overlay from the 2D integer masks array."""
        if masks_arr is not None and masks_arr.size > 0:
            h, w = masks_arr.shape
            unique_labels = np.unique(masks_arr)
            
            # Create a 4-channel RGBA output array (h, w, 4) initialized to 0 (fully transparent)
            rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Seed generator for consistent colors per label ID
            np.random.seed(42)
            
            import colorsys
            for label in unique_labels:
                if label == 0:
                    continue
                # Generate a high-contrast color for this label via HSV mapping
                h_val = np.random.rand()
                s_val = 0.8 + 0.2 * np.random.rand()
                v_val = 0.8 + 0.2 * np.random.rand()
                
                r, g, b = colorsys.hsv_to_rgb(h_val, s_val, v_val)
                rgba_arr[masks_arr == label] = [int(r * 255), int(g * 255), int(b * 255), 102]  # 40% opacity
                
            # Convert RGBA array to QImage
            from PySide6.QtGui import QImage
            bytes_per_line = w * 4
            qimg = QImage(rgba_arr.tobytes(), w, h, bytes_per_line, QImage.Format_RGBA8888).copy()
            
            # Convert to QPixmap and apply to mask_item
            pixmap = QPixmap.fromImage(qimg)
            self.mask_item.setPixmap(pixmap)
            self.mask_item.setVisible(state.show_segmentation_overlay)
            self.scene.update()
            logger.info("ImageViewer: Labeled mask overlay rendered. Shape: %dx%d, unique labels: %d", w, h, len(unique_labels) - 1)
        else:
            self.mask_item.setPixmap(QPixmap())
            self.mask_item.setVisible(False)
            self.scene.update()

    def wheelEvent(self, event):
        """Handles zooming with mouse wheel scrolls."""
        if not self.pixmap_item.pixmap().isNull():
            event.accept()
            delta = event.angleDelta().y()
            if delta == 0:
                return

            current_scale = self.transform().m11()
            if self._initial_fit_scale is None:
                self._initial_fit_scale = current_scale

            # Incremental scale factor
            factor = self._zoom_factor if delta > 0 else (1.0 / self._zoom_factor)
            new_scale = current_scale * factor

            # Clamping constraints (10x-20x zoom-in max, 0.1x zoom-out min)
            max_limit = self._initial_fit_scale * 20.0
            min_limit = self._initial_fit_scale * 0.1

            if new_scale > max_limit:
                new_scale = max_limit
            elif new_scale < min_limit:
                new_scale = min_limit

            relative_factor = new_scale / current_scale
            self.scale(relative_factor, relative_factor)
            self._zoom_touched = True
            self.update_viewer_cursor()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._placeholder.setGeometry(self.rect())
        if hasattr(self, "loading_overlay") and self.loading_overlay:
            self.loading_overlay.setGeometry(self.rect())
        # Only fit to view automatically on resize if user has not yet interacted with zoom/pan
        if not self.pixmap_item.pixmap().isNull() and not self._zoom_touched:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.pixmap_item.pixmap().isNull():
            self._is_panning = True
            self._pan_start_x = event.position().x()
            self._pan_start_y = event.position().y()
            self._h_bar_start = self.horizontalScrollBar().value()
            self._v_bar_start = self.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            self._press_pos = event.position()
            self._panned_distance = 0.0
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, "_is_panning") and self._is_panning:
            dx = event.position().x() - self._pan_start_x
            dy = event.position().y() - self._pan_start_y
            self.horizontalScrollBar().setValue(self._h_bar_start - dx)
            self.verticalScrollBar().setValue(self._v_bar_start - dy)
            self._zoom_touched = True
            if hasattr(self, "_press_pos"):
                dist = (event.position() - self._press_pos).manhattanLength()
                self._panned_distance = max(self._panned_distance, dist)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, "_is_panning") and self._is_panning:
            self._is_panning = False
            self.update_viewer_cursor()
            if hasattr(self, "_panned_distance") and self._panned_distance < 5.0:
                self._handle_cell_click(event)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_analysis_results(self, results: dict):
        """Sets the active results context for cell-click inspection."""
        self._analysis_results = results

    def _handle_cell_click(self, event):
        self.clear_highlight()
        from PySide6.QtWidgets import QToolTip
        QToolTip.hideText()
        
        results = self._analysis_results
        if not results:
            return
            
        masks = results.get("masks")
        if masks is None or masks.size == 0:
            return
            
        scene_pos = self.mapToScene(event.position().toPoint())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        
        h, w = masks.shape
        if 0 <= x < w and 0 <= y < h:
            cell_id = int(masks[y, x])
            if cell_id > 0:
                metrics_dict = results.get("cell_metrics", {})
                cell_info = metrics_dict.get(cell_id)
                if cell_info:
                    area = cell_info["area_px"]
                    diam = cell_info["diameter_px"]
                    cx, cy = cell_info["centroid"]
                    
                    tooltip_text = (
                        f"<b>Cell ID:</b> {cell_id}<br/>"
                        f"<b>Area:</b> {area} px<br/>"
                        f"<b>Diameter:</b> {diam} px<br/>"
                        f"<b>Centroid:</b> ({cx}, {cy})"
                    )
                    
                    from PySide6.QtGui import QColor
                    QToolTip.showText(QCursor.pos(), tooltip_text, self)
                    self._highlight_cell(cell_id, masks)

    def _highlight_cell(self, cell_id: int, masks: np.ndarray):
        try:
            h, w = masks.shape
            rows, cols = np.where(masks == cell_id)
            if len(rows) == 0:
                return
                
            min_row, max_row = int(np.min(rows)), int(np.max(rows))
            min_col, max_col = int(np.min(cols)), int(np.max(cols))
            
            sub_h = max_row - min_row + 1
            sub_w = max_col - min_col + 1
            
            from PySide6.QtGui import QImage, QColor, QPixmap
            qimg = QImage(sub_w, sub_h, QImage.Format_ARGB32)
            qimg.fill(Qt.transparent)
            
            sub_mask = masks[min_row:max_row+1, min_col:max_col+1]
            
            for r in range(sub_h):
                for c in range(sub_w):
                    if sub_mask[r, c] == cell_id:
                        is_boundary = False
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if nr < 0 or nr >= sub_h or nc < 0 or nc >= sub_w or sub_mask[nr, nc] != cell_id:
                                is_boundary = True
                                break
                        
                        if is_boundary:
                            qimg.setPixelColor(c, r, QColor(255, 255, 0, 255))
                        else:
                            qimg.setPixelColor(c, r, QColor(255, 255, 0, 90))
                            
            self.highlight_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
            self.highlight_item.setPos(min_col, min_row)
            self.highlight_item.setZValue(2.0)
            self.scene.addItem(self.highlight_item)
            logger.debug("ImageViewer: Interactive highlight drawn for Cell %d", cell_id)
        except Exception as e:
            logger.error("ImageViewer: Cell highlight generation failed: %s", e)

    def set_mask_opacity(self, opacity: int):
        """Sets the opacity of the segmentation mask layer."""
        self.mask_item.setOpacity(opacity / 100.0)

    def set_show_original(self, show: bool):
        """Toggles visibility of the original raw image background."""
        self.pixmap_item.setVisible(show)

    def set_show_overlay(self, show: bool):
        """Toggles visibility of the segmentation mask layer."""
        self.mask_item.setVisible(show)

    def clear_highlight(self):
        """Clears any active interactive cell highlight from the scene."""
        if self.highlight_item:
            try:
                self.scene.removeItem(self.highlight_item)
            except Exception:
                pass
            self.highlight_item = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self.pixmap_item.pixmap().isNull():
            # Reset view to fit-to-window
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._zoom_touched = False
            self.update_viewer_cursor()
            logger.debug("ImageViewer: View reset to fit-to-window by double click.")
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def update_viewer_cursor(self):
        """Dynamic cursor representation indicating whether panning is active."""
        if self.pixmap_item.pixmap().isNull():
            self.setCursor(Qt.ArrowCursor)
            return
        
        # Hovering hand indicates zoomed in and inspectable panning capability
        if self._zoom_touched:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def sync_theme(self, theme_name: str):
        self._placeholder.sync_theme(theme_name)
        if theme_name == "light":
            self.setStyleSheet("background-color: #F3F4F6; border: 1px solid #D1D5DB; border-radius: 6px;")
        else:
            self.setStyleSheet("background-color: #0B0B0D; border: 1px solid #2B2B35; border-radius: 6px;")



class AnalysisPage(QWidget):
    """Workspace page layout including config panel, viewer, and run controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_image_path = None
        self._loaded_image_origin = None
        self._setup_ui()
        self._init_connections()
        self._sync_theme()

    def _setup_ui(self):
        # Spacing: LG = 24px outer margins
        self.page_layout = QVBoxLayout(self)
        self.page_layout.setObjectName("PageVerticalLayout")
        self.page_layout.setContentsMargins(UI_SPACING_LG, UI_SPACING_LG, UI_SPACING_LG, UI_SPACING_LG)
        self.page_layout.setSpacing(UI_SPACING_SM)

        from lumen.ui.workspace_switcher import WorkspaceSwitcher
        self.workspace_switcher = WorkspaceSwitcher("single")
        self.page_layout.addWidget(self.workspace_switcher)

        self.main_layout = QHBoxLayout()
        self.main_layout.setObjectName("PageContainer")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(UI_SPACING_MD)

        # 1. Left Panel: Image Properties Card (resizable)
        self.left_panel = QFrame()
        self.left_panel.setObjectName("AnalysisLeftPanel")
        self.left_panel.setMinimumWidth(220)
        self.left_panel.setMaximumWidth(320)
        self.left_panel.setStyleSheet(f"""
            #AnalysisLeftPanel {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(UI_SPACING_MD, UI_SPACING_MD, UI_SPACING_MD, UI_SPACING_MD)
        left_layout.setSpacing(UI_SPACING_MD)

        left_title = QLabel("Image Properties")
        left_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        left_layout.addWidget(left_title)

        # Empty State Placeholder
        self.meta_placeholder = QLabel("No active image metadata.\nImport an image to inspect details.")
        self.meta_placeholder.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px;")
        self.meta_placeholder.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.meta_placeholder, 1)

        # Metadata Card Layout
        self.meta_container = QWidget()
        meta_layout = QVBoxLayout(self.meta_container)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(UI_SPACING_SM)

        # Section 1: Image Details Info
        image_sec = QFrame()
        image_sec.setStyleSheet(f"border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: {UI_SPACING_SM}px;")
        image_sec_layout = QGridLayout(image_sec)
        image_sec_layout.setContentsMargins(0, 0, 0, 0)
        image_sec_layout.setSpacing(UI_SPACING_XS)

        fn_lbl = QLabel("📄 Filename:")
        fn_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        self.fn_val = QLabel("-")
        self.fn_val.setWordWrap(True)
        self.fn_val.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")

        res_lbl = QLabel("📐 Resolution:")
        res_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        self.res_val = QLabel("-")
        self.res_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        ch_lbl = QLabel("🧬 Channels:")
        ch_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        self.ch_val = QLabel("-")
        self.ch_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        mode_lbl = QLabel("🔬 Mode:")
        mode_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        self.mode_val = QLabel("-")
        self.mode_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        image_sec_layout.addWidget(fn_lbl, 0, 0)
        image_sec_layout.addWidget(self.fn_val, 0, 1)
        image_sec_layout.addWidget(res_lbl, 1, 0)
        image_sec_layout.addWidget(self.res_val, 1, 1)
        image_sec_layout.addWidget(ch_lbl, 2, 0)
        image_sec_layout.addWidget(self.ch_val, 2, 1)
        image_sec_layout.addWidget(mode_lbl, 3, 0)
        image_sec_layout.addWidget(self.mode_val, 3, 1)
        meta_layout.addWidget(image_sec)

        # Section 2: Calibration Info
        calib_sec = QFrame()
        calib_sec.setStyleSheet(f"border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: {UI_SPACING_SM}px;")
        calib_sec_layout = QGridLayout(calib_sec)
        calib_sec_layout.setContentsMargins(0, 0, 0, 0)
        calib_sec_layout.setSpacing(UI_SPACING_XS)

        voxel_lbl = QLabel("📏 Pixel Size:")
        voxel_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        self.voxel_val = QLabel("-")
        self.voxel_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        type_lbl = QLabel("📍 Type:")
        type_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        self.type_val = QLabel("-")
        self.type_val.setWordWrap(True)
        self.type_val.setStyleSheet("color: #6366F1; font-size: 12px; font-weight: bold;")

        calib_sec_layout.addWidget(voxel_lbl, 0, 0)
        calib_sec_layout.addWidget(self.voxel_val, 0, 1)
        calib_sec_layout.addWidget(type_lbl, 1, 0)
        calib_sec_layout.addWidget(self.type_val, 1, 1)
        meta_layout.addWidget(calib_sec)

        left_layout.addWidget(self.meta_container, 1)
        self.meta_container.setVisible(False)

        # 2. Center Panel: Image Viewer with controls underneath
        self.center_container = QFrame()
        center_layout = QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(UI_SPACING_SM)

        self.image_viewer = InteractiveImageViewer(self)
        center_layout.addWidget(self.image_viewer, 1)

        # Viewer Controls Bar underneath the canvas
        self.viewer_controls_bar = QFrame()
        self.viewer_controls_bar.setObjectName("ViewerControlsBar")
        self.viewer_controls_bar.setStyleSheet(f"""
            #ViewerControlsBar {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: {UI_SPACING_SM}px;
            }}
        """)
        controls_bar_layout = QVBoxLayout(self.viewer_controls_bar)
        controls_bar_layout.setContentsMargins(UI_SPACING_SM, UI_SPACING_SM, UI_SPACING_SM, UI_SPACING_SM)
        controls_bar_layout.setSpacing(UI_SPACING_XS)

        # Row 1: Visibility Layers & Opacity Slider
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(UI_SPACING_MD)

        self.show_original_chk = QCheckBox("Show Original Image")
        self.show_original_chk.setChecked(True)
        self.show_original_chk.setCursor(Qt.PointingHandCursor)
        self.show_original_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")
        
        self.show_overlay_chk = QCheckBox("Show Segmentation Overlay")
        self.show_overlay_chk.setChecked(True)
        self.show_overlay_chk.setCursor(Qt.PointingHandCursor)
        self.show_overlay_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")

        opacity_lbl = QLabel("Mask Opacity:")
        opacity_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")
        
        self.opacity_slider = FocusWheelSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.setCursor(Qt.PointingHandCursor)
        
        self.opacity_val_lbl = QLabel("40%")
        self.opacity_val_lbl.setStyleSheet("font-size: 11px; color: #E5E7EB; min-width: 30px;")

        row1_layout.addWidget(self.show_original_chk)
        row1_layout.addWidget(self.show_overlay_chk)
        row1_layout.addWidget(opacity_lbl)
        row1_layout.addWidget(self.opacity_slider)
        row1_layout.addWidget(self.opacity_val_lbl)
        row1_layout.addStretch(1)
        controls_bar_layout.addLayout(row1_layout)

        # Row 2: Channel Display dropdown, Zoom control buttons, Auto contrast
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(UI_SPACING_MD)

        ch_disp_lbl = QLabel("Channel Display:")
        ch_disp_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY}; font-weight: bold;")
        self.viewer_combo = FocusWheelComboBox()
        self.viewer_combo.setCursor(Qt.PointingHandCursor)
        self.viewer_combo.setFixedWidth(160)
        self.viewer_combo.addItem("Channel 0: Grayscale", 0)

        # Zoom buttons
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(4)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.setFixedSize(24, 24)
        self.zoom_in_btn.setCursor(Qt.PointingHandCursor)
        self.zoom_in_btn.setStyleSheet("font-weight: bold; font-size: 12px; padding: 0;")
        
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.setFixedSize(24, 24)
        self.zoom_out_btn.setCursor(Qt.PointingHandCursor)
        self.zoom_out_btn.setStyleSheet("font-weight: bold; font-size: 12px; padding: 0;")

        self.zoom_fit_btn = QPushButton("⛶")
        self.zoom_fit_btn.setToolTip("Fit Screen")
        self.zoom_fit_btn.setFixedSize(24, 24)
        self.zoom_fit_btn.setCursor(Qt.PointingHandCursor)
        self.zoom_fit_btn.setStyleSheet("font-size: 12px; padding: 0;")

        self.maximize_btn = QPushButton("🗖")
        self.maximize_btn.setToolTip("Toggle Maximize Viewer")
        self.maximize_btn.setFixedSize(24, 24)
        self.maximize_btn.setCheckable(True)
        self.maximize_btn.setCursor(Qt.PointingHandCursor)
        self.maximize_btn.setStyleSheet("font-size: 12px; padding: 0;")

        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_fit_btn)
        zoom_layout.addWidget(self.maximize_btn)

        self.viewer_contrast_lbl = QLabel("✨ Auto Contrast Applied")
        self.viewer_contrast_lbl.setStyleSheet("font-size: 10px; color: #818CF8; font-weight: bold;")
        self.viewer_contrast_lbl.setVisible(False)

        row2_layout.addWidget(ch_disp_lbl)
        row2_layout.addWidget(self.viewer_combo)
        row2_layout.addLayout(zoom_layout)
        row2_layout.addStretch(1)
        row2_layout.addWidget(self.viewer_contrast_lbl)
        controls_bar_layout.addLayout(row2_layout)

        center_layout.addWidget(self.viewer_controls_bar)

        # 3. Right Sidebar Panel: Parameters & Actions
        self.right_panel = QFrame()
        self.right_panel.setObjectName("AnalysisRightPanel")
        self.right_panel.setMinimumWidth(340)
        self.right_panel.setMaximumWidth(450)
        self.right_panel.setStyleSheet(f"""
            #AnalysisRightPanel {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # Scroll Area for parameters
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setStyleSheet("background: transparent; border: none;")

        scroll_widget = QWidget()
        scroll_widget.setObjectName("RightScrollWidget")
        scroll_widget.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 16, 0)
        scroll_layout.setSpacing(20)

        # Active Workflow (Always visible)
        wf_card = AnalysisParameterCard("Active Workflow")
        self.wf_lbl = QLabel("Active Workflow:")
        self.wf_lbl.setObjectName("WorkflowLabel")
        self.wf_lbl.setStyleSheet("font-weight: bold; font-size: 10px; color: #6366F1; text-transform: uppercase;")
        self.wf_combo = FocusWheelComboBox()
        self.wf_combo.setObjectName("WorkflowComboBox")
        self.wf_combo.addItem("Cell Segmentation", "cell_counting")
        self.wf_combo.addItem("Fluorescence Analysis", "fluorescence")
        self.wf_combo.setCursor(QCursor(Qt.PointingHandCursor))
        wf_card.layout.addWidget(self.wf_lbl)
        wf_card.layout.addWidget(self.wf_combo)
        scroll_layout.addWidget(wf_card)

        # Segmentation Parameter Card (Always visible)
        self.segmentation_card = AnalysisParameterCard("Segmentation Settings")
        
        self.method_lbl = QLabel("Segmentation Method:")
        self.method_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        self.method_combo = FocusWheelComboBox()
        self.method_combo.addItems(["AI Segmentation (Cellpose)"])
        self.method_combo.setCurrentText("AI Segmentation (Cellpose)")
        
        self.model_lbl = QLabel("Segmentation Model:")
        self.model_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        self.model_combo = FocusWheelComboBox()
        self.model_combo.addItems(["Auto", "Cyto3", "Nuclei"])
        self.model_combo.setCurrentText(state.segmentation_model)
        
        self.quality_lbl = QLabel("Segmentation Quality:")
        self.quality_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        self.quality_combo = FocusWheelComboBox()
        self.quality_combo.addItems(["Fast", "Balanced", "Sensitive", "Precise"])
        self.quality_combo.setCurrentText("Balanced")

        self.channel_frame = QWidget()
        channel_lay = QVBoxLayout(self.channel_frame)
        channel_lay.setContentsMargins(0, 0, 0, 0)
        channel_lay.setSpacing(4)
        self.channel_lbl = QLabel("Segmentation Channel:")
        self.channel_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        self.channel_combo = FocusWheelComboBox()
        channel_lay.addWidget(self.channel_lbl)
        channel_lay.addWidget(self.channel_combo)
        
        self.segmentation_card.layout.addWidget(self.method_lbl)
        self.segmentation_card.layout.addWidget(self.method_combo)
        self.segmentation_card.layout.addWidget(self.model_lbl)
        self.segmentation_card.layout.addWidget(self.model_combo)
        self.segmentation_card.layout.addWidget(self.quality_lbl)
        self.segmentation_card.layout.addWidget(self.quality_combo)
        self.segmentation_card.layout.addWidget(self.channel_frame)
        self.channel_frame.hide() # Shown dynamically
        scroll_layout.addWidget(self.segmentation_card)

        # Calibration Settings (Always visible)
        self.calib_card = AnalysisParameterCard("Calibration Settings")
        calib_lbl = QLabel("Calibration Mode:")
        calib_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        self.calib_combo = FocusWheelComboBox()
        self.calib_combo.addItem("Pixel Mode (px)", "pixel")
        self.calib_combo.addItem("Micron Mode (µm)", "micron")
        self.calib_card.layout.addWidget(calib_lbl)
        self.calib_card.layout.addWidget(self.calib_combo)
        scroll_layout.addWidget(self.calib_card)

        # Collapsible Section 1: Preprocessing controls
        self.prep_section = CollapsibleSection("Image Preprocessing")
        self.preprocess_container = QWidget()
        preprocess_card_layout = QVBoxLayout(self.preprocess_container)
        preprocess_card_layout.setContentsMargins(0, 0, 0, 0)
        
        from lumen.ui.preprocessing_panel import PreprocessingPanel
        self.preprocess_panel = PreprocessingPanel(self.preprocess_container)
        preprocess_card_layout.addWidget(self.preprocess_panel)
        self.prep_section.set_content(self.preprocess_container)
        scroll_layout.addWidget(self.prep_section)

        # Collapsible Section 2: Fluorescence Settings
        self.naming_section = CollapsibleSection("Fluorescence Settings")
        self.channel_controls_container = QWidget()
        channel_controls_layout = QVBoxLayout(self.channel_controls_container)
        channel_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        from lumen.ui.fluorescence_panel import FluorescencePanel
        self.fluorescence_panel = FluorescencePanel(self.channel_controls_container)
        channel_controls_layout.addWidget(self.fluorescence_panel)
        self.naming_section.set_content(self.channel_controls_container)
        scroll_layout.addWidget(self.naming_section)
        self.naming_section.hide() # Shown dynamically if channels > 1

        # Collapsible Section 3: Puncta Settings Placeholder
        self.puncta_section = CollapsibleSection("Puncta Settings")
        puncta_card = QWidget()
        puncta_lay = QVBoxLayout(puncta_card)
        puncta_lay.setContentsMargins(0, 0, 0, 0)
        puncta_placeholder_lbl = QLabel("Puncta analysis configuration will be available in upcoming phase.")
        puncta_placeholder_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")
        puncta_lay.addWidget(puncta_placeholder_lbl)
        self.puncta_section.set_content(puncta_card)
        scroll_layout.addWidget(self.puncta_section)
        self.puncta_section.hide() # Shown dynamically if puncta workflow selected

        scroll_layout.addStretch(1)
        self.right_scroll.setWidget(scroll_widget)
        right_layout.addWidget(self.right_scroll, 1)

        # Bottom section: Pinned action controls
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: {COLOR_BORDER}; max-height: 1px; margin: 4px 0;")
        right_layout.addWidget(sep)

        # Progress and status labels
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("AnalysisStatusLabel")
        self.status_lbl.setVisible(False)
        right_layout.addWidget(self.status_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("AnalysisProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Action Buttons Layout (Pinned bottom action dock)
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("RunAnalysisButton")
        self.run_btn.setProperty("class", "PrimaryButton")
        self.run_btn.setFixedHeight(50) # Prominent anchoring height
        self.run_btn.setCursor(QCursor(Qt.ArrowCursor))
        self.run_btn.setEnabled(False)
        right_layout.addWidget(self.run_btn)

        button_sep = QFrame()
        button_sep.setFrameShape(QFrame.HLine)
        button_sep.setStyleSheet(f"background-color: {COLOR_BORDER}; max-height: 1px; margin: 2px 0;")
        right_layout.addWidget(button_sep)

        self.edit_btn = QPushButton("✏ Edit Masks")
        self.edit_btn.setObjectName("EditMasksButton")
        self.edit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.edit_btn.setEnabled(False)
        right_layout.addWidget(self.edit_btn)

        secondary_btns_layout = QHBoxLayout()
        secondary_btns_layout.setSpacing(UI_SPACING_XS)
        
        self.save_analysis_btn = QPushButton("💾 Save Analysis")
        self.save_analysis_btn.setObjectName("SaveAnalysisButton")
        self.save_analysis_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.save_analysis_btn.setEnabled(False)
        
        self.reset_changes_btn = QPushButton("🔄 Reset Changes")
        self.reset_changes_btn.setObjectName("ResetChangesButton")
        self.reset_changes_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.reset_changes_btn.setEnabled(False)

        secondary_btns_layout.addWidget(self.save_analysis_btn)
        secondary_btns_layout.addWidget(self.reset_changes_btn)
        right_layout.addLayout(secondary_btns_layout)

        self.dirty_lbl = QLabel("")
        self.dirty_lbl.setObjectName("DirtyStatusLabel")
        self.dirty_lbl.setAlignment(Qt.AlignCenter)
        self.dirty_lbl.setVisible(False)
        self.dirty_lbl.setStyleSheet("color: #ff9800; font-weight: bold; margin: 4px;")
        right_layout.addWidget(self.dirty_lbl)

        # 4. Main Resizable QSplitter Configuration
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("AnalysisMainSplitter")
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.center_container)
        self.main_splitter.addWidget(self.right_panel)
        
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        
        self.main_splitter.setCollapsible(0, True) # Allow collapsing left
        self.main_splitter.setCollapsible(1, False) # Do not collapse viewer
        self.main_splitter.setCollapsible(2, True) # Allow collapsing right

        self.main_layout.addWidget(self.main_splitter, 1)
        self.page_layout.addLayout(self.main_layout, 1)

        # 5. Permanent Window Footer Status Bar
        self.footer_bar = QFrame()
        self.footer_bar.setObjectName("AnalysisFooterBar")
        self.footer_bar.setStyleSheet(f"""
            #AnalysisFooterBar {{
                background-color: #0E0E12;
                border-top: 1px solid {COLOR_BORDER};
                padding: 4px 12px;
            }}
        """)
        footer_layout = QHBoxLayout(self.footer_bar)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(16)
        
        self.footer_status = QLabel("Ready")
        self.footer_status.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")
        
        self.footer_gpu = QLabel("GPU: Active (CUDA)")
        self.footer_gpu.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")

        self.footer_res = QLabel("-")
        self.footer_res.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")
        
        self.footer_zoom = QLabel("Zoom: 100%")
        self.footer_zoom.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")

        footer_layout.addWidget(self.footer_status)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.footer_gpu)
        footer_layout.addWidget(self.footer_res)
        footer_layout.addWidget(self.footer_zoom)
        self.page_layout.addWidget(self.footer_bar)

        # Restore Splitter/Workflow layouts from QSettings
        self._restore_layout_state()
        
        # Connect splitter move triggers
        self.main_splitter.splitterMoved.connect(self._save_layout_state)

    def _init_connections(self):
        # Trigger actual Cellpose analysis on click
        self.run_btn.clicked.connect(self._on_run_analysis_clicked)
        
        # Trigger manual editing workspace dialog on click
        self.edit_btn.clicked.connect(self._on_edit_masks_clicked)
        self.reset_changes_btn.clicked.connect(self._on_reset_changes_clicked)
        
        # Connect visual controls updates
        self.show_original_chk.toggled.connect(self._on_show_original_toggled)
        self.show_overlay_chk.toggled.connect(self._on_show_overlay_toggled)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        self.quality_combo.currentTextChanged.connect(self._on_quality_combo_changed)

        # Zoom & maximize connections
        self.zoom_in_btn.clicked.connect(self.image_viewer.zoom_in)
        self.zoom_out_btn.clicked.connect(self.image_viewer.zoom_out)
        self.zoom_fit_btn.clicked.connect(self.image_viewer.fit_screen)
        self.maximize_btn.toggled.connect(self._toggle_viewer_maximize)

        # Viewer Display & Calibration connections
        self.viewer_combo.currentIndexChanged.connect(self._on_viewer_channel_changed)
        self.calib_combo.currentIndexChanged.connect(self._on_calib_mode_changed)

        # Segmentation controls connections
        self.method_combo.currentTextChanged.connect(self._on_method_combo_changed)
        self.wf_combo.currentIndexChanged.connect(self._on_workflow_combo_changed)
        self.model_combo.currentTextChanged.connect(self._on_model_combo_changed)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_combo_changed)
        state.segmentation_model_changed.connect(self._on_state_segmentation_model_changed)
        
        # Handle state triggers
        state.image_loaded.connect(self._on_image_loaded)
        state.workflow_selected.connect(self._on_workflow_selected)
        state.theme_changed.connect(self._sync_theme)
        state.page_changed.connect(self._on_page_changed)
        state.channel_names_changed.connect(self._sync_viewer_channel_combo)
        
        # Connect Phase 2D transient state change listener slots
        state.show_original_changed.connect(self._on_state_show_original_changed)
        state.show_overlay_changed.connect(self._on_state_show_overlay_changed)
        state.mask_opacity_changed.connect(self._on_state_mask_opacity_changed)
        state.quality_mode_changed.connect(self._on_state_quality_mode_changed)

        # Connect Calibration mode state changes
        state.calibration_mode_changed.connect(self._on_state_calibration_mode_changed)

        # Connect Segmentation state change listener slots
        state.segmentation_method_changed.connect(self._on_state_segmentation_method_changed)
        state.segmentation_channel_changed.connect(self._on_state_segmentation_channel_changed)
        
        # Invalidate loaded image cache on analysis results changes
        state.analysis_completed.connect(self._invalidate_analysis_cache)
        state.analysis_results_updated.connect(self._invalidate_analysis_cache)
        
        # Handle dirty state transitions
        state.dirty_state_changed.connect(self._on_dirty_state_changed)
        state.calibration_mode_changed.connect(self._on_calibration_mode_changed)
        self.save_analysis_btn.clicked.connect(self._on_save_clicked)
        
        # Initial boot check
        self._sync_state()

    def _save_layout_state(self, pos=0, index=0):
        settings = QSettings("Lumen", "AnalysisPage")
        settings.setValue("main_splitter_sizes", self.main_splitter.sizes())
        settings.setValue("left_collapsed", self.main_splitter.sizes()[0] == 0)
        settings.setValue("right_collapsed", self.main_splitter.sizes()[2] == 0)
        settings.setValue("viewer_maximized", self.maximize_btn.isChecked())
        settings.setValue("last_selected_workflow", self.wf_combo.currentData())

    def _restore_layout_state(self):
        settings = QSettings("Lumen", "AnalysisPage")
        
        # Splitter sizes
        sizes = settings.value("main_splitter_sizes")
        if sizes:
            try:
                int_sizes = [int(s) for s in sizes]
                self.main_splitter.setSizes(int_sizes)
            except Exception as e:
                logger.error("Error restoring main_splitter_sizes: %s", e)
                self.main_splitter.setSizes([260, 680, 340])
        else:
            self.main_splitter.setSizes([260, 680, 340])

        # Collapsed states
        left_collapsed = settings.value("left_collapsed", type=bool)
        right_collapsed = settings.value("right_collapsed", type=bool)
        # Apply collapsed states
        sizes = self.main_splitter.sizes()
        if left_collapsed and len(sizes) > 0:
            sizes[0] = 0
        if right_collapsed and len(sizes) > 2:
            sizes[2] = 0
        self.main_splitter.setSizes(sizes)

        # Maximize viewer state
        viewer_maximized = settings.value("viewer_maximized", type=bool)
        if viewer_maximized:
            self.maximize_btn.setChecked(True)
            self._toggle_viewer_maximize(True)

        # Workflow
        last_wf = settings.value("last_selected_workflow")
        if last_wf:
            idx = self.wf_combo.findData(last_wf)
            if idx >= 0:
                self.wf_combo.setCurrentIndex(idx)

    @Slot(bool)
    def _toggle_viewer_maximize(self, checked: bool):
        if checked:
            # Save pre-maximized sizes
            self._pre_maximized_sizes = self.main_splitter.sizes()
            # Set left and right sizes to 0
            self.main_splitter.setSizes([0, self.width(), 0])
        else:
            if hasattr(self, "_pre_maximized_sizes"):
                self.main_splitter.setSizes(self._pre_maximized_sizes)
            else:
                self.main_splitter.setSizes([260, 680, 340])
        self._save_layout_state()

    @Slot(int)
    def _on_viewer_channel_changed(self, combo_idx: int):
        val = self.viewer_combo.currentData()
        if val is not None:
            logger.info("AnalysisPage: Switching visible display channel selection to index %s", val)
            state.active_viewer_channel = val
            image_manager.set_active_channel(val)
            
            # Broadcast image reload to repaint GraphicsView
            state.image_loaded.emit(state.current_image_path)

    @Slot(int)
    def _on_calib_mode_changed(self, combo_idx: int):
        val = self.calib_combo.currentData()
        if val is not None:
            logger.info("AnalysisPage: Switching calibration mode selection to %s", val)
            state.calibration_mode = val

    @Slot(str)
    def _on_state_calibration_mode_changed(self, mode: str):
        self.calib_combo.blockSignals(True)
        calib_idx = self.calib_combo.findData(mode)
        if calib_idx >= 0:
            self.calib_combo.setCurrentIndex(calib_idx)
        self.calib_combo.blockSignals(False)

    def _sync_viewer_channel_combo(self):
        meta = image_manager.get_metadata()
        if not meta:
            return
        channels_count = meta.get("channels", 1)
        
        self.viewer_combo.blockSignals(True)
        self.viewer_combo.clear()
        
        names = state.channel_names
        if not names or len(names) != channels_count:
            from lumen.core.fluorescence.channels import get_default_channel_names
            names = get_default_channel_names(channels_count, meta.get("filename", ""))
            state.channel_names = names
            
        if channels_count > 1:
            self.viewer_combo.addItem("Composite (All Merged)", -1)
            for idx, name in enumerate(names):
                self.viewer_combo.addItem(f"Channel {idx}: {name}", idx)
        else:
            self.viewer_combo.addItem("Channel 0: Grayscale", 0)
            
        viewer_idx = self.viewer_combo.findData(state.active_viewer_channel)
        if viewer_idx >= 0:
            self.viewer_combo.setCurrentIndex(viewer_idx)
        else:
            self.viewer_combo.setCurrentIndex(0)
            
        self.viewer_combo.blockSignals(False)
        self._sync_channel_selector()

    def clear_selection(self):
        """Resets highlight overlay in image viewer and clears active tooltip."""
        if hasattr(self, 'image_viewer') and self.image_viewer:
            self.image_viewer.clear_highlight()
        from PySide6.QtWidgets import QToolTip
        QToolTip.hideText()

    def force_layout_refresh(self):
        """Forces a recursive layout invalidation and geometry refresh on this page."""
        lay = self.layout
        layout = lay() if callable(lay) else lay
        if layout:
            layout.invalidate()
            layout.activate()
        from PySide6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            child.updateGeometry()
            child_lay = child.layout
            child_layout = child_lay() if callable(child_lay) else child_lay
            if child_layout:
                child_layout.invalidate()
                child_layout.activate()
        self.updateGeometry()
        self.update()

    def _sync_state(self):
        """Initial check for loaded state."""
        self.clear_selection()
        image_path = state.current_image_path
        if not image_path:
            self._on_image_loaded("")
            return
            
        # Update dynamic save button text based on immutably tagged session origin
        if state.current_origin_type == "batch":
            self.save_analysis_btn.setText("💾 Save to Batch")
        else:
            self.save_analysis_btn.setText("💾 Save Analysis")
            
        is_dirty = state.is_dirty
        self.save_analysis_btn.setEnabled(is_dirty)
        self.reset_changes_btn.setEnabled(is_dirty)
            
        session = state.workspace_manager.get_analysis_session(image_path)
        logger.warning(
            "TIMELINE [4. Before restoring Analysis]: state.current_workflow=%s, session.current_workflow=%s",
            state.current_workflow,
            session.current_workflow if session else None
        )
        if session:
            # Check if session values match current state and page is already loaded
            state_matches_session = (
                state.quality_mode == session.quality_mode and
                state.mask_opacity == session.mask_opacity and
                state.show_original_image == session.show_original_image and
                state.show_segmentation_overlay == session.show_segmentation_overlay and
                state.segmentation_method == session.segmentation_method and
                state._current_workflow == session.current_workflow and
                state.analysis_results is session.analysis_results
            )
            
            # Also check if the page's widgets match the session
            # (Just in case the widgets were modified or the test reset them)
            widgets_match_session = (
                self.quality_combo.currentText() == session.quality_mode and
                self.opacity_slider.value() == session.mask_opacity and
                self.show_original_chk.isChecked() == session.show_original_image and
                self.show_overlay_chk.isChecked() == session.show_segmentation_overlay
            )
            
            if (hasattr(self, "_loaded_image_path") and 
                self._loaded_image_path == image_path and 
                hasattr(self, "_loaded_image_origin") and
                self._loaded_image_origin == state.current_origin_type and
                state_matches_session and 
                widgets_match_session):
                logger.info("AnalysisPage: Image path %s already loaded and matches session. Skipping sync.", image_path)
                return

            logger.info("AnalysisPage: Restoring from persistent session state.")
            state.quality_mode = session.quality_mode
            state.mask_opacity = session.mask_opacity
            state.show_original_image = session.show_original_image
            state.show_segmentation_overlay = session.show_segmentation_overlay
            state.segmentation_method = session.segmentation_method
            state.current_workflow = session.current_workflow
            state.analysis_results = session.analysis_results
            state.calibration_mode = getattr(session, "calibration_mode", "pixel")
            
            self._restore_from_session(session)
        else:
            if (hasattr(self, "_loaded_image_path") and 
                self._loaded_image_path == image_path and 
                hasattr(self, "_loaded_image_origin") and
                self._loaded_image_origin == state.current_origin_type and
                self.image_viewer._analysis_results is state.analysis_results):
                logger.info("AnalysisPage: Image path %s already loaded (no session). Skipping sync.", image_path)
                return

            state.workspace_manager.start_analysis_session(image_path)
            self._on_image_loaded(image_path)
            if state.current_workflow:
                self._on_workflow_selected(state.current_workflow)
                
            # Sync Phase 2D state properties
            self._on_state_show_original_changed(state.show_original_image)
            self._on_state_show_overlay_changed(state.show_segmentation_overlay)
            self._on_state_mask_opacity_changed(state.mask_opacity)
            self._on_state_quality_mode_changed(state.quality_mode)
            self._on_state_segmentation_method_changed(state.segmentation_method)
            self._sync_channel_selector()

    def _sync_channel_selector(self):
        """Populates and displays the segmentation channel selector based on metadata."""
        meta = image_manager.get_metadata()
        if meta:
            channels_count = meta.get("channels", 1)
            if channels_count > 1:
                self.channel_combo.blockSignals(True)
                self.channel_combo.clear()
                
                names = state.channel_names
                if not names or len(names) != channels_count:
                    from lumen.core.fluorescence.channels import get_default_channel_names
                    names = get_default_channel_names(channels_count, meta.get("filename", ""))
                    state.channel_names = names
                
                for idx, name in enumerate(names):
                    self.channel_combo.addItem(f"Channel {idx}: {name}", idx)
                
                seg_idx = self.channel_combo.findData(state.segmentation_channel)
                if seg_idx >= 0:
                    self.channel_combo.setCurrentIndex(seg_idx)
                else:
                    self.channel_combo.setCurrentIndex(0)
                
                self.channel_combo.blockSignals(False)
                self.channel_frame.show()
            else:
                self.channel_frame.hide()
        else:
            self.channel_frame.hide()

    def _save_to_session(self):
        image_path = state.current_image_path
        if not image_path:
            return
            
        session = state.workspace_manager.start_analysis_session(
            image_path,
            origin_type=state.current_origin_type,
            batch_origin_context=state.current_batch_origin_context
        )
        session.analysis_results = state.analysis_results
        # Only update committed_results if not dirty
        if not session.dirty:
            session.committed_results = state.analysis_results
            session.committed_fluorescence_results = state.fluorescence_results
            session.committed_fluorescence_summary = state.fluorescence_summary
        session.quality_mode = state.quality_mode
        session.mask_opacity = state.mask_opacity
        session.show_original_image = state.show_original_image
        session.show_segmentation_overlay = state.show_segmentation_overlay
        session.segmentation_method = state.segmentation_method
        session.current_workflow = state.current_workflow
        session.calibration_mode = state.calibration_mode
        
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
        logger.info("AnalysisPage: Saved session state to workspace manager.")

    def _restore_from_session(self, session):
        logger.warning(
            "TIMELINE [5. Inside _restore_from_session - Before]: session.current_workflow=%s, state.current_workflow=%s",
            session.current_workflow,
            state.current_workflow
        )
        path = session.image_path
        if path:
            path = path.replace('\\', '/')
        if path and os.path.exists(path):
            if image_manager._current_path != path:
                image_manager.load_image(path)
                
            pixmap = image_manager.get_qpixmap()
            if pixmap:
                self.image_viewer.set_image(pixmap, restore_state=session.viewer_state)
                self.viewer_contrast_lbl.setVisible(True)
                
                if session.analysis_results:
                    self.image_viewer.set_analysis_results(session.analysis_results)
                    masks = session.analysis_results.get("masks")
                    if masks is not None:
                        self.image_viewer.set_masks(masks)
                        self.image_viewer.set_show_original(session.show_original_image)
                        self.image_viewer.set_show_overlay(session.show_segmentation_overlay)
                        self.image_viewer.set_mask_opacity(session.mask_opacity)
                        self.edit_btn.setEnabled(True)
                    else:
                        self.edit_btn.setEnabled(False)
                else:
                    self.edit_btn.setEnabled(False)
                
                meta = image_manager.get_metadata()
                if meta:
                    self.fn_val.setText(meta.get("filename", "-"))
                    self.res_val.setText(f"{meta.get('width')} × {meta.get('height')}")
                    self.ch_val.setText(str(meta.get("channels", "-")))
                    self.mode_val.setText(meta.get("mode", "-").upper())
                    self.type_val.setText(meta.get("classification", "-"))

                    voxel = meta.get("voxel_size")
                    if voxel and isinstance(voxel, (list, tuple)) and len(voxel) >= 2:
                        self.voxel_val.setText(f"{voxel[0]:.4f} {meta.get('physical_units', 'µm')}")
                    else:
                        self.voxel_val.setText("1.0000 px (Uncalibrated)")

                    self.meta_placeholder.setVisible(False)
                    self.meta_container.setVisible(True)
                    
                    self._sync_viewer_channel_combo()
                    
                    # Collapsible Naming Section visibility
                    channels_count = meta.get("channels", 1)
                    self.naming_section.setVisible(channels_count > 1)

                    # Update Footer Status Bar
                    self.footer_res.setText(f"Size: {meta.get('width')} × {meta.get('height')}")
                    self.footer_zoom.setText(f"Zoom: {int(self.image_viewer.transform().m11() * 100)}%")

                self.run_btn.setEnabled(True)
                self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
                
                if state.current_workflow:
                    self._on_workflow_selected(state.current_workflow)
                elif session.current_workflow:
                    self._on_workflow_selected(session.current_workflow)
                    
                # Synchronize Preprocessing Panel sliders with state
                if hasattr(self, "preprocess_panel"):
                    self.preprocess_panel.sync_from_state()
                    
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentText(state.segmentation_model)
                self.model_combo.blockSignals(False)
                    
                self._loaded_image_path = path
                self._loaded_image_origin = state.current_origin_type
                self.force_layout_refresh()
        logger.warning(
            "TIMELINE [5. Inside _restore_from_session - After]: session.current_workflow=%s, state.current_workflow=%s",
            session.current_workflow,
            state.current_workflow
        )

    # Slots to update state from controls
    def _on_show_original_toggled(self, checked: bool):
        state.show_original_image = checked
        self._save_to_session()

    def _on_show_overlay_toggled(self, checked: bool):
        state.show_segmentation_overlay = checked
        self._save_to_session()

    def _on_opacity_slider_changed(self, val: int):
        state.mask_opacity = val
        self._save_to_session()

    def _on_quality_combo_changed(self, text: str):
        state.quality_mode = text
        self._save_to_session()

    def _on_method_combo_changed(self, text: str):
        if text == "AI Segmentation (Cellpose)":
            state.segmentation_method = "AI Segmentation"
        else:
            state.segmentation_method = text
        self._save_to_session()

    def _on_channel_combo_changed(self, index: int):
        val = self.channel_combo.currentData()
        if val is not None:
            logger.info("AnalysisPage: Setting target segmentation channel selection to index %s", val)
            state.segmentation_channel = val
            self._save_to_session()

    # Slots to update UI controls from state changes
    @Slot(bool)
    def _on_state_show_original_changed(self, val: bool):
        self.show_original_chk.blockSignals(True)
        self.show_original_chk.setChecked(val)
        self.show_original_chk.blockSignals(False)
        self.image_viewer.set_show_original(val)

    @Slot(bool)
    def _on_state_show_overlay_changed(self, val: bool):
        self.show_overlay_chk.blockSignals(True)
        self.show_overlay_chk.setChecked(val)
        self.show_overlay_chk.blockSignals(False)
        self.image_viewer.set_show_overlay(val)

    @Slot(int)
    def _on_state_mask_opacity_changed(self, val: int):
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(val)
        self.opacity_slider.blockSignals(False)
        self.opacity_val_lbl.setText(f"{val}%")
        self.image_viewer.set_mask_opacity(val)

    @Slot(str)
    def _on_state_quality_mode_changed(self, val: str):
        self.quality_combo.blockSignals(True)
        self.quality_combo.setCurrentText(val)
        self.quality_combo.blockSignals(False)

    @Slot(str)
    def _on_state_segmentation_method_changed(self, val: str):
        self.method_combo.blockSignals(True)
        if val == "AI Segmentation":
            self.method_combo.setCurrentText("AI Segmentation (Cellpose)")
        else:
            self.method_combo.setCurrentText(val)
        self.method_combo.blockSignals(False)
        self.force_layout_refresh()

    @Slot(int)
    def _on_state_segmentation_channel_changed(self, val: int):
        self.channel_combo.blockSignals(True)
        idx = self.channel_combo.findData(val)
        if idx >= 0:
            self.channel_combo.setCurrentIndex(idx)
        self.channel_combo.blockSignals(False)

    @Slot(str)
    def _on_image_loaded(self, path: str):
        if path:
            path = path.replace('\\', '/')
        if path and os.path.exists(path):
            if image_manager._current_path != path:
                image_manager.load_image(path)
                
            pixmap = image_manager.get_qpixmap()
            if pixmap:
                self.image_viewer.set_image(pixmap)
                self.viewer_contrast_lbl.setVisible(True)
                
                # Render the mask overlay in the image viewer if existing results are in the state
                if state.analysis_results:
                    self.image_viewer.set_analysis_results(state.analysis_results)
                    masks = state.analysis_results.get("masks")
                    if masks is not None:
                        self.image_viewer.set_masks(masks)
                        self.image_viewer.set_show_original(state.show_original_image)
                        self.image_viewer.set_show_overlay(state.show_segmentation_overlay)
                        self.image_viewer.set_mask_opacity(state.mask_opacity)
                        self.edit_btn.setEnabled(True)
                    else:
                        self.edit_btn.setEnabled(False)
                else:
                    self.edit_btn.setEnabled(False)
                
                # Extract simplified metadata fields
                meta = image_manager.get_metadata()
                if meta:
                    self.fn_val.setText(meta.get("filename", "-"))
                    self.res_val.setText(f"{meta.get('width')} × {meta.get('height')}")
                    self.ch_val.setText(str(meta.get("channels", "-")))
                    self.mode_val.setText(meta.get("mode", "-").upper())
                    self.type_val.setText(meta.get("classification", "-"))

                    voxel = meta.get("voxel_size")
                    if voxel and isinstance(voxel, (list, tuple)) and len(voxel) >= 2:
                        self.voxel_val.setText(f"{voxel[0]:.4f} {meta.get('physical_units', 'µm')}")
                    else:
                        self.voxel_val.setText("1.0000 px (Uncalibrated)")

                    # Swap visibility
                    self.meta_placeholder.setVisible(False)
                    self.meta_container.setVisible(True)
                    
                    self._sync_viewer_channel_combo()
                    
                    # Collapsible Naming Section visibility
                    channels_count = meta.get("channels", 1)
                    self.naming_section.setVisible(channels_count > 1)

                    # Update Footer Status Bar
                    self.footer_res.setText(f"Size: {meta.get('width')} × {meta.get('height')}")
                    self.footer_zoom.setText(f"Zoom: {int(self.image_viewer.transform().m11() * 100)}%")

                self.run_btn.setEnabled(True)
                self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
                self._loaded_image_path = path
                self._loaded_image_origin = state.current_origin_type
                self.force_layout_refresh()
                return
        
        # Clear/empty state
        self._loaded_image_path = None
        self._loaded_image_origin = None
        self.image_viewer.clear()
        self.naming_section.hide()
        self.puncta_section.hide()
        self.footer_res.setText("-")
        self.footer_zoom.setText("Zoom: 100%")
        
        self.force_layout_refresh()
        self.image_viewer.set_analysis_results(None)
        self.viewer_contrast_lbl.setVisible(False)
        self.meta_placeholder.setVisible(True)
        self.meta_container.setVisible(False)
        self.edit_btn.setEnabled(False)
        
        self.run_btn.setEnabled(False)
        self.run_btn.setCursor(QCursor(Qt.ArrowCursor))

    @Slot(str)
    def _on_workflow_selected(self, wf_id: str):
        logger.warning(
            "TIMELINE [6. Inside _on_workflow_selected]: argument wf_id=%s, sender=%s, current combobox value=%s",
            wf_id,
            self.sender(),
            self.wf_combo.currentData()
        )
        # Update combobox selection silently
        self.wf_combo.blockSignals(True)
        idx = self.wf_combo.findData(wf_id)
        if idx >= 0:
            self.wf_combo.setCurrentIndex(idx)
        self.wf_combo.blockSignals(False)

        meta = image_manager.get_metadata()
        channels_count = meta.get("channels", 1) if meta else 1

        # Show/hide workflow-specific collapsible sections
        if wf_id == "fluorescence":
            self.naming_section.setVisible(channels_count > 1)
            self.puncta_section.setVisible(False)
        elif wf_id == "puncta":
            self.naming_section.setVisible(False)
            self.puncta_section.setVisible(True)
        else:
            self.naming_section.setVisible(False)
            self.puncta_section.setVisible(False)

        self.force_layout_refresh()

    def _on_workflow_combo_changed(self, index: int):
        wf_id = self.wf_combo.itemData(index)
        if not wf_id:
            return
        logger.info("AnalysisPage: Switching workflow mode in workspace to %s", wf_id)
        state.current_workflow = wf_id

    def _on_model_combo_changed(self, text: str):
        state.segmentation_model = text

    @Slot(str)
    def _on_state_segmentation_model_changed(self, val: str):
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(val)
        self.model_combo.blockSignals(False)

    def _run_fluorescence_quantification(self, results_dict: dict = None):
        """Runs the standalone fluorescence quantification on the active image and mask."""
        if state.current_workflow != "fluorescence":
            return
            
        res = results_dict if results_dict is not None else state.analysis_results
        masks = res.get("masks") if res else None
        if masks is None:
            logger.info("AnalysisPage: No mask available for fluorescence quantification.")
            state.fluorescence_results = []
            state.fluorescence_summary = {}
            return
            
        from lumen.processing.image_manager import image_manager
        raw_channels = image_manager._raw_channels
        if not raw_channels:
            logger.info("AnalysisPage: No raw channels loaded for fluorescence quantification.")
            state.fluorescence_results = []
            state.fluorescence_summary = {}
            return
            
        # Get channel names
        channel_names = state.channel_names
        if not channel_names or len(channel_names) != len(raw_channels):
            channel_names = image_manager._channel_names
            
        from lumen.core.fluorescence.quantifier import quantify_fluorescence
        try:
            metadata = image_manager.get_metadata()
            voxel_size = metadata.get("voxel_size", (1.0, 1.0, 1.0))
            calibration_mode = state.calibration_mode

            results = quantify_fluorescence(
                raw_channels=raw_channels,
                masks=masks,
                channel_names=channel_names,
                voxel_size=voxel_size,
                calibration_mode=calibration_mode
            )
            state.fluorescence_results = results
            
            # Calculate and store summary stats
            if results:
                # Infer channels from results[0]
                channels = []
                for key in results[0].keys():
                    if key.endswith("_mean"):
                        channels.append(key[:-5])
                        
                avg_area = float(np.mean([r["area"] for r in results]))
                summary = {
                    "total_cell_count": len(results),
                    "average_area": avg_area,
                }
                for ch in channels:
                    mean_key = f"{ch}_mean"
                    median_key = f"{ch}_median"
                    mean_vals = [r[mean_key] for r in results if mean_key in r]
                    median_vals = [r[median_key] for r in results if median_key in r]
                    summary[f"{ch}_mean_average"] = float(np.mean(mean_vals)) if mean_vals else 0.0
                    summary[f"{ch}_median_average"] = float(np.mean(median_vals)) if median_vals else 0.0
                state.fluorescence_summary = summary
            else:
                state.fluorescence_summary = {
                    "total_cell_count": 0,
                    "average_area": 0.0
                }
            logger.info("AnalysisPage: Fluorescence quantification completed successfully.")
            
        except ValueError as e:
            logger.error("AnalysisPage: Fluorescence quantification failed (shape mismatch): %s", e)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Quantification Error",
                f"Fluorescence quantification failed:\n{str(e)}",
                QMessageBox.Ok
            )
            state.fluorescence_results = []
            state.fluorescence_summary = {}
            raise e
        except Exception as e:
            logger.error("AnalysisPage: Fluorescence quantification failed: %s", e, exc_info=True)
            state.fluorescence_results = []
            state.fluorescence_summary = {}

    def _set_controls_enabled(self, enabled: bool):
        self.method_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.quality_combo.setEnabled(enabled)

    def _on_run_analysis_clicked(self):
        """Launches actual segmentation analysis pipeline asynchronously."""
        image_path = state.current_image_path
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please load a microscopy image before running analysis.",
                QMessageBox.Ok
            )
            return

        # Guard: Check image dimensionality and channel boundaries if workflow is fluorescence
        if state.current_workflow == "fluorescence":
            from lumen.processing.image_manager import image_manager
            raw_arr = image_manager._raw_numpy_arr
            if raw_arr is None:
                QMessageBox.critical(
                    self,
                    "No Image Data",
                    "No image data is loaded in the workspace.",
                    QMessageBox.Ok
                )
                return

            available_channels = raw_arr.shape[2] if raw_arr.ndim == 3 else 1
            seg_channel = state.segmentation_channel
            if not isinstance(seg_channel, int) or not (0 <= seg_channel < available_channels):
                QMessageBox.critical(
                    self,
                    "Invalid Channel Selected",
                    f"Selected segmentation channel {seg_channel} is out of bounds for the image with {available_channels} channels.",
                    QMessageBox.Ok
                )
                return

        # Disable buttons/combo to prevent double execution and prepare progress display
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Analyzing...")
        self._set_controls_enabled(False)
        
        self.status_lbl.setText("Starting analysis pipeline...")
        self.status_lbl.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Show loading overlay
        is_gpu, _ = gpu_service.resolve_execution_backend("Use Global Setting")
        self.image_viewer.show_loading("Starting analysis pipeline...", gpu_active=is_gpu)
        self.footer_status.setText("Starting analysis...")
        
        self.force_layout_refresh()

        logger.info(
            "AnalysisPage: Initiating %s pipeline for image: %s",
            state.segmentation_method, image_path
        )

        callbacks = {
            "progress": self._on_analysis_progress,
            "status": self._on_analysis_status,
            "finished": self._on_analysis_finished,
            "failed": self._on_analysis_failed
        }

        # Build parameters dictionary containing selected settings
        model_override = state.segmentation_model.lower()
        if model_override == "auto":
            model_override = None

        parameters = {
            "segmentation_method": state.segmentation_method,
            "quality_mode": state.quality_mode,
            "model_type_override": model_override,
            "current_workflow": state.current_workflow,
            "segmentation_channel": state.segmentation_channel
        }

        success = processing_manager.run_analysis(image_path, parameters, callbacks)
        if not success:
            QMessageBox.warning(
                self,
                "Analysis In Progress",
                "An analysis is already running. Please wait for it to complete.",
                QMessageBox.Ok
            )
            # Restore UI
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Analysis")
            self._set_controls_enabled(True)
            self.status_lbl.setVisible(False)
            self.progress_bar.setVisible(False)

    @Slot(int)
    def _on_analysis_progress(self, value: int):
        self.progress_bar.setValue(value)
        self.footer_status.setText(f"Analyzing... {value}%")

    @Slot(str)
    def _on_analysis_status(self, status_msg: str):
        self.status_lbl.setText(status_msg)
        is_gpu, _ = gpu_service.resolve_execution_backend("Use Global Setting")
        self.image_viewer.show_loading(status_msg, gpu_active=is_gpu)
        self.footer_status.setText(status_msg)

    @Slot(dict)
    def _on_analysis_finished(self, results: dict):
        logger.info("AnalysisPage: Analysis finished successfully.")
        
        self.image_viewer.hide_loading()
        self.footer_status.setText("Analysis finished")
        
        # Restore button and hide progress
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_btn.setFocus() # Anchor focus back to sidebar
        self._set_controls_enabled(True)
        self.status_lbl.setVisible(False)
        self.progress_bar.setVisible(False)
        self.force_layout_refresh()
        
        if state.current_workflow == "fluorescence":
            try:
                self._run_fluorescence_quantification(results)
            except ValueError:
                pass

        # Populate results_dict into the state
        state.analysis_results = results
        self.image_viewer.set_analysis_results(results)
        
        # Render the mask overlay in the image viewer
        masks = results.get("masks")
        if masks is not None:
            self.image_viewer.set_masks(masks)
            self.edit_btn.setEnabled(True)
        else:
            self.edit_btn.setEnabled(False)

        # --- Dirty-state tracking for reanalysis ---
        # Compare new scientific results against the last committed snapshot.
        # If they differ (any origin type), mark session dirty so Save/Reset become available.
        image_path = state.current_image_path
        if image_path:
            session = state.workspace_manager.get_analysis_session(
                image_path, state.workspace_manager._active_analysis_origin
            )
            if session:
                new_cell_count = results.get("cell_count")
                committed = session.committed_results
                if committed is None:
                    # First ever result for this image — committed_results not yet set.
                    # Only mark dirty if actual masks were produced (non-trivial result).
                    if new_cell_count is not None:
                        state.is_dirty = True
                        logger.info("AnalysisPage: First analysis result produced, marking session dirty.")
                else:
                    committed_cell_count = committed.get("cell_count")
                    if new_cell_count != committed_cell_count:
                        state.is_dirty = True
                        logger.info(
                            "AnalysisPage: Reanalysis produced different results "
                            "(prev cell_count=%s, new cell_count=%s). Marking session dirty.",
                            committed_cell_count, new_cell_count
                        )
                    else:
                        logger.info(
                            "AnalysisPage: Reanalysis result matches committed snapshot (cell_count=%s). "
                            "Dirty state unchanged.", new_cell_count
                        )

        # Trigger lightweight session checkpoint save
        self._save_to_session()


    @Slot(str)
    def _on_page_changed(self, page_name: str):
        if page_name == "analysis":
            self._sync_state()
        else:
            sess = state.workspace_manager.get_analysis_session(state.current_image_path) if state.current_image_path else None
            logger.warning(
                "TIMELINE [1. Before leaving Analysis]: state.current_workflow=%s, session.current_workflow=%s",
                state.current_workflow,
                sess.current_workflow if sess else None
            )
            self._save_to_session()

    @Slot()
    def _invalidate_analysis_cache(self, *args, **kwargs):
        logger.info("AnalysisPage: Invalidating loaded image cache due to analysis results update.")
        self._loaded_image_path = None
        self._loaded_image_origin = None

    @Slot()
    def _on_edit_masks_clicked(self):
        image_path = state.current_image_path
        results = state.analysis_results
        
        if not image_path or not results or "masks" not in results:
            return
            
        original_mask = results["masks"]
        if original_mask is None:
            return
            
        from lumen.ui.mask_editor_dialog import MaskEditorDialog, update_results_mask
        from PySide6.QtWidgets import QDialog
        
        editor = MaskEditorDialog(image_path, original_mask, self)
        # Clear existing highlight in viewer before editor opens
        self.clear_selection()
        res = editor.exec()
        
        # Clear highlights again when editor closes
        self.clear_selection()
        
        if res == QDialog.Accepted:
            edited_mask = getattr(editor, "edited_mask", None)
            import numpy as np
            if not isinstance(edited_mask, np.ndarray):
                edited_mask = None
                
            if edited_mask is None and hasattr(editor, "canvas") and editor.canvas:
                wmask = editor.canvas.working_mask
                if isinstance(wmask, np.ndarray):
                    edited_mask = wmask
                
            if edited_mask is not None:
                # Update masks, cell_count, cell_metrics, areas, diameters, and densities
                edit_log = getattr(editor, "edit_operation_log", [])
                if not isinstance(edit_log, list):
                    edit_log = []
                updated_results = update_results_mask(results, edited_mask, edit_log=edit_log)
                
                # Write to state bypass property setter side-effects (avoiding analysis_completed.emit)
                state._analysis_results = updated_results
                
                # Directly update viewer
                self.image_viewer.set_analysis_results(updated_results)
                self.image_viewer.set_masks(edited_mask)
                
                # Mark dirty only if the editor actually recorded edits (Feature 3)
                if editor.has_unsaved_changes():
                    state.is_dirty = True

                if state.current_workflow == "fluorescence":
                    try:
                        self._run_fluorescence_quantification(updated_results)
                    except ValueError:
                        pass

                # Save session checkpoint
                self._save_to_session()
                
                # Emit explicit manual correction update signals
                state.analysis_results_updated.emit(updated_results)
                state.manual_mask_saved.emit(image_path)

                # ------------------------------------------------
                # Selection Synchronization (Feature 7)
                # ------------------------------------------------
                selected_cell_id = editor.canvas.selected_label_id
                if selected_cell_id is not None and selected_cell_id > 0:
                    if np.any(edited_mask == selected_cell_id):
                        self.image_viewer._highlight_cell(selected_cell_id, edited_mask)
                        
                        # Show updated tooltip at centroid of the selected cell
                        metrics_dict = updated_results.get("cell_metrics", {})
                        cell_info = metrics_dict.get(selected_cell_id)
                        if cell_info:
                            area = cell_info["area_px"]
                            diam = cell_info["diameter_px"]
                            cx, cy = cell_info["centroid"]
                            tooltip_text = (
                                f"<b>Cell ID:</b> {selected_cell_id}<br/>"
                                f"<b>Area:</b> {area} px<br/>"
                                f"<b>Diameter:</b> {diam} px<br/>"
                                f"<b>Centroid:</b> ({cx}, {cy})"
                            )
                            # Mapped scene centroid to viewport coordinate
                            view_pos = self.image_viewer.mapFromScene(cx, cy)
                            global_pos = self.image_viewer.viewport().mapToGlobal(view_pos)
                            from PySide6.QtWidgets import QToolTip
                            QToolTip.showText(global_pos, tooltip_text, self.image_viewer)
                            
    def save_analysis(self) -> bool:
        """Canonical save authority for the active analysis session."""
        from pathlib import Path
        if not state.is_dirty:
            return True
            
        image_path = state.current_image_path
        results = state.analysis_results
        if not image_path or not results or "masks" not in results:
            return False
            
        edited_mask = results["masks"]
        
        # Explicitly recompute metrics from the latest committed edited mask before saving
        from lumen.ui.mask_editor_dialog import update_results_mask
        edit_log = results.get("edit_operation_log", [])
        if not isinstance(edit_log, list):
            edit_log = []
        results = update_results_mask(results, edited_mask, edit_log=edit_log)
        state._analysis_results = results
        session = state.workspace_manager.get_analysis_session(image_path, state.current_origin_type)
        if session:
            session.analysis_results = results
            session.committed_results = results
            session.committed_fluorescence_results = state.fluorescence_results
            session.committed_fluorescence_summary = state.fluorescence_summary
        
        # Behavior depends on ORIGIN CONTEXT
        if state.current_origin_type == "batch":
            active_batch_dir = state.current_batch_origin_context
            if not active_batch_dir:
                active_batch_dir = state.batch_results_dir
            if not active_batch_dir:
                active_batch_dir = state.workspace_manager._active_batch_dir
                
            filename = os.path.basename(image_path)
            
            # Robust active batch directory lookup fallback
            if not active_batch_dir:
                for b_dir, session in state.workspace_manager._batch_sessions.items():
                    if session.records:
                        for rec in session.records:
                            if rec.get("image_name", "").lower() == filename.lower():
                                active_batch_dir = b_dir
                                break
                    if active_batch_dir:
                        break
                        
            if not active_batch_dir:
                candidate = Path(image_path).parent / "batch_results"
                if candidate.exists() and (candidate / "batch_summary.csv").exists():
                    active_batch_dir = str(candidate)
                    
            if active_batch_dir:
                img_folder = Path(active_batch_dir) / filename
                os.makedirs(img_folder, exist_ok=True)
                
                logger.info("AnalysisPage: Syncing edited mask to batch results on Save: %s", img_folder)
                labels_path = img_folder / f"{filename}_labels_raw.tif"
                csv_path = img_folder / f"{filename}_cell_metrics.csv"
                preview_path = img_folder / f"{filename}_overlay_preview.png"
                
                try:
                    import tifffile
                    import csv
                    import json
                    
                    # 1. Save raw masks
                    tifffile.imwrite(str(labels_path), edited_mask.astype(np.uint16))
                    
                    # 2. Save cell metrics CSV
                    from lumen.pages.results_page import export_cell_metrics_csv
                    export_cell_metrics_csv(str(csv_path), results["cell_metrics"])
                    
                    # 2b. Save edit operation log JSON
                    edit_log_path = img_folder / f"{filename}_edit_log.json"
                    edit_log = results.get("edit_operation_log", [])
                    with open(edit_log_path, "w", encoding="utf-8") as elf:
                        json.dump(edit_log, elf, indent=2)
                    
                    # 3. Save visual overlay preview
                    from lumen.pages.results_page import generate_overlay_image
                    generate_overlay_image(image_path, edited_mask).save(str(preview_path))
                    
                    # 3b. Save PDF report (overwriting/regenerating it with committed edited state)
                    from lumen.pages.results_page import export_pdf_report
                    pdf_path = img_folder / f"{filename}_report.pdf"
                    export_pdf_report(
                        str(pdf_path),
                        image_path,
                        state.quality_mode,
                        state.current_workflow,
                        results
                    )
                    
                    # 4. Update memory session records
                    batch_session = state.workspace_manager.get_batch_session(active_batch_dir)
                    if not batch_session:
                        batch_session = state.workspace_manager.start_batch_session(active_batch_dir)
                        
                    if batch_session:
                        # Pre-populate records if empty (e.g. fresh start)
                        if not batch_session.records:
                            summary_csv = Path(active_batch_dir) / "batch_summary.csv"
                            if summary_csv.exists():
                                with open(summary_csv, mode="r", newline="", encoding="utf-8") as f:
                                    reader = csv.DictReader(f)
                                    batch_session.records = list(reader)
                                    
                        # Pre-populate manifest if empty
                        if not batch_session.manifest_data:
                            manifest_json = Path(active_batch_dir) / "run_manifest.json"
                            if manifest_json.exists():
                                with open(manifest_json, mode="r", encoding="utf-8") as f:
                                    batch_session.manifest_data = json.load(f)
                                    
                        # Update list records
                        for rec in batch_session.records:
                            if rec.get("image_name", "").lower() == filename.lower():
                                rec["edited"] = True
                                rec["cell_count"] = str(results["cell_count"])
                                rec["mean_area_px"] = f"{results['mean_cell_area_px']:.2f}"
                                rec["median_area_px"] = f"{results['median_cell_area_px']:.2f}"
                                rec["average_diameter_px"] = f"{results['average_diameter_px']:.2f}"
                                rec["cell_density"] = f"{results['cell_density']:.2e}"
                                rec["status"] = "SUCCESS"
                                
                        # Update manifest data
                        if "images" in batch_session.manifest_data:
                            for img_rec in batch_session.manifest_data["images"]:
                                if img_rec.get("image_name", "").lower() == filename.lower():
                                    img_rec["edited"] = True
                                    img_rec["cell_count"] = results["cell_count"]
                                    img_rec["mean_area_px"] = results["mean_cell_area_px"]
                                    img_rec["median_area_px"] = results["median_cell_area_px"]
                                    img_rec["average_diameter_px"] = results["average_diameter_px"]
                                    img_rec["cell_density"] = results["cell_density"]
                                    img_rec["status"] = "SUCCESS"
                                    
                    # 5. Update batch_summary.csv on disk
                    summary_csv = Path(active_batch_dir) / "batch_summary.csv"
                    if summary_csv.exists():
                        summary_records = []
                        fields = []
                        with open(summary_csv, mode="r", newline="", encoding="utf-8") as sf:
                            reader = csv.DictReader(sf)
                            fields = list(reader.fieldnames) if reader.fieldnames else []
                            for row in reader:
                                if row.get("image_name", "").lower() == filename.lower():
                                    row["cell_count"] = str(results["cell_count"])
                                    row["mean_area_px"] = f"{results['mean_cell_area_px']:.2f}"
                                    row["median_area_px"] = f"{results['median_cell_area_px']:.2f}"
                                    row["average_diameter_px"] = f"{results['average_diameter_px']:.2f}"
                                    row["cell_density"] = f"{results['cell_density']:.2e}"
                                    row["status"] = "SUCCESS"
                                    row["edited"] = "True"
                                summary_records.append(row)
                                
                        for field_name in ["cell_count", "mean_area_px", "median_area_px", "average_diameter_px", "cell_density", "status", "edited"]:
                            if field_name not in fields:
                                fields.append(field_name)
                            
                        with open(summary_csv, mode="w", newline="", encoding="utf-8") as sf:
                            writer = csv.DictWriter(sf, fieldnames=fields)
                            writer.writeheader()
                            for row in summary_records:
                                if "edited" not in row:
                                    row["edited"] = "False"
                                writer.writerow(row)
                                
                    # 6. Update run_manifest.json on disk
                    manifest_json = Path(active_batch_dir) / "run_manifest.json"
                    if manifest_json.exists():
                        with open(manifest_json, mode="r", encoding="utf-8") as mf:
                            mdata = json.load(mf)
                        if "images" in mdata:
                            for img_rec in mdata["images"]:
                                if img_rec.get("image_name", "").lower() == filename.lower():
                                    img_rec["cell_count"] = results["cell_count"]
                                    img_rec["mean_area_px"] = results["mean_cell_area_px"]
                                    img_rec["median_area_px"] = results["median_cell_area_px"]
                                    img_rec["average_diameter_px"] = results["average_diameter_px"]
                                    img_rec["cell_density"] = results["cell_density"]
                                    img_rec["status"] = "SUCCESS"
                                    img_rec["edited"] = True
                        with open(manifest_json, mode="w", encoding="utf-8") as mf:
                            json.dump(mdata, mf, indent=2)
                            
                    # 7. Invalidate batch explorer cache to force UI refresh
                    p = self.parent()
                    main_win = None
                    while p:
                        if hasattr(p, "batch_explorer_page"):
                            main_win = p
                            break
                        p = p.parent()
                        
                    if main_win:
                        main_win.batch_explorer_page._loaded_batch_dir = None
                        
                except Exception as ex:
                    logger.error("AnalysisPage: Failed to synchronize batch records on Save: %s", ex, exc_info=True)
                    return False
                    
        state.is_dirty = False
        session = state.workspace_manager.get_analysis_session(image_path)
        if session:
            session.committed_results = results
            session.committed_fluorescence_results = state.fluorescence_results
            session.committed_fluorescence_summary = state.fluorescence_summary
        return True

    @Slot()
    def _on_save_clicked(self):
        success = self.save_analysis()
        if success:
            from PySide6.QtWidgets import QMessageBox
            if state.current_origin_type == "batch":
                QMessageBox.information(self, "Save Complete", "Manual mask corrections committed to Batch Results successfully!")
            else:
                QMessageBox.information(self, "Save Complete", "Analysis state saved successfully!")

    @Slot()
    def _on_reset_changes_clicked(self):
        import sys
        is_testing = "unittest" in sys.modules or "pytest" in sys.modules
        if not is_testing:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Reset Changes",
                "Are you sure you want to discard all uncommitted modifications and restore the last committed state?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        self.reset_analysis_changes()

    def reset_analysis_changes(self) -> bool:
        """Restores the active session to the last committed state."""
        self.clear_selection()
        image_path = state.current_image_path
        if not image_path:
            return False
            
        session = state.workspace_manager.get_analysis_session(image_path)
        if not session:
            return False
            
        # Revert in-memory results to committed_results
        committed = session.committed_results
        committed_fluor = getattr(session, "committed_fluorescence_results", {})
        committed_summary = getattr(session, "committed_fluorescence_summary", {})
        session.fluorescence_results = committed_fluor
        session.fluorescence_summary = committed_summary
        state.fluorescence_results = committed_fluor
        state.fluorescence_summary = committed_summary

        state.analysis_results = committed
        session.analysis_results = committed
        
        # Directly update viewer
        self.image_viewer.set_analysis_results(committed)
        if committed and "masks" in committed:
            masks = committed["masks"]
            self.image_viewer.set_masks(masks)
            self.edit_btn.setEnabled(True)
        else:
            self.image_viewer.set_masks(None)
            self.edit_btn.setEnabled(False)
            
        state.is_dirty = False
        self._save_to_session()
        
        # Emit explicit update signals to sync results cards
        state.analysis_results_updated.emit(committed or {})
        state.manual_mask_saved.emit(image_path)
        return True

    @Slot(str)
    def _on_calibration_mode_changed(self, mode: str):
        logger.info("AnalysisPage: Calibration mode changed to %s. Re-running quantification.", mode)
        if state.current_workflow == "fluorescence":
            self._run_fluorescence_quantification()

    @Slot(bool)
    def _on_dirty_state_changed(self, is_dirty: bool):
        self.save_analysis_btn.setEnabled(is_dirty)
        self.reset_changes_btn.setEnabled(is_dirty)
        if is_dirty:
            self.dirty_lbl.setText("⚠️ Unsaved Changes")
            self.dirty_lbl.setVisible(True)
        else:
            self.dirty_lbl.setVisible(False)

    @Slot(str)
    def _on_analysis_failed(self, error_msg: str):
        logger.error("AnalysisPage: Analysis failed: %s", error_msg)
        
        self.image_viewer.hide_loading()
        self.footer_status.setText("Analysis failed")
        
        # Restore button and hide progress
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._set_controls_enabled(True)
        self.status_lbl.setVisible(False)
        self.progress_bar.setVisible(False)
        self.force_layout_refresh()
        
        QMessageBox.critical(
            self,
            "Analysis Failed",
            f"An error occurred during analysis:\n\n{error_msg}",
            QMessageBox.Ok
        )

    @Slot(str)
    def _sync_theme(self, theme_name: str = ""):
        theme = theme_name if theme_name else theme_service.current_theme
        self.image_viewer.sync_theme(theme)
        if hasattr(self, 'workspace_switcher'):
            self.workspace_switcher.sync_theme(theme)
            
        if theme == "light":
            # 1. Main Splitter handle light mode styling
            self.main_splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #E5E7EB;
                    width: 5px;
                }
                QSplitter::handle:hover {
                    background-color: #4F46E5;
                }
            """)
            
            # 2. Controls bar light mode
            self.viewer_controls_bar.setStyleSheet("""
                #ViewerControlsBar {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                }
            """)
            self.show_original_chk.setStyleSheet("font-size: 11px; color: #1F2937;")
            self.show_overlay_chk.setStyleSheet("font-size: 11px; color: #1F2937;")
            self.opacity_val_lbl.setStyleSheet("font-size: 11px; color: #1F2937;")
            self.viewer_contrast_lbl.setStyleSheet("font-size: 10px; color: #4F46E5; font-weight: bold; background: transparent;")

            # 3. Left Panel light mode
            self.left_panel.setStyleSheet("""
                #AnalysisLeftPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                }
            """)
            
            # 4. Right Panel light mode
            self.right_panel.setStyleSheet("""
                #AnalysisRightPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                }
            """)

            # 5. Metadata values light mode
            self.fn_val.setStyleSheet("color: #111827; font-weight: bold; font-size: 12px;")
            self.res_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.ch_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.mode_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.voxel_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.type_val.setStyleSheet("color: #4F46E5; font-size: 12px; font-weight: bold;")

            # 6. Action buttons light mode
            btn_style = """
                QPushButton {
                    padding: 8px;
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    color: #4B5563;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #F3F4F6; }
                QPushButton:disabled {
                    background-color: #F9FAFB;
                    border: 1px solid #E5E7EB;
                    color: #9CA3AF;
                }
            """
            self.edit_btn.setStyleSheet(btn_style)
            self.save_analysis_btn.setStyleSheet(btn_style)
            self.reset_changes_btn.setStyleSheet(btn_style)

            self.run_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4F46E5;
                    color: #FFFFFF;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #4338CA;
                }
                QPushButton:disabled {
                    background-color: #E5E7EB;
                    color: #9CA3AF;
                }
            """)

            self.status_lbl.setStyleSheet("font-size: 11px; color: #4B5563; margin-bottom: 4px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #E5E7EB;
                    border: 1px solid #D1D5DB;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background-color: #4F46E5;
                    border-radius: 3px;
                }
            """)
            
            # Footer styling light mode
            self.footer_bar.setStyleSheet("""
                #AnalysisFooterBar {
                    background-color: #F3F4F6;
                    border-top: 1px solid #D1D5DB;
                }
            """)
            self.footer_status.setStyleSheet("font-size: 11px; color: #4B5563;")
            self.footer_gpu.setStyleSheet("font-size: 11px; color: #4B5563;")
            self.footer_res.setStyleSheet("font-size: 11px; color: #4B5563;")
            self.footer_zoom.setStyleSheet("font-size: 11px; color: #4B5563;")

        else:
            # 1. Main Splitter handle dark mode styling
            self.main_splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #2B2B35;
                    width: 5px;
                }
                QSplitter::handle:hover {
                    background-color: #6366F1;
                }
            """)
            
            # 2. Controls bar dark mode
            self.viewer_controls_bar.setStyleSheet("""
                #ViewerControlsBar {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                }
            """)
            self.show_original_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")
            self.show_overlay_chk.setStyleSheet("font-size: 11px; color: #E5E7EB;")
            self.opacity_val_lbl.setStyleSheet("font-size: 11px; color: #E5E7EB;")
            self.viewer_contrast_lbl.setStyleSheet("font-size: 10px; color: #818CF8; font-weight: bold; background: transparent;")

            # 3. Left Panel dark mode
            self.left_panel.setStyleSheet("""
                #AnalysisLeftPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                }
            """)
            
            # 4. Right Panel dark mode
            self.right_panel.setStyleSheet("""
                #AnalysisRightPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                }
            """)

            # 5. Metadata values dark mode
            self.fn_val.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 12px;")
            self.res_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.ch_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.mode_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.voxel_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.type_val.setStyleSheet("color: #6366F1; font-size: 12px; font-weight: bold;")

            # 6. Action buttons dark mode
            btn_style = """
                QPushButton {
                    padding: 8px;
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    color: #D1D5DB;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #2D2D37; color: #FFFFFF; }
                QPushButton:disabled {
                    background-color: #16161A;
                    border: 1px solid #222227;
                    color: #4B5563;
                }
            """
            self.edit_btn.setStyleSheet(btn_style)
            self.save_analysis_btn.setStyleSheet(btn_style)
            self.reset_changes_btn.setStyleSheet(btn_style)

            self.run_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6366F1;
                    color: #FFFFFF;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #4F46E5;
                }
                QPushButton:disabled {
                    background-color: #1F1F24;
                    color: #4B5563;
                }
            """)

            self.status_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF; margin-bottom: 4px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background-color: #6366F1;
                    border-radius: 3px;
                }
            """)
            
            # Footer styling dark mode
            self.footer_bar.setStyleSheet("""
                #AnalysisFooterBar {
                    background-color: #0E0E12;
                    border-top: 1px solid #2B2B35;
                }
            """)
            self.footer_status.setStyleSheet("font-size: 11px; color: #9EA4B0;")
            self.footer_gpu.setStyleSheet("font-size: 11px; color: #9EA4B0;")
            self.footer_res.setStyleSheet("font-size: 11px; color: #9EA4B0;")
            self.footer_zoom.setStyleSheet("font-size: 11px; color: #9EA4B0;")
