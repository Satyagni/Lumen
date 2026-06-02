import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QSizePolicy, QGridLayout, QProgressBar, QComboBox, QCheckBox, QSlider, QScrollArea
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QPainter, QCursor
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager
from lumen.processing.processing_manager import processing_manager
from lumen.workflows.workflow_manager import workflow_manager
from lumen.core.services.theme_service import theme_service

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
        desc_lbl = QLabel("Upload a microscopy image to begin biological analysis.")
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
        self.setStyleSheet("background-color: #0B0B0D; border: 1px solid #2B2B35; border-radius: 6px;")

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

    def set_image(self, pixmap: QPixmap):
        """Sets canvas image and resets viewport zoom."""
        self.clear_highlight()
        if pixmap and not pixmap.isNull():
            self._placeholder.hide()
            self.pixmap_item.setPixmap(pixmap)
            # Clear existing mask overlay on loading new image
            self.mask_item.setPixmap(QPixmap())
            self.mask_item.setVisible(False)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            
            # Reset transform and fit completely to window on initial load
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            
            # Store initial scale for clamping ratios
            self._initial_fit_scale = self.transform().m11()
            self._zoom_touched = False
            self.update_viewer_cursor()
            logger.debug("ImageViewer: Displaying loaded image on canvas. Fit scale: %s", self._initial_fit_scale)
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
            self.mask_item.setVisible(True)
            logger.info("ImageViewer: Labeled mask overlay rendered. Shape: %dx%d, unique labels: %d", w, h, len(unique_labels) - 1)
        else:
            self.mask_item.setPixmap(QPixmap())
            self.mask_item.setVisible(False)

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
        self._setup_ui()
        self._init_connections()
        self._sync_theme()

    def _setup_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setObjectName("PageContainer")
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(16)

        # 1. Left Panel: Image Metadata List (Simplified and Concise)
        self.left_panel = QFrame()
        self.left_panel.setObjectName("AnalysisLeftPanel")
        self.left_panel.setFixedWidth(280)
        self.left_panel.setStyleSheet("""
            #AnalysisLeftPanel {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(16)

        # Panel Header
        left_title = QLabel("Image Properties")
        left_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        left_layout.addWidget(left_title)

        # Clear description placeholder shown when empty
        self.meta_placeholder = QLabel("No active image metadata.\nImport an image to inspect details.")
        self.meta_placeholder.setStyleSheet("color: #6B7280; font-size: 11px;")
        self.meta_placeholder.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.meta_placeholder, 1)

        # Main Metadata Grid
        self.meta_container = QFrame()
        self.meta_grid = QGridLayout(self.meta_container)
        self.meta_grid.setContentsMargins(0, 0, 0, 0)
        self.meta_grid.setSpacing(12)

        # Metadata labels
        fn_lbl = QLabel("Filename:")
        fn_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: bold; text-transform: uppercase;")
        self.fn_val = QLabel("-")
        self.fn_val.setWordWrap(True)
        self.fn_val.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")

        res_lbl = QLabel("Resolution:")
        res_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: bold; text-transform: uppercase;")
        self.res_val = QLabel("-")
        self.res_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        ch_lbl = QLabel("Channels:")
        ch_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: bold; text-transform: uppercase;")
        self.ch_val = QLabel("-")
        self.ch_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        mode_lbl = QLabel("Image Mode:")
        mode_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: bold; text-transform: uppercase;")
        self.mode_val = QLabel("-")
        self.mode_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")

        type_lbl = QLabel("Detected Type:")
        type_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: bold; text-transform: uppercase;")
        self.type_val = QLabel("-")
        self.type_val.setWordWrap(True)
        self.type_val.setStyleSheet("color: #6366F1; font-size: 12px; font-weight: 600;")

        self.meta_grid.addWidget(fn_lbl, 0, 0)
        self.meta_grid.addWidget(self.fn_val, 0, 1)
        self.meta_grid.addWidget(res_lbl, 1, 0)
        self.meta_grid.addWidget(self.res_val, 1, 1)
        self.meta_grid.addWidget(ch_lbl, 2, 0)
        self.meta_grid.addWidget(self.ch_val, 2, 1)
        self.meta_grid.addWidget(mode_lbl, 3, 0)
        self.meta_grid.addWidget(self.mode_val, 3, 1)
        self.meta_grid.addWidget(type_lbl, 4, 0)
        self.meta_grid.addWidget(self.type_val, 4, 1)

        left_layout.addWidget(self.meta_container, 1)
        self.meta_container.setVisible(False)

        self.main_layout.addWidget(self.left_panel)

        # 2. Center Panel: Image Visualization View
        self.center_container = QFrame()
        center_layout = QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)

        self.image_viewer = InteractiveImageViewer(self)
        center_layout.addWidget(self.image_viewer, 1)

        # 2A. Controls Bar underneath the image viewer
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
        
        sep_lbl = QLabel("|")
        sep_lbl.setStyleSheet("color: #2B2B35;")
        controls_layout.addWidget(sep_lbl)
        
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
        
        # Auto Contrast status text inside controls layout
        self.viewer_contrast_lbl = QLabel("✨ Auto Contrast Applied")
        self.viewer_contrast_lbl.setStyleSheet("font-size: 10px; color: #818CF8; font-weight: bold;")
        self.viewer_contrast_lbl.setVisible(False)
        controls_layout.addWidget(self.viewer_contrast_lbl)
        
        center_layout.addWidget(self.viewer_controls_bar)

        # Viewer container is added to self.right_splitter at the end of setup_ui

        # 3. Right Panel: Workflow outlines / action controls
        self.right_panel = QFrame()
        self.right_panel.setObjectName("AnalysisRightPanel")
        self.right_panel.setMinimumWidth(220)
        self.right_panel.setMaximumWidth(500)
        self.right_panel.setStyleSheet("""
            #AnalysisRightPanel {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(12)

        right_title = QLabel("Selected Pipeline")
        right_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        right_layout.addWidget(right_title)

        # Scroll area for parameter settings and guidance profile to prevent UI clipping
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setStyleSheet("background: transparent; border: none;")

        scroll_widget = QWidget()
        scroll_widget.setObjectName("RightScrollWidget")
        scroll_widget.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(14)

        # Active workflow title
        self.wf_title = QLabel("No active workflow")
        self.wf_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #6366F1;")
        scroll_layout.addWidget(self.wf_title)

        # Steps log (not stretching, set to factor 0)
        self.steps_container = QFrame()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(4, 4, 4, 4)
        self.steps_layout.setSpacing(10)
        self.steps_layout.setAlignment(Qt.AlignTop)
        scroll_layout.addWidget(self.steps_container, 0)

        # Method Selector
        self.method_lbl = QLabel("Segmentation Method:")
        self.method_lbl.setObjectName("MethodLabel")
        self.method_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        
        self.method_combo = QComboBox()
        self.method_combo.setObjectName("MethodComboBox")
        self.method_combo.addItems(["AI Segmentation (Cellpose)"])
        self.method_combo.setCurrentText("AI Segmentation (Cellpose)")
        self.method_combo.setCursor(QCursor(Qt.PointingHandCursor))
        
        scroll_layout.addWidget(self.method_lbl)
        scroll_layout.addWidget(self.method_combo)

        # Quality Selector layout (AI params)
        self.quality_frame = QFrame()
        self.quality_frame.setObjectName("QualitySelectorFrame")
        self.quality_frame.setStyleSheet("background: transparent; border: none; margin-bottom: 4px;")
        quality_layout = QVBoxLayout(self.quality_frame)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(4)
        
        self.quality_lbl = QLabel("Segmentation Quality:")
        self.quality_lbl.setObjectName("QualityLabel")
        self.quality_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
        
        self.quality_combo = QComboBox()
        self.quality_combo.setObjectName("QualityComboBox")
        self.quality_combo.addItems(["Fast", "Balanced", "Sensitive", "Precise"])
        self.quality_combo.setCurrentText("Balanced")
        self.quality_combo.setCursor(QCursor(Qt.PointingHandCursor))
        
        quality_layout.addWidget(self.quality_lbl)
        quality_layout.addWidget(self.quality_combo)
        scroll_layout.addWidget(self.quality_frame)
        
        scroll_layout.addStretch(1)

        self.right_scroll.setWidget(scroll_widget)
        right_layout.addWidget(self.right_scroll, 1)

        # Bottom section: Fixed separator line and analysis execution controls
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #2B2B35; max-height: 1px; margin: 4px 0;")
        right_layout.addWidget(sep)

        # Progress and Status indicators (hidden by default)
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

        # Action Buttons
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("RunAnalysisButton")
        self.run_btn.setProperty("class", "PrimaryButton")
        self.run_btn.setCursor(QCursor(Qt.ArrowCursor))
        self.run_btn.setEnabled(False)
        right_layout.addWidget(self.run_btn)

        # Create a horizontal QSplitter to allow dragging and resizing of the right pane
        from PySide6.QtWidgets import QSplitter
        self.right_splitter = QSplitter(Qt.Horizontal)
        self.right_splitter.setObjectName("AnalysisRightSplitter")
        self.right_splitter.addWidget(self.center_container)
        self.right_splitter.addWidget(self.right_panel)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 0)
        self.right_splitter.setCollapsible(0, False)
        self.right_splitter.setCollapsible(1, False)
        self.right_splitter.setSizes([800, 260])
        
        self.main_layout.addWidget(self.right_splitter, 1)

    def _init_connections(self):
        # Trigger actual Cellpose analysis on click
        self.run_btn.clicked.connect(self._on_run_analysis_clicked)
        
        # Connect visual controls updates
        self.show_original_chk.toggled.connect(self._on_show_original_toggled)
        self.show_overlay_chk.toggled.connect(self._on_show_overlay_toggled)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        self.quality_combo.currentTextChanged.connect(self._on_quality_combo_changed)

        # Segmentation controls connections
        self.method_combo.currentTextChanged.connect(self._on_method_combo_changed)
        
        # Handle state triggers
        state.image_loaded.connect(self._on_image_loaded)
        state.workflow_selected.connect(self._on_workflow_selected)
        state.theme_changed.connect(self._sync_theme)
        state.page_changed.connect(self._on_page_changed)
        
        # Connect Phase 2D transient state change listener slots
        state.show_original_changed.connect(self._on_state_show_original_changed)
        state.show_overlay_changed.connect(self._on_state_show_overlay_changed)
        state.mask_opacity_changed.connect(self._on_state_mask_opacity_changed)
        state.quality_mode_changed.connect(self._on_state_quality_mode_changed)

        # Connect Segmentation state change listener slots
        state.segmentation_method_changed.connect(self._on_state_segmentation_method_changed)
        
        # Initial boot check
        self._sync_state()

    def _sync_state(self):
        """Initial check for loaded state."""
        if state.current_image_path:
            self._on_image_loaded(state.current_image_path)
        else:
            self._on_image_loaded("")
        if state.current_workflow:
            self._on_workflow_selected(state.current_workflow)
            
        # Sync Phase 2D state properties
        self._on_state_show_original_changed(state.show_original_image)
        self._on_state_show_overlay_changed(state.show_segmentation_overlay)
        self._on_state_mask_opacity_changed(state.mask_opacity)
        self._on_state_quality_mode_changed(state.quality_mode)

        # Sync Segmentation state properties
        self._on_state_segmentation_method_changed(state.segmentation_method)

    # Slots to update state from controls
    def _on_show_original_toggled(self, checked: bool):
        state.show_original_image = checked

    def _on_show_overlay_toggled(self, checked: bool):
        state.show_segmentation_overlay = checked

    def _on_opacity_slider_changed(self, val: int):
        state.mask_opacity = val

    def _on_quality_combo_changed(self, text: str):
        state.quality_mode = text

    def _on_method_combo_changed(self, text: str):
        if text == "AI Segmentation (Cellpose)":
            state.segmentation_method = "AI Segmentation"
        else:
            state.segmentation_method = text

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
            self.quality_frame.setVisible(True)
        else:
            self.method_combo.setCurrentText(val)
            self.quality_frame.setVisible(False)
        self.method_combo.blockSignals(False)

    @Slot(str)
    def _on_image_loaded(self, path: str):
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
                
                # Extract simplified metadata fields
                meta = image_manager.get_metadata()
                if meta:
                    self.fn_val.setText(meta.get("filename", "-"))
                    self.res_val.setText(f"{meta.get('width')} × {meta.get('height')}")
                    self.ch_val.setText(str(meta.get("channels", "-")))
                    self.mode_val.setText(meta.get("mode", "-").upper())
                    self.type_val.setText(meta.get("classification", "-"))

                    # Swap visibility
                    self.meta_placeholder.setVisible(False)
                    self.meta_container.setVisible(True)

                self.run_btn.setEnabled(True)
                self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
                return
        
        # Clear/empty state
        self.image_viewer.clear()
        self.image_viewer.set_analysis_results(None)
        self.viewer_contrast_lbl.setVisible(False)
        self.meta_placeholder.setVisible(True)
        self.meta_container.setVisible(False)
        
        self.run_btn.setEnabled(False)
        self.run_btn.setCursor(QCursor(Qt.ArrowCursor))

    @Slot(str)
    def _on_workflow_selected(self, wf_id: str):
        wf = workflow_manager.get_workflow(wf_id)
        if wf:
            self.wf_title.setText(wf.name)
            
            # Clear previous steps
            for i in reversed(range(self.steps_layout.count())): 
                widget = self.steps_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Render standard pipeline outline steps: 1. Preprocessing, 2. Segmentation, etc.
            pipeline_steps = [
                "Preprocessing",
                "Segmentation",
                "Counting & Extraction",
                "Quantification Report"
            ]
            for idx, step in enumerate(pipeline_steps):
                step_lbl = QLabel(f" {idx + 1}. {step}")
                step_lbl.setToolTip("Coming in upcoming analysis phases")
                step_lbl.setCursor(Qt.ArrowCursor)
                step_lbl.setStyleSheet("""
                    QLabel {
                        background-color: #24242B;
                        border: 1px solid #2B2B35;
                        border-radius: 4px;
                        padding: 8px;
                        font-size: 11px;
                        color: #E5E7EB;
                    }
                """)
                self.steps_layout.addWidget(step_lbl)
        else:
            self.wf_title.setText("No active workflow")
            
            # Clean layout
            for i in reversed(range(self.steps_layout.count())): 
                widget = self.steps_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

    def _set_controls_enabled(self, enabled: bool):
        self.method_combo.setEnabled(enabled)
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

        # Disable buttons/combo to prevent double execution and prepare progress display
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Analyzing...")
        self._set_controls_enabled(False)
        
        self.status_lbl.setText("Starting analysis pipeline...")
        self.status_lbl.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

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
        parameters = {
            "segmentation_method": state.segmentation_method,
            "quality_mode": state.quality_mode
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

    @Slot(str)
    def _on_analysis_status(self, status_msg: str):
        self.status_lbl.setText(status_msg)

    @Slot(dict)
    def _on_analysis_finished(self, results: dict):
        logger.info("AnalysisPage: Analysis finished successfully.")
        
        # Restore button and hide progress
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._set_controls_enabled(True)
        self.status_lbl.setVisible(False)
        self.progress_bar.setVisible(False)
        
        # Populate results_dict into the state
        state.analysis_results = results
        self.image_viewer.set_analysis_results(results)
        
        # Render the mask overlay in the image viewer
        masks = results.get("masks")
        if masks is not None:
            self.image_viewer.set_masks(masks)
            
        QMessageBox.information(
            self,
            "Analysis Completed",
            "Segmentation and analysis completed successfully!",
            QMessageBox.Ok
        )
        
        # Navigate to Results view
        from lumen.core.services.navigation_service import navigation_service
        navigation_service.navigate_to("results")

    @Slot(str)
    def _on_page_changed(self, page_name: str):
        if page_name == "analysis":
            self._sync_state()

    @Slot(str)
    def _on_analysis_failed(self, error_msg: str):
        logger.error("AnalysisPage: Analysis failed: %s", error_msg)
        
        # Restore button and hide progress
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._set_controls_enabled(True)
        self.status_lbl.setVisible(False)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(
            self,
            "Analysis Failed",
            f"An error occurred during analysis:\n\n{error_msg}",
            QMessageBox.Ok
        )

    @Slot(str)
    def _sync_theme(self, theme_name: str = ""):
        theme = theme_service.current_theme
        self.image_viewer.sync_theme(theme)
        if theme == "light":
            # Style splitter handle for light mode
            self.right_splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #E5E7EB;
                    width: 5px;
                }
                QSplitter::handle:hover {
                    background-color: #4F46E5;
                }
            """)
            # Adjust controls bar style for light mode
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
            self.viewer_contrast_lbl.setStyleSheet("font-size: 10px; color: #4F46E5; font-weight: bold; background: transparent;")
            
            # Adjust panels styling
            self.left_panel.setStyleSheet("""
                #AnalysisLeftPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.right_panel.setStyleSheet("""
                #AnalysisRightPanel {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            left_lbls = self.left_panel.findChildren(QLabel)
            for lbl in left_lbls:
                if lbl.objectName() != "PageTitle":
                    lbl.setStyleSheet("color: #1F2937;")
            
            self.fn_val.setStyleSheet("color: #111827; font-weight: bold; font-size: 12px;")
            self.res_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.ch_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.mode_val.setStyleSheet("color: #4B5563; font-size: 12px;")
            self.type_val.setStyleSheet("color: #4F46E5; font-size: 12px; font-weight: 600;")
            
            self.wf_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #4F46E5;")
            self.right_panel.findChildren(QLabel)[0].setStyleSheet("font-size: 15px; font-weight: bold; color: #111827;")
 
            # Sync steps labels inside layout
            for i in range(self.steps_layout.count()):
                w = self.steps_layout.itemAt(i).widget()
                if w:
                    w.setStyleSheet("""
                        background-color: #F3F4F6;
                        border: 1px solid #D1D5DB;
                        border-radius: 4px;
                        padding: 8px;
                        font-size: 11px;
                        color: #1F2937;
                    """)
            
            # Light mode quality selector label
            self.quality_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #4B5563; text-transform: uppercase;")
            self.method_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #4B5563; text-transform: uppercase;")
            
            # Light mode progress and status styles
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
        else:
            # Style splitter handle for dark mode
            self.right_splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #2B2B35;
                    width: 5px;
                }
                QSplitter::handle:hover {
                    background-color: #6366F1;
                }
            """)
            # Adjust controls bar style for dark mode
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
            self.viewer_contrast_lbl.setStyleSheet("font-size: 10px; color: #818CF8; font-weight: bold; background: transparent;")
            
            self.left_panel.setStyleSheet("""
                #AnalysisLeftPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.right_panel.setStyleSheet("""
                #AnalysisRightPanel {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.fn_val.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 12px;")
            self.res_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.ch_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.mode_val.setStyleSheet("color: #E5E7EB; font-size: 12px;")
            self.type_val.setStyleSheet("color: #6366F1; font-size: 12px; font-weight: 600;")
            
            self.wf_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #6366F1;")
            self.right_panel.findChildren(QLabel)[0].setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
            
            for i in range(self.steps_layout.count()):
                w = self.steps_layout.itemAt(i).widget()
                if w:
                    w.setStyleSheet("""
                        background-color: #24242B;
                        border: 1px solid #2B2B35;
                        border-radius: 4px;
                        padding: 8px;
                        font-size: 11px;
                        color: #E5E7EB;
                    """)
            
            # Dark mode quality selector label
            self.quality_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
            self.method_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #6B7280; text-transform: uppercase;")
            
            # Dark mode progress and status styles
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
