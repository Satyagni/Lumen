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
        self._loaded_image_path = None
        self._loaded_image_origin = None
        self._setup_ui()
        self._init_connections()
        self._sync_theme()

    def _setup_ui(self):
        self.page_layout = QVBoxLayout(self)
        self.page_layout.setObjectName("PageVerticalLayout")
        self.page_layout.setContentsMargins(20, 20, 20, 20)
        self.page_layout.setSpacing(12)

        from lumen.ui.workspace_switcher import WorkspaceSwitcher
        self.workspace_switcher = WorkspaceSwitcher("single")
        self.page_layout.addWidget(self.workspace_switcher)

        self.main_layout = QHBoxLayout()
        self.main_layout.setObjectName("PageContainer")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
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
        self.edit_btn = QPushButton("✏ Edit Masks")
        self.edit_btn.setObjectName("EditMasksButton")
        self.edit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.edit_btn.setEnabled(False)
        right_layout.addWidget(self.edit_btn)

        self.save_analysis_btn = QPushButton("💾 Save Analysis")
        self.save_analysis_btn.setObjectName("SaveAnalysisButton")
        self.save_analysis_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.save_analysis_btn.setEnabled(False)
        right_layout.addWidget(self.save_analysis_btn)

        self.reset_changes_btn = QPushButton("🔄 Reset Changes")
        self.reset_changes_btn.setObjectName("ResetChangesButton")
        self.reset_changes_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.reset_changes_btn.setEnabled(False)
        right_layout.addWidget(self.reset_changes_btn)

        self.dirty_lbl = QLabel("")
        self.dirty_lbl.setObjectName("DirtyStatusLabel")
        self.dirty_lbl.setAlignment(Qt.AlignCenter)
        self.dirty_lbl.setVisible(False)
        self.dirty_lbl.setStyleSheet("color: #ff9800; font-weight: bold; margin: 4px;")
        right_layout.addWidget(self.dirty_lbl)

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
        self.page_layout.addLayout(self.main_layout, 1)

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
        
        # Invalidate loaded image cache on analysis results changes
        state.analysis_completed.connect(self._invalidate_analysis_cache)
        state.analysis_results_updated.connect(self._invalidate_analysis_cache)
        
        # Handle dirty state transitions
        state.dirty_state_changed.connect(self._on_dirty_state_changed)
        self.save_analysis_btn.clicked.connect(self._on_save_clicked)
        
        # Initial boot check
        self._sync_state()

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
        if session:
            # Check if session values match current state and page is already loaded
            state_matches_session = (
                state.quality_mode == session.quality_mode and
                state.mask_opacity == session.mask_opacity and
                state.show_original_image == session.show_original_image and
                state.show_segmentation_overlay == session.show_segmentation_overlay and
                state.segmentation_method == session.segmentation_method and
                state.current_workflow == session.current_workflow and
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
        session.quality_mode = state.quality_mode
        session.mask_opacity = state.mask_opacity
        session.show_original_image = state.show_original_image
        session.show_segmentation_overlay = state.show_segmentation_overlay
        session.segmentation_method = state.segmentation_method
        session.current_workflow = state.current_workflow
        
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
        path = session.image_path
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

                    self.meta_placeholder.setVisible(False)
                    self.meta_container.setVisible(True)

                self.run_btn.setEnabled(True)
                self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
                
                if session.current_workflow:
                    self._on_workflow_selected(session.current_workflow)
                    
                self._loaded_image_path = path
                self._loaded_image_origin = state.current_origin_type
                self.force_layout_refresh()

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
        self.force_layout_refresh()

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

                    # Swap visibility
                    self.meta_placeholder.setVisible(False)
                    self.meta_container.setVisible(True)

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
        self.force_layout_refresh()
        
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
            
        # Trigger lightweight session checkpoint save
        self._save_to_session()

    @Slot(str)
    def _on_page_changed(self, page_name: str):
        if page_name == "analysis":
            self._sync_state()
        else:
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
            
            self.edit_btn.setStyleSheet("""
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
            """)
            
            self.save_analysis_btn.setStyleSheet("""
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
            """)
            
            self.reset_changes_btn.setStyleSheet("""
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
            """)
            
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
            
            self.edit_btn.setStyleSheet("""
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
            """)
            
            self.save_analysis_btn.setStyleSheet("""
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
            """)
            
            self.reset_changes_btn.setStyleSheet("""
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
            """)
            
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
