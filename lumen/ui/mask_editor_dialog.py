import os
import colorsys
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QButtonGroup, QMessageBox
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QPainter

from lumen.core.logger import logger
from lumen.core.services.theme_service import theme_service

def update_results_mask(original_results: dict, edited_mask: np.ndarray) -> dict:
    """Updates masks, cell_count, cell_metrics, and statistics in results dictionary."""
    new_results = dict(original_results)
    new_results["masks"] = edited_mask
    
    unique_labels = np.unique(edited_mask)
    valid_labels = [label for label in unique_labels if label != 0]
    cell_count = len(valid_labels)
    new_results["cell_count"] = cell_count
    
    cell_metrics = {}
    cell_areas = []
    diams = []
    for label in valid_labels:
        indices = np.argwhere(edited_mask == label)
        area = len(indices)
        cell_areas.append(area)
        if area > 0:
            mean_y, mean_x = np.mean(indices, axis=0)
            diameter = round(2 * np.sqrt(area / np.pi), 2)
            diams.append(diameter)
            cell_metrics[int(label)] = {
                "area_px": int(area),
                "centroid": (round(float(mean_x), 1), round(float(mean_y), 1)),
                "diameter_px": float(diameter)
            }
            
    new_results["cell_metrics"] = cell_metrics
    
    if cell_count > 0:
        mean_cell_area_px = float(np.mean(cell_areas))
        median_cell_area_px = float(np.median(cell_areas))
        avg_diameter = float(np.mean(diams))
    else:
        mean_cell_area_px = 0.0
        median_cell_area_px = 0.0
        avg_diameter = 0.0
        
    new_results["average_diameter_px"] = round(avg_diameter, 2)
    new_results["mean_cell_area_px"] = round(mean_cell_area_px, 2)
    new_results["median_cell_area_px"] = round(median_cell_area_px, 2)
    
    # Update cell density
    h, w = edited_mask.shape[:2]
    image_area = h * w
    new_results["cell_density"] = float(cell_count / image_area) if image_area > 0 else 0.0
    
    return new_results


class MaskEditorCanvas(QGraphicsView):
    """Drawing canvas for manual mask editing, supporting brush strokes, zoom, and pan."""
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Raw background image item
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Labeled masks overlay item
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(1.0)
        self.scene.addItem(self.mask_item)

        # Selection overlay item
        self.selection_item = None

        # Circular brush outline cursor item
        self.brush_cursor_item = QGraphicsEllipseItem()
        self.brush_cursor_item.setPen(QPen(QColor(255, 255, 255, 220), 1.5, Qt.SolidLine))
        self.brush_cursor_item.setBrush(QBrush(QColor(255, 255, 255, 40)))
        self.brush_cursor_item.setZValue(3.0)
        self.scene.addItem(self.brush_cursor_item)
        self.brush_cursor_item.hide()

        # Render configurations
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Navigation and drawing states
        self._zoom_factor = 1.15
        self._zoom_touched = False
        self._initial_fit_scale = None
        self._is_panning = False
        self.drawing = False
        self.last_scene_pos = None

        self.working_mask = None
        self.color_lut = None
        self.brush_mode = "pointer"  # Default checked
        self.brush_size = 10
        
        # Object-aware selection state
        self.selected_labels = set()
        self._selected_label_id = None

        # Capped undo/redo stacks (max 10 items)
        self.undo_stack = []
        self.redo_stack = []

    @property
    def selected_label_id(self):
        return self._selected_label_id

    @selected_label_id.setter
    def selected_label_id(self, val):
        self._selected_label_id = val
        if val is not None and val > 0:
            if val not in self.selected_labels:
                self.selected_labels = {val}
        else:
            self.selected_labels.clear()

    def set_data(self, pixmap: QPixmap, mask_arr: np.ndarray, color_lut: np.ndarray):
        """Loads Raw pixmap and segmentation mask array into drawing scene."""
        self.working_mask = mask_arr
        self.color_lut = color_lut
        self.selected_labels.clear()
        self.selected_label_id = None
        
        # Clear selection overlay if present
        if self.selection_item:
            try:
                self.scene.removeItem(self.selection_item)
            except Exception:
                pass
            self.selection_item = None

        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        self.update_mask_overlay()

        # Initial fit in view
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._initial_fit_scale = self.transform().m11()
        self._zoom_touched = False

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selection_changed.emit()

    def update_mask_overlay(self):
        """Renders colorized mask overlay using the RGBA Color LUT."""
        if self.working_mask is None or self.color_lut is None:
            return

        rgba_arr = self.color_lut[self.working_mask]
        h, w = self.working_mask.shape

        qimg = QImage(rgba_arr.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.mask_item.setPixmap(pixmap)
        self.mask_item.setVisible(True)

    def update_selection_highlight(self):
        """Draws a boundary highlight around all selected cell labels."""
        if self.selection_item:
            try:
                self.scene.removeItem(self.selection_item)
            except Exception:
                pass
            self.selection_item = None
            
        if self.working_mask is None or not self.selected_labels:
            return
            
        # Find coordinates of all selected labels
        sel_mask = np.isin(self.working_mask, list(self.selected_labels))
        rows, cols = np.where(sel_mask)
        if len(rows) == 0:
            return
            
        min_row, max_row = int(np.min(rows)), int(np.max(rows))
        min_col, max_col = int(np.min(cols)), int(np.max(cols))
        
        sub_h = max_row - min_row + 1
        sub_w = max_col - min_col + 1
        
        qimg = QImage(sub_w, sub_h, QImage.Format_ARGB32)
        qimg.fill(Qt.transparent)
        
        sub_mask = self.working_mask[min_row:max_row+1, min_col:max_col+1]
        
        for r in range(sub_h):
            for c in range(sub_w):
                val = sub_mask[r, c]
                if val in self.selected_labels:
                    is_boundary = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= sub_h or nc < 0 or nc >= sub_w or sub_mask[nr, nc] != val:
                            is_boundary = True
                            break
                    
                    if is_boundary:
                        # Bright high-contrast cyan highlight
                        qimg.setPixelColor(c, r, QColor(0, 255, 255, 255))
                    else:
                        qimg.setPixelColor(c, r, QColor(0, 255, 255, 80))
                        
        self.selection_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.selection_item.setPos(min_col, min_row)
        self.selection_item.setZValue(2.0)
        self.scene.addItem(self.selection_item)

    def ensure_lut_color(self, label_id: int):
        """Generates random HSV color if label ID exceeds LUT mapping bounds."""
        if label_id >= len(self.color_lut):
            padding = np.zeros((max(label_id - len(self.color_lut) + 1000, 1000), 4), dtype=np.uint8)
            self.color_lut = np.vstack([self.color_lut, padding])

        if np.all(self.color_lut[label_id] == 0):
            np.random.seed(label_id)
            h = np.random.rand()
            s = 0.8 + 0.2 * np.random.rand()
            v = 0.8 + 0.2 * np.random.rand()
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            self.color_lut[label_id] = [int(r * 255), int(g * 255), int(b * 255), 102]

    # ----------------------------------------------------
    # Undo / Redo Stack logic (Capped at 10 items)
    # ----------------------------------------------------
    def push_undo(self):
        if len(self.undo_stack) >= 10:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.working_mask.copy())
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            if len(self.redo_stack) >= 10:
                self.redo_stack.pop(0)
            self.redo_stack.append(self.working_mask.copy())
            self.working_mask = self.undo_stack.pop()
            
            # Keep selection valid if cells still exist
            still_valid = set()
            for label in self.selected_labels:
                if np.any(self.working_mask == label):
                    still_valid.add(label)
            self.selected_labels = still_valid
            
            if self.selected_label_id not in self.selected_labels:
                self.selected_label_id = list(self.selected_labels)[0] if self.selected_labels else None
            
            self.update_selection_highlight()
            self.update_mask_overlay()
            self.selection_changed.emit()

    def redo(self):
        if self.redo_stack:
            if len(self.undo_stack) >= 10:
                self.undo_stack.pop(0)
            self.undo_stack.append(self.working_mask.copy())
            self.working_mask = self.redo_stack.pop()
            
            # Keep selection valid if cells still exist
            still_valid = set()
            for label in self.selected_labels:
                if np.any(self.working_mask == label):
                    still_valid.add(label)
            self.selected_labels = still_valid
            
            if self.selected_label_id not in self.selected_labels:
                self.selected_label_id = list(self.selected_labels)[0] if self.selected_labels else None
                    
            self.update_selection_highlight()
            self.update_mask_overlay()
            self.selection_changed.emit()

    # ----------------------------------------------------
    # Drawing & Brush stroke interpolation
    # ----------------------------------------------------
    def draw_brush_circle(self, cx: int, cy: int, r: int, value: int):
        h, w = self.working_mask.shape
        x_min = max(0, cx - r)
        x_max = min(w, cx + r + 1)
        y_min = max(0, cy - r)
        y_max = min(h, cy + r + 1)

        if x_min >= x_max or y_min >= y_max:
            return

        Y, X = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (X - cx)**2 + (Y - cy)**2
        
        if value == 0: # Erase brush
            if self.selected_labels:
                mask_to_erase = (dist_sq <= r**2) & np.isin(self.working_mask[y_min:y_max, x_min:x_max], list(self.selected_labels))
                self.working_mask[y_min:y_max, x_min:x_max][mask_to_erase] = 0
        else: # Add brush
            mask_to_paint = dist_sq <= r**2
            self.working_mask[y_min:y_max, x_min:x_max][mask_to_paint] = value

    def draw_brush_stroke(self, p1: tuple, p2: tuple, radius: int, value: int):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.hypot(dx, dy)
        step = max(1.0, radius / 4.0)

        if dist < step:
            self.draw_brush_circle(p2[0], p2[1], radius, value)
        else:
            num_steps = int(dist / step)
            for i in range(num_steps + 1):
                t = i / num_steps
                cx = int(p1[0] + t * dx)
                cy = int(p1[1] + t * dy)
                self.draw_brush_circle(cx, cy, radius, value)

    # ----------------------------------------------------
    # Deletion, Insertion & Merging helpers
    # ----------------------------------------------------
    def delete_selected_cells(self):
        if self.working_mask is None or not self.selected_labels:
            return
        self.push_undo()
        for label in self.selected_labels:
            self.working_mask[self.working_mask == label] = 0
        self.selected_labels.clear()
        self.selected_label_id = None
        self.update_selection_highlight()
        self.update_mask_overlay()
        self.selection_changed.emit()

    def merge_selected_cells(self):
        if self.working_mask is None or len(self.selected_labels) < 2:
            return
        self.push_undo()
        surviving_id = min(self.selected_labels)
        for label in self.selected_labels:
            if label != surviving_id:
                self.working_mask[self.working_mask == label] = surviving_id
        self.selected_labels = {surviving_id}
        self.selected_label_id = surviving_id
        self.update_selection_highlight()
        self.update_mask_overlay()
        self.selection_changed.emit()

    def add_new_cell(self):
        if self.working_mask is None:
            return
        # Generate a brand new label ID
        new_id = int(np.max(self.working_mask) + 1) if self.working_mask.size > 0 else 1
        self.ensure_lut_color(new_id)
        self.selected_labels = {new_id}
        self.selected_label_id = new_id
        self.update_selection_highlight()
        self.selection_changed.emit()

    # ----------------------------------------------------
    # Mouse & Event Overrides
    # ----------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() in [Qt.RightButton, Qt.MiddleButton]:
            self._is_panning = True
            self._pan_start_x = event.position().x()
            self._pan_start_y = event.position().y()
            self._h_bar_start = self.horizontalScrollBar().value()
            self._v_bar_start = self.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.working_mask is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            cx, cy = int(scene_pos.x()), int(scene_pos.y())

            h, w = self.working_mask.shape
            if 0 <= cx < w and 0 <= cy < h:
                clicked_label = self.working_mask[cy, cx]
                
                if self.brush_mode == "pointer":
                    # Mode A: Selection ONLY in Pointer Mode
                    if clicked_label > 0:
                        if event.modifiers() & Qt.ControlModifier:
                            if int(clicked_label) in self.selected_labels:
                                self.selected_labels.remove(int(clicked_label))
                            else:
                                self.selected_labels.add(int(clicked_label))
                        else:
                            self.selected_labels = {int(clicked_label)}
                            
                        self.selected_label_id = list(self.selected_labels)[0] if self.selected_labels else None
                        self.update_selection_highlight()
                        self.selection_changed.emit()
                    else:
                        # Clicked empty space
                        if not (event.modifiers() & Qt.ControlModifier):
                            self.selected_labels.clear()
                            self.selected_label_id = None
                            self.update_selection_highlight()
                            self.selection_changed.emit()
                        
                    # Drag to pan in pointer mode
                    self._is_panning = True
                    self._pan_start_x = event.position().x()
                    self._pan_start_y = event.position().y()
                    self._h_bar_start = self.horizontalScrollBar().value()
                    self._v_bar_start = self.verticalScrollBar().value()
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    # Brush mode (Add / Erase)
                    # Auto-select the clicked cell if none was selected and we clicked a cell
                    if self.selected_label_id is None and clicked_label > 0:
                        self.selected_labels = {int(clicked_label)}
                        self.selected_label_id = int(clicked_label)
                        self.update_selection_highlight()
                        self.selection_changed.emit()

                    # Mode B: Paint or erase on selected cell (NO Ctrl required)
                    if self.selected_label_id is not None:
                        self.push_undo()
                        self.drawing = True
                        self.last_scene_pos = (cx, cy)
                        draw_value = 0 if self.brush_mode == "erase" else self.selected_label_id
                        self.draw_brush_circle(cx, cy, self.brush_size, draw_value)
                        self.update_mask_overlay()

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        cx, cy = int(scene_pos.x()), int(scene_pos.y())

        # Update cursor outline position
        if self.working_mask is not None:
            r = self.brush_size
            self.brush_cursor_item.setRect(cx - r, cy - r, 2 * r, 2 * r)
            if self.brush_mode != "pointer":
                self.brush_cursor_item.show()
            else:
                self.brush_cursor_item.hide()

        if getattr(self, "_is_panning", False):
            dx = event.position().x() - self._pan_start_x
            dy = event.position().y() - self._pan_start_y
            self.horizontalScrollBar().setValue(self._h_bar_start - dx)
            self.verticalScrollBar().setValue(self._v_bar_start - dy)
            self._zoom_touched = True
            event.accept()
            return

        if getattr(self, "drawing", False) and self.working_mask is not None and self.selected_label_id is not None:
            draw_value = 0 if self.brush_mode == "erase" else self.selected_label_id
            if self.last_scene_pos:
                self.draw_brush_stroke(self.last_scene_pos, (cx, cy), self.brush_size, draw_value)
            else:
                self.draw_brush_circle(cx, cy, self.brush_size, draw_value)

            self.last_scene_pos = (cx, cy)
            self.update_mask_overlay()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in [Qt.RightButton, Qt.MiddleButton]:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            if getattr(self, "drawing", False):
                self.drawing = False
                self.last_scene_pos = None
                self.update_selection_highlight()
                self.selection_changed.emit()
            elif getattr(self, "_is_panning", False):
                self._is_panning = False
                self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.brush_cursor_item.hide()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if not self.pixmap_item.pixmap().isNull():
            event.accept()
            delta = event.angleDelta().y()
            if delta == 0:
                return

            current_scale = self.transform().m11()
            if self._initial_fit_scale is None:
                self._initial_fit_scale = current_scale

            factor = self._zoom_factor if delta > 0 else (1.0 / self._zoom_factor)
            new_scale = current_scale * factor

            max_limit = self._initial_fit_scale * 20.0
            min_limit = self._initial_fit_scale * 0.1

            if new_scale > max_limit:
                new_scale = max_limit
            elif new_scale < min_limit:
                new_scale = min_limit

            relative_factor = new_scale / current_scale
            self.scale(relative_factor, relative_factor)
            self._zoom_touched = True
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.pixmap_item.pixmap().isNull() and not self._zoom_touched:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self.pixmap_item.pixmap().isNull():
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._zoom_touched = False
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MaskEditorDialog(QDialog):
    """Scientific manual mask correction dialog containing brushes, sliders, and canvas views."""

    def __init__(self, image_path: str, original_mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Mask Refinement")
        self.setMinimumSize(1000, 700)
        self.setObjectName("MaskEditorDialog")
        
        # Maximize and minimize flags
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        # Copy original masks for non-destructive session staging
        self.original_mask = original_mask
        self.working_mask = original_mask.copy()

        # Load microscopy raw image
        from lumen.pages.batch_explorer_page import load_microscopy_pixmap
        self.pixmap = load_microscopy_pixmap(image_path)

        # Pre-seed RGBA Color LUT with existing segment labels
        max_label = np.max(original_mask) if original_mask.size > 0 else 0
        lut_size = max(max_label + 1000, 2000)
        self.color_lut = np.zeros((lut_size, 4), dtype=np.uint8)

        np.random.seed(42)
        import colorsys
        unique_labels = np.unique(original_mask)
        for label in unique_labels:
            if label == 0:
                continue
            h = np.random.rand()
            s = 0.8 + 0.2 * np.random.rand()
            v = 0.8 + 0.2 * np.random.rand()
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            self.color_lut[label] = [int(r * 255), int(g * 255), int(b * 255), 102]

        self._setup_ui()
        self._sync_theme()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 1. Header Layout
        header = QHBoxLayout()
        title_lbl = QLabel("Manual Mask Refinement")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold;")
        hint_lbl = QLabel("Pointer: left-click to select / drag to pan. Add/Erase: edit selected cells.")
        hint_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        header.addWidget(title_lbl)
        header.addStretch(1)
        header.addWidget(hint_lbl)
        layout.addLayout(header)

        # 1B. Selection Info Status Bar
        status_bar = QHBoxLayout()
        self.status_info_lbl = QLabel("No cell selected. Click a cell to edit, or click '➕ New Cell'.")
        self.status_info_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #6366F1; margin-left: 4px;")
        status_bar.addWidget(self.status_info_lbl)
        status_bar.addStretch(1)
        layout.addLayout(status_bar)

        # 2. Toolbar layout
        self.toolbar = QFrame()
        self.toolbar.setObjectName("EditorToolbar")
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(12)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self.pointer_btn = QPushButton("🔍 Pointer")
        self.pointer_btn.setCheckable(True)
        self.pointer_btn.setChecked(True)
        self.pointer_btn.setCursor(Qt.PointingHandCursor)

        self.add_btn = QPushButton("✏ Add Brush")
        self.add_btn.setCheckable(True)
        self.add_btn.setCursor(Qt.PointingHandCursor)

        self.erase_btn = QPushButton("🧽 Erase Brush")
        self.erase_btn.setCheckable(True)
        self.erase_btn.setCursor(Qt.PointingHandCursor)

        self.btn_group.addButton(self.pointer_btn)
        self.btn_group.addButton(self.add_btn)
        self.btn_group.addButton(self.erase_btn)

        toolbar_layout.addWidget(self.pointer_btn)
        toolbar_layout.addWidget(self.add_btn)
        toolbar_layout.addWidget(self.erase_btn)

        # Separator after Brushes
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color: #2B2B35; max-width: 1px;")
        toolbar_layout.addWidget(sep1)

        # Cell Actions
        self.new_cell_btn = QPushButton("➕ New Cell")
        self.new_cell_btn.setCursor(Qt.PointingHandCursor)
        
        self.delete_btn = QPushButton("❌ Delete Selected")
        self.delete_btn.setObjectName("DeleteCellButton")
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setEnabled(False)

        self.merge_btn = QPushButton("🔗 Merge Selected")
        self.merge_btn.setCursor(Qt.PointingHandCursor)
        self.merge_btn.setEnabled(False)

        self.reset_btn = QPushButton("🔄 Reset All")
        self.reset_btn.setCursor(Qt.PointingHandCursor)

        toolbar_layout.addWidget(self.new_cell_btn)
        toolbar_layout.addWidget(self.delete_btn)
        toolbar_layout.addWidget(self.merge_btn)
        toolbar_layout.addWidget(self.reset_btn)

        # Separator after Actions
        sep_actions = QFrame()
        sep_actions.setFrameShape(QFrame.VLine)
        sep_actions.setStyleSheet("background-color: #2B2B35; max-width: 1px;")
        toolbar_layout.addWidget(sep_actions)

        # Slider and Label
        self.size_lbl = QLabel("Size: 10px")
        self.size_lbl.setStyleSheet("font-size: 11px; font-weight: bold; min-width: 70px;")
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 100)
        self.size_slider.setValue(10)
        self.size_slider.setFixedWidth(120)
        self.size_slider.setCursor(Qt.PointingHandCursor)

        toolbar_layout.addWidget(self.size_lbl)
        toolbar_layout.addWidget(self.size_slider)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background-color: #2B2B35; max-width: 1px;")
        toolbar_layout.addWidget(sep2)

        # Undo/Redo Buttons
        self.undo_btn = QPushButton("↩ Undo")
        self.undo_btn.setCursor(Qt.PointingHandCursor)

        self.redo_btn = QPushButton("↪ Redo")
        self.redo_btn.setCursor(Qt.PointingHandCursor)

        toolbar_layout.addWidget(self.undo_btn)
        toolbar_layout.addWidget(self.redo_btn)

        toolbar_layout.addStretch(1)

        # Cancel/Save
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)

        self.save_btn = QPushButton("Save Mask")
        self.save_btn.setCursor(Qt.PointingHandCursor)

        toolbar_layout.addWidget(self.cancel_btn)
        toolbar_layout.addWidget(self.save_btn)

        layout.addWidget(self.toolbar)

        # 3. Canvas setup
        self.canvas = MaskEditorCanvas(self)
        layout.addWidget(self.canvas, 1)

        # Attach raw elements to canvas
        self.canvas.set_data(self.pixmap, self.working_mask, self.color_lut)

        # Connect actions
        self.pointer_btn.clicked.connect(self._set_pointer_mode)
        self.add_btn.clicked.connect(self._set_add_mode)
        self.erase_btn.clicked.connect(self._set_erase_mode)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn.clicked.connect(self.canvas.redo)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.new_cell_btn.clicked.connect(self._on_new_cell_clicked)
        self.delete_btn.clicked.connect(self._on_delete_cell_clicked)
        self.merge_btn.clicked.connect(self._on_merge_clicked)
        self.reset_btn.clicked.connect(self._on_reset_all_clicked)
        self.canvas.selection_changed.connect(self._update_selection_status)

    def _set_pointer_mode(self):
        self.canvas.brush_mode = "pointer"

    def _set_add_mode(self):
        self.canvas.brush_mode = "add"

    def _set_erase_mode(self):
        self.canvas.brush_mode = "erase"

    def _on_size_changed(self, value: int):
        self.canvas.brush_size = value
        self.size_lbl.setText(f"Size: {value}px")

    def _on_new_cell_clicked(self):
        self.canvas.add_new_cell()
        self.add_btn.setChecked(True)
        self.canvas.brush_mode = "add"

    def _on_delete_cell_clicked(self):
        self.canvas.delete_selected_cells()

    def _on_merge_clicked(self):
        self.canvas.merge_selected_cells()

    def _on_reset_all_clicked(self):
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Reset All Masks",
            "Remove all segmentation masks for this image?\nThis will clear the entire canvas.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.canvas.push_undo()
            self.canvas.working_mask.fill(0)
            self.canvas.selected_labels.clear()
            self.canvas.selected_label_id = None
            self.canvas.update_selection_highlight()
            self.canvas.update_mask_overlay()
            self.canvas.selection_changed.emit()

    def _update_selection_status(self):
        sel_count = len(self.canvas.selected_labels)
        theme = theme_service.current_theme
        color = "#4F46E5" if theme == "light" else "#818CF8"
        
        if sel_count == 0:
            self.status_info_lbl.setText("No cell selected. Click a cell to edit, or click '➕ New Cell'.")
            self.status_info_lbl.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color}; margin-left: 4px;")
            self.delete_btn.setEnabled(False)
            self.merge_btn.setEnabled(False)
        elif sel_count == 1:
            sel_id = self.canvas.selected_label_id
            if self.canvas.working_mask is not None:
                pixel_count = np.sum(self.canvas.working_mask == sel_id)
            else:
                pixel_count = 0
                
            if pixel_count == 0:
                self.status_info_lbl.setText(f"Active Cell: #{sel_id} (Empty - Paint on canvas)")
            else:
                self.status_info_lbl.setText(f"Active Cell: #{sel_id} (Size: {pixel_count} px)")
            self.status_info_lbl.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color}; margin-left: 4px;")
            self.delete_btn.setEnabled(True)
            self.merge_btn.setEnabled(False)
        else:
            self.status_info_lbl.setText(f"Multi-Selection Active: {sel_count} cells selected.")
            self.status_info_lbl.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color}; margin-left: 4px;")
            self.delete_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)

    def has_unsaved_changes(self) -> bool:
        return self.canvas.working_mask is not None and not np.array_equal(self.canvas.working_mask, self.original_mask)

    def closeEvent(self, event):
        if self.has_unsaved_changes():
            from PySide6.QtWidgets import QMessageBox
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Unsaved Changes")
            msg_box.setText("You have unsaved manual mask modifications.\nDo you want to save changes before closing?")
            msg_box.setIcon(QMessageBox.Question)
            
            save_btn = msg_box.addButton("Save", QMessageBox.AcceptRole)
            discard_btn = msg_box.addButton("Discard", QMessageBox.DestructiveRole)
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            
            msg_box.setDefaultButton(save_btn)
            msg_box.exec()
            
            clicked = msg_box.clickedButton()
            if clicked == save_btn:
                self.accept()
            elif clicked == discard_btn:
                self.canvas.working_mask = self.original_mask.copy()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def reject(self):
        if self.has_unsaved_changes():
            from PySide6.QtWidgets import QMessageBox
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Unsaved Changes")
            msg_box.setText("You have unsaved manual mask modifications.\nDo you want to save changes before closing?")
            msg_box.setIcon(QMessageBox.Question)
            
            save_btn = msg_box.addButton("Save", QMessageBox.AcceptRole)
            discard_btn = msg_box.addButton("Discard", QMessageBox.DestructiveRole)
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            
            msg_box.setDefaultButton(save_btn)
            msg_box.exec()
            
            clicked = msg_box.clickedButton()
            if clicked == save_btn:
                self.accept()
            elif clicked == discard_btn:
                self.canvas.working_mask = self.original_mask.copy()
                super().reject()
            else:
                # Cancel (remain in editor)
                pass
        else:
            super().reject()

    def _sync_theme(self):
        theme = theme_service.current_theme
        self._update_selection_status()
        
        if theme == "light":
            self.setStyleSheet("""
                QDialog { background-color: #F3F4F6; }
                QLabel { color: #1F2937; }
            """)
            self.toolbar.setStyleSheet("background-color: #FFFFFF; border: 1px solid #D1D5DB; border-radius: 6px; padding: 8px;")
            self.canvas.setStyleSheet("background-color: #E5E7EB; border: 1px solid #D1D5DB; border-radius: 6px;")

            button_style = """
                QPushButton {
                    padding: 6px 12px;
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    color: #374151;
                    border-radius: 4px;
                    font-weight: 500;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #F9FAFB; }
            """
            self.undo_btn.setStyleSheet(button_style)
            self.redo_btn.setStyleSheet(button_style)
            self.new_cell_btn.setStyleSheet(button_style)
            self.reset_btn.setStyleSheet(button_style)
            self.merge_btn.setStyleSheet(button_style)

            self.delete_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #FFFFFF;
                    border: 1px solid #FCA5A5;
                    color: #DC2626;
                    border-radius: 4px;
                    font-weight: 500;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #FEF2F2; }
                QPushButton:disabled {
                    background-color: #F9FAFB;
                    border: 1px solid #E5E7EB;
                    color: #9CA3AF;
                }
            """)

            self.pointer_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    color: #374151;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #6B7280;
                    color: #FFFFFF;
                    border: 1px solid #4B5563;
                }
            """)

            self.add_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    color: #374151;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #4F46E5;
                    color: #FFFFFF;
                    border: 1px solid #4338CA;
                }
            """)
            self.erase_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    color: #374151;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #EF4444;
                    color: #FFFFFF;
                    border: 1px solid #DC2626;
                }
            """)

            self.cancel_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 16px;
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    color: #4B5563;
                    border-radius: 4px;
                    font-weight: 600;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #F3F4F6; }
            """)

            self.save_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 16px;
                    background-color: #4F46E5;
                    border: 1px solid #4338CA;
                    color: #FFFFFF;
                    border-radius: 4px;
                    font-weight: 600;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #4338CA; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background-color: #0B0B0D; }
                QLabel { color: #E5E7EB; }
            """)
            self.toolbar.setStyleSheet("background-color: #1C1C22; border: 1px solid #2B2B35; border-radius: 6px; padding: 8px;")
            self.canvas.setStyleSheet("background-color: #050507; border: 1px solid #2B2B35; border-radius: 6px;")

            button_style = """
                QPushButton {
                    padding: 6px 12px;
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    color: #9CA3AF;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #2D2D37; color: #FFFFFF; }
            """
            self.undo_btn.setStyleSheet(button_style)
            self.redo_btn.setStyleSheet(button_style)
            self.new_cell_btn.setStyleSheet(button_style)
            self.reset_btn.setStyleSheet(button_style)
            self.merge_btn.setStyleSheet(button_style)

            self.delete_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #24242B;
                    border: 1px solid #7F1D1D;
                    color: #F87171;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #451A1A; color: #FFFFFF; }
                QPushButton:disabled {
                    background-color: #16161A;
                    border: 1px solid #222227;
                    color: #4B5563;
                }
            """)

            self.pointer_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    color: #9CA3AF;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #4B5563;
                    color: #FFFFFF;
                    border: 1px solid #374151;
                }
            """)

            self.add_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    color: #9CA3AF;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #6366F1;
                    color: #FFFFFF;
                    border: 1px solid #4F46E5;
                }
            """)
            self.erase_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    color: #9CA3AF;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #EF4444;
                    color: #FFFFFF;
                    border: 1px solid #DC2626;
                }
            """)

            self.cancel_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 16px;
                    background-color: #24242B;
                    border: 1px solid #2B2B35;
                    color: #D1D5DB;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #2D2D37; color: #FFFFFF; }
            """)

            self.save_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 16px;
                    background-color: #6366F1;
                    border: 1px solid #4F46E5;
                    color: #FFFFFF;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #4F46E5; }
            """)
