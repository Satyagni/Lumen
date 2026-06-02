import os
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PySide6.QtCore import Qt, Slot
from lumen.workflows.state import state
from lumen.core.services.theme_service import theme_service
from lumen.core.services.gpu_service import gpu_service
from lumen.workflows.workflow_manager import workflow_manager
from lumen.core.logger import logger

class NavbarWidget(QFrame):
    """Header bar providing contextual indicators for active files, workflows, and hardware."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NavbarWidget")
        self.setFixedHeight(50)

        self._setup_ui()
        self._init_connections()
        self._sync_all_states()

    def _setup_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 0, 16, 0)
        self.layout.setSpacing(16)

        # 1. Page Title (Left)
        self.title_label = QLabel("Dashboard")
        self.title_label.setObjectName("NavbarTitle")
        self.title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #FFFFFF;")
        self.layout.addWidget(self.title_label)

        # Flex Spacer to push status indicators to the right
        self.layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # 2. Status Board Layout (Horizontal row of parameters)
        self.status_container = QFrame()
        self.status_layout = QHBoxLayout(self.status_container)
        self.status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_layout.setSpacing(12)

        # Image Status
        self.img_lbl_prefix = QLabel("Image:")
        self.img_lbl_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.img_val_lbl = QLabel("No Image Loaded")
        self.img_val_lbl.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: 500;")
        
        self.status_layout.addWidget(self.img_lbl_prefix)
        self.status_layout.addWidget(self.img_val_lbl)

        # Separator 1
        self.sep1 = QLabel("|")
        self.sep1.setStyleSheet("color: #2B2B35; font-size: 13px; font-weight: 300;")
        self.status_layout.addWidget(self.sep1)

        # Workflow Status
        self.wf_lbl_prefix = QLabel("Workflow:")
        self.wf_lbl_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.wf_val_lbl = QLabel("None Selected")
        self.wf_val_lbl.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: 500;")
        
        self.status_layout.addWidget(self.wf_lbl_prefix)
        self.status_layout.addWidget(self.wf_val_lbl)

        # Separator 2
        self.sep2 = QLabel("|")
        self.sep2.setStyleSheet("color: #2B2B35; font-size: 13px; font-weight: 300;")
        self.status_layout.addWidget(self.sep2)

        # GPU Backend Status
        self.gpu_lbl_prefix = QLabel("Backend:")
        self.gpu_lbl_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        
        from PySide6.QtWidgets import QComboBox
        from PySide6.QtGui import QCursor
        self.gpu_combo = QComboBox()
        self.gpu_combo.setCursor(QCursor(Qt.PointingHandCursor))
        self.gpu_combo.addItems(["Auto", "CUDA (GPU)", "CPU"])
        
        # Disable CUDA option if not available on machine
        if not gpu_service.is_cuda_available:
            self.gpu_combo.setItemText(1, "CUDA (Unavailable)")
            self.gpu_combo.model().item(1).setEnabled(False)
            
        self.gpu_active_lbl = QLabel()
        self.gpu_active_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
        
        self.status_layout.addWidget(self.gpu_lbl_prefix)
        self.status_layout.addWidget(self.gpu_combo)
        self.status_layout.addWidget(self.gpu_active_lbl)
        
        self.layout.addWidget(self.status_container)

        # Separator 3
        self.sep3 = QLabel("|")
        self.sep3.setStyleSheet("color: #2B2B35; font-size: 13px; font-weight: 300;")
        self.layout.addWidget(self.sep3)

        # 3. Theme Toggle Button (Right)
        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("ThemeToggleButton")
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.update_theme_button_text()
        self.layout.addWidget(self.theme_btn)

    def _init_connections(self):
        # Click handler
        self.theme_btn.clicked.connect(self._on_theme_toggle_clicked)
        
        # Combo selector handler
        self.gpu_combo.currentIndexChanged.connect(self._on_backend_pref_combo_changed)
 
        # AppState synchronization
        state.page_changed.connect(self._on_page_changed)
        state.image_loaded.connect(self._on_image_loaded)
        state.workflow_selected.connect(self._on_workflow_selected)
        state.theme_changed.connect(self._on_theme_changed)
        state.backend_preference_changed.connect(self._on_state_backend_pref_changed)

    def _sync_all_states(self):
        """Pre-populates values from current state on initialization."""
        self._on_page_changed(state.current_page)
        self._on_image_loaded(state.current_image_path)
        self._on_workflow_selected(state.current_workflow)
        self._on_theme_changed(theme_service.current_theme)
        self._on_state_backend_pref_changed(state.backend_preference)

    def update_theme_button_text(self):
        theme = theme_service.current_theme
        symbol = "☀️" if theme == "dark" else "🌙"
        self.theme_btn.setText(symbol)
        self.theme_btn.setToolTip("Switch to Light Mode" if theme == "dark" else "Switch to Dark Mode")

    def _on_theme_toggle_clicked(self):
        theme_service.toggle_theme()

    @Slot(str)
    def _on_page_changed(self, page_name: str):
        titles = {
            "home": "Dashboard",
            "upload": "File Upload",
            "analysis": "Image Analysis",
            "results": "Quantification Results",
            "batch_explorer": "Batch Results Explorer",
            "settings": "System Settings"
        }
        display_title = titles.get(page_name, "Lumen")
        self.title_label.setText(display_title)

    @Slot(str)
    def _on_image_loaded(self, path: str):
        if path:
            filename = os.path.basename(path)
            self.img_val_lbl.setText(filename)
            self.img_val_lbl.setToolTip(path)
            self.img_val_lbl.setStyleSheet("color: #6366F1; font-size: 12px; font-weight: 600;") # Indigo highlight when loaded
        else:
            self.img_val_lbl.setText("No Image Loaded")
            self.img_val_lbl.setToolTip("")
            self.img_val_lbl.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: 500;")

    @Slot(str)
    def _on_workflow_selected(self, wf_id: str):
        wf = workflow_manager.get_workflow(wf_id)
        if wf:
            self.wf_val_lbl.setText(wf.name)
            self.wf_val_lbl.setStyleSheet("color: #34D399; font-size: 12px; font-weight: 600;") # Green highlight when selected
        else:
            self.wf_val_lbl.setText("None Selected")
            self.wf_val_lbl.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: 500;")

    @Slot(str)
    def _on_theme_changed(self, theme_name: str):
        self.update_theme_button_text()
        
        # Color adjustments for light theme
        if theme_name == "light":
            self.title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #111827;")
            self.img_lbl_prefix.setStyleSheet("color: #9CA3AF; font-size: 11px; font-weight: bold; text-transform: uppercase;")
            self.wf_lbl_prefix.setStyleSheet("color: #9CA3AF; font-size: 11px; font-weight: bold; text-transform: uppercase;")
            self.gpu_lbl_prefix.setStyleSheet("color: #9CA3AF; font-size: 11px; font-weight: bold; text-transform: uppercase;")
            
            # Reset values if not highlighted
            if not state.current_image_path:
                self.img_val_lbl.setStyleSheet("color: #4B5563; font-size: 12px; font-weight: 500;")
            else:
                self.img_val_lbl.setStyleSheet("color: #4F46E5; font-size: 12px; font-weight: 600;")
                
            if not state.current_workflow:
                self.wf_val_lbl.setStyleSheet("color: #4B5563; font-size: 12px; font-weight: 500;")
            else:
                self.wf_val_lbl.setStyleSheet("color: #059669; font-size: 12px; font-weight: 600;")
                
            self.gpu_combo.setStyleSheet("""
                QComboBox {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 4px;
                    padding: 2px 24px 2px 8px;
                    color: #4F46E5;
                    font-size: 11px;
                    font-weight: bold;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 14px;
                }
            """)
                
            self.sep1.setStyleSheet("color: #E5E7EB; font-size: 13px; font-weight: 300;")
            self.sep2.setStyleSheet("color: #E5E7EB; font-size: 13px; font-weight: 300;")
            self.sep3.setStyleSheet("color: #E5E7EB; font-size: 13px; font-weight: 300;")
        else:
            self.title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #FFFFFF;")
            self.img_lbl_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
            self.wf_lbl_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
            self.gpu_lbl_prefix.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: bold; text-transform: uppercase;")
            
            if not state.current_image_path:
                self.img_val_lbl.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: 500;")
            else:
                self.img_val_lbl.setStyleSheet("color: #6366F1; font-size: 12px; font-weight: 600;")
                
            if not state.current_workflow:
                self.wf_val_lbl.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: 500;")
            else:
                self.wf_val_lbl.setStyleSheet("color: #34D399; font-size: 12px; font-weight: 600;")
                
            self.gpu_combo.setStyleSheet("""
                QComboBox {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 4px;
                    padding: 2px 24px 2px 8px;
                    color: #818CF8;
                    font-size: 11px;
                    font-weight: bold;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 14px;
                }
            """)
                
            self.sep1.setStyleSheet("color: #2B2B35; font-size: 13px; font-weight: 300;")
            self.sep2.setStyleSheet("color: #2B2B35; font-size: 13px; font-weight: 300;")
            self.sep3.setStyleSheet("color: #2B2B35; font-size: 13px; font-weight: 300;")

    def _on_backend_pref_combo_changed(self, index: int):
        mapping = {0: "Auto", 1: "CUDA (GPU)", 2: "CPU"}
        pref_val = mapping.get(index, "Auto")
        state.backend_preference = pref_val
        self._update_gpu_active_label()

    @Slot(str)
    def _on_state_backend_pref_changed(self, val: str):
        mapping = {"Auto": 0, "CUDA (GPU)": 1, "CUDA": 1, "CPU": 2}
        idx = mapping.get(val, 0)
        self.gpu_combo.blockSignals(True)
        self.gpu_combo.setCurrentIndex(idx)
        self.gpu_combo.blockSignals(False)
        self._update_gpu_active_label()

    def _update_gpu_active_label(self):
        pref = state.backend_preference
        from lumen.core.services.gpu_service import gpu_service
        use_gpu, resolved_backend_name = gpu_service.resolve_execution_backend(pref)
        
        # update current active backend state
        state.current_backend = resolved_backend_name
        
        # Determine label text and style
        if "fallback" in resolved_backend_name.lower():
            self.gpu_active_lbl.setText("Running: CPU (fallback)")
            self.gpu_active_lbl.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: bold; margin-left: 4px;") # Amber warning
        elif resolved_backend_name == "CUDA":
            self.gpu_active_lbl.setText("Running: CUDA active")
            self.gpu_active_lbl.setStyleSheet("color: #34D399; font-size: 11px; font-weight: bold; margin-left: 4px;") # Green active
        else:
            self.gpu_active_lbl.setText("Running: CPU active")
            self.gpu_active_lbl.setStyleSheet("color: #9CA3AF; font-size: 11px; font-weight: 500; margin-left: 4px;") # Grey muted
