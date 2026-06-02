from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QListWidget, QListWidgetItem, QSizePolicy, QGraphicsDropShadowEffect, QPushButton
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCursor, QColor
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.core.services.navigation_service import navigation_service

class QuickActionCard(QFrame):
    """Interactive card layout with hover highlights and elevation shadow effects."""
    
    clicked = Signal(str)

    def __init__(self, workflow_id: str, title: str, desc: str, symbol: str, parent=None):
        super().__init__(parent)
        self.workflow_id = workflow_id
        self.setObjectName("ActionCard")
        self.setProperty("class", "CardFrame")
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._setup_ui(title, desc, symbol)
        self._setup_shadow()

    def _setup_ui(self, title: str, desc: str, symbol: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        # Top layout: Symbol / Icon and Title
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        
        symbol_label = QLabel(symbol)
        symbol_label.setStyleSheet("font-size: 20px; color: #6366F1;")
        
        title_label = QLabel(title)
        title_label.setProperty("class", "CardTitle")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        top_layout.addWidget(symbol_label)
        top_layout.addWidget(title_label, 1)
        layout.addLayout(top_layout)

        # Description
        desc_label = QLabel(desc)
        desc_label.setProperty("class", "CardDesc")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

    def _setup_shadow(self):
        """Initializes a soft drop shadow effect matching modern design parameters."""
        self.shadow_effect = QGraphicsDropShadowEffect(self)
        self.shadow_effect.setBlurRadius(15)
        self.shadow_effect.setXOffset(0)
        self.shadow_effect.setYOffset(4)
        
        # Soft dark shadow default
        self.shadow_effect.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(self.shadow_effect)

    # Hover elevation overrides
    def enterEvent(self, event):
        # Slightly lighten background and change shadow to Indigo tint
        self.shadow_effect.setBlurRadius(25)
        self.shadow_effect.setYOffset(6)
        
        # Check current theme to draw appropriate shadow color
        theme = state.current_theme
        if theme == "light":
            self.shadow_effect.setColor(QColor(79, 70, 229, 20)) # Light indigo glow
        else:
            self.shadow_effect.setColor(QColor(99, 102, 241, 25))  # Dark indigo glow
            
        # Trigger redraw
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Restore standard soft shadow
        self.shadow_effect.setBlurRadius(15)
        self.shadow_effect.setYOffset(4)
        self.shadow_effect.setColor(QColor(0, 0, 0, 30))
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.workflow_id)
        super().mousePressEvent(event)


class HomePage(QWidget):
    """The central dashboard / landing page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setObjectName("PageContainer")
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # 1. Welcome Section
        welcome_frame = QFrame()
        welcome_layout = QVBoxLayout(welcome_frame)
        welcome_layout.setContentsMargins(0, 0, 0, 0)
        welcome_layout.setSpacing(4)

        title = QLabel("Lumen")
        title.setObjectName("PageTitle")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #FFFFFF;")

        subtitle = QLabel("AI-Assisted Biological Image Analysis Platform")
        subtitle.setObjectName("PageSubtitle")
        subtitle.setStyleSheet("font-size: 14px; color: #9CA3AF;")

        welcome_layout.addWidget(title)
        welcome_layout.addWidget(subtitle)
        main_layout.addWidget(welcome_frame)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #2B2B35; max-height: 1px; border: none;")
        main_layout.addWidget(line)

        # 2. Dynamic Current Session Panel (Hidden by default, shown when session active)
        self.session_card = QFrame()
        self.session_card.setObjectName("CurrentSessionCard")
        self.session_card.setStyleSheet("""
            #CurrentSessionCard {
                background-color: #1C1C22;
                border: 2px solid #4F46E5;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        session_layout = QHBoxLayout(self.session_card)
        session_layout.setSpacing(20)

        # Image Thumbnail Preview in vertical container
        self.session_thumb_container = QFrame()
        self.session_thumb_container.setFixedWidth(100)
        thumb_vbox = QVBoxLayout(self.session_thumb_container)
        thumb_vbox.setContentsMargins(0, 0, 0, 0)
        thumb_vbox.setSpacing(4)
        thumb_vbox.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.session_thumb = QLabel()
        self.session_thumb.setFixedSize(100, 100)
        self.session_thumb.setStyleSheet("background-color: #0B0B0D; border: 1px solid #2B2B35; border-radius: 6px;")
        self.session_thumb.setAlignment(Qt.AlignCenter)
        thumb_vbox.addWidget(self.session_thumb)

        self.session_contrast_lbl = QLabel("Auto Contrast Applied")
        self.session_contrast_lbl.setStyleSheet("font-size: 9px; color: #818CF8; font-weight: bold; background: transparent;")
        self.session_contrast_lbl.setAlignment(Qt.AlignCenter)
        thumb_vbox.addWidget(self.session_contrast_lbl)

        session_layout.addWidget(self.session_thumb_container)

        # Details Panel
        session_details = QVBoxLayout()
        session_details.setSpacing(4)
        session_details.setAlignment(Qt.AlignVCenter)

        session_hdr = QLabel("Current Active Session")
        session_hdr.setStyleSheet("font-size: 11px; font-weight: bold; color: #818CF8; text-transform: uppercase; letter-spacing: 0.5px;")
        
        self.session_fn = QLabel("Filename: -")
        self.session_fn.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        
        self.session_type = QLabel("Detected Type: -")
        self.session_type.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        
        self.session_wf = QLabel("Selected Workflow: -")
        self.session_wf.setStyleSheet("font-size: 11px; color: #9CA3AF;")

        session_details.addWidget(session_hdr)
        session_details.addWidget(self.session_fn)
        session_details.addWidget(self.session_type)
        session_details.addWidget(self.session_wf)
        session_layout.addLayout(session_details, 1)

        # Action CTAs
        session_actions = QVBoxLayout()
        session_actions.setSpacing(8)
        session_actions.setAlignment(Qt.AlignVCenter)

        self.continue_btn = QPushButton("Continue Analysis")
        self.continue_btn.setProperty("class", "PrimaryButton")
        self.continue_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        self.upload_new_btn = QPushButton("Upload New Image")
        self.upload_new_btn.setProperty("class", "SecondaryButton")
        self.upload_new_btn.setCursor(QCursor(Qt.PointingHandCursor))

        session_actions.addWidget(self.continue_btn)
        session_actions.addWidget(self.upload_new_btn)
        session_layout.addLayout(session_actions)

        main_layout.addWidget(self.session_card)
        self.session_card.setVisible(False)

        # Section Header: Quick Actions
        qa_header = QLabel("Quick Actions")
        qa_header.setStyleSheet("font-size: 15px; font-weight: 600; color: #F3F4F6;")
        main_layout.addWidget(qa_header)

        # 3. Quick Action Grid Cards
        grid_frame = QFrame()
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(16)

        self.cards = [
            QuickActionCard("cell_counting", "Cell Counting", "Detect, segment, and quantify cell nuclei or bodies.", "🧫", self),
            QuickActionCard("fluorescence", "Fluorescence Analysis", "Measure channel-specific signal intensity profiles.", "🧬", self),
            QuickActionCard("colony", "Colony Analysis", "Detect and analyze agar plate bacterial culture colonies.", "🧫", self),
            QuickActionCard("custom", "Custom Workflow", "Chain custom biological pre-processing pipeline blocks.", "⚙️", self)
        ]

        # Add to 2x2 grid
        grid_layout.addWidget(self.cards[0], 0, 0)
        grid_layout.addWidget(self.cards[1], 0, 1)
        grid_layout.addWidget(self.cards[2], 1, 0)
        grid_layout.addWidget(self.cards[3], 1, 1)
        main_layout.addWidget(grid_frame)

        # Section Header: Recent Files
        recent_header = QLabel("Recent Files")
        recent_header.setStyleSheet("font-size: 15px; font-weight: 600; color: #F3F4F6; margin-top: 10px;")
        main_layout.addWidget(recent_header)

        # 4. Recent Projects List
        self.recent_list = QListWidget()
        self.recent_list.setObjectName("RecentList")
        self.recent_list.setMinimumHeight(120)
        self.recent_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.recent_list.setStyleSheet("""
            QListWidget {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item {
                background-color: #24242B;
                border-radius: 4px;
                padding: 10px 14px;
                margin-bottom: 6px;
                color: #E5E7EB;
            }
            QListWidget::item:hover {
                background-color: #2E2E38;
                color: #FFFFFF;
            }
        """)

        # Add mock project paths
        mock_recents = [
            "C:/Microscopy/ProjectA/DAPI_channel_01.tif",
            "C:/Microscopy/ProjectA/FITC_channel_01.tif",
            "C:/CellTests/2026-06-01/Plate_well_B4_brightfield.png"
        ]
        for path in mock_recents:
            item = QListWidgetItem(path)
            self.recent_list.addItem(item)
            
        main_layout.addWidget(self.recent_list)

        # Setup interactions
        for card in self.cards:
            card.clicked.connect(self._on_card_clicked)

        self.recent_list.itemClicked.connect(self._on_recent_item_clicked)
        
        # Connect CTAs for session state
        self.continue_btn.clicked.connect(self._on_continue_clicked)
        self.upload_new_btn.clicked.connect(self._on_upload_new_clicked)
        
        # Monitor changes to reload sessions dynamically
        state.image_loaded.connect(self._on_image_loaded)
        state.workflow_selected.connect(self._on_workflow_selected)
        state.theme_changed.connect(self._sync_theme)
        
        # Check current state on load
        self._on_image_loaded(state.current_image_path)

    def _on_card_clicked(self, workflow_id: str):
        state.current_workflow = workflow_id
        navigation_service.navigate_to("upload")

    def _on_recent_item_clicked(self, item):
        path = item.text()
        logger.info("Home: Recent file path double clicked: %s", path)
        from lumen.processing.image_manager import image_manager
        success, msg = image_manager.load_image(path)
        if success:
            navigation_service.navigate_to("analysis")
        else:
            logger.warning("Home: Failed to load recent item directly: %s", msg)
            navigation_service.navigate_to("upload")
            state.current_image_path = path

    def _on_continue_clicked(self):
        navigation_service.navigate_to("analysis")

    def _on_upload_new_clicked(self):
        state.reset_session()
        navigation_service.navigate_to("upload")

    @Slot(str)
    def _on_image_loaded(self, path: str):
        """Shows or hides Current Session card based on active session existence."""
        if path:
            meta = state.current_image_metadata
            if meta:
                self.session_fn.setText(f"File: {meta.get('filename', '-')}")
                self.session_type.setText(f"Detected Type: {meta.get('classification', '-')}")
                
                # Check active workflow
                self._on_workflow_selected(state.current_workflow)
                
                # Render preview
                from lumen.processing.image_manager import image_manager
                pixmap = image_manager.get_thumbnail(100, 100)
                if pixmap:
                    self.session_thumb.setPixmap(pixmap)
                    self.session_thumb.setText("")
                else:
                    self.session_thumb.setText("No Preview")
                
                self.session_card.setVisible(True)
                return
        self.session_card.setVisible(False)

    @Slot(str)
    def _on_workflow_selected(self, wf_id: str):
        from lumen.workflows.workflow_manager import workflow_manager
        wf = workflow_manager.get_workflow(wf_id)
        if wf:
            self.session_wf.setText(f"Selected Workflow: {wf.name}")
            self.session_wf.setStyleSheet("font-size: 11px; color: #34D399; font-weight: bold;") # Highlighted green
        else:
            self.session_wf.setText("Selected Workflow: None Selected")
            self.session_wf.setStyleSheet("font-size: 11px; color: #9CA3AF;")

    @Slot(str)
    def _sync_theme(self, theme_name: str = ""):
        theme = theme_service.current_theme
        if theme == "light":
            self.session_card.setStyleSheet("""
                #CurrentSessionCard {
                    background-color: #FFFFFF;
                    border: 2px solid #4F46E5;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.session_thumb.setStyleSheet("background-color: #F9FAFB; border: 1px solid #D1D5DB; border-radius: 6px;")
            self.session_contrast_lbl.setStyleSheet("font-size: 9px; color: #4F46E5; font-weight: bold; background: transparent;")
            self.session_fn.setStyleSheet("font-size: 14px; font-weight: bold; color: #111827;")
            self.session_type.setStyleSheet("font-size: 11px; color: #4B5563;")
            
            # Check selected workflow color
            if state.current_workflow:
                self.session_wf.setStyleSheet("font-size: 11px; color: #059669; font-weight: bold;")
            else:
                self.session_wf.setStyleSheet("font-size: 11px; color: #4B5563;")
                
            self.recent_list.setStyleSheet("""
                QListWidget {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 8px;
                }
                QListWidget::item {
                    background-color: #F3F4F6;
                    border-radius: 4px;
                    padding: 10px 14px;
                    margin-bottom: 6px;
                    color: #4B5563;
                }
                QListWidget::item:hover {
                    background-color: #E5E7EB;
                    color: #111827;
                }
            """)
        else:
            self.session_card.setStyleSheet("""
                #CurrentSessionCard {
                    background-color: #1C1C22;
                    border: 2px solid #4F46E5;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.session_thumb.setStyleSheet("background-color: #0B0B0D; border: 1px solid #2B2B35; border-radius: 6px;")
            self.session_contrast_lbl.setStyleSheet("font-size: 9px; color: #818CF8; font-weight: bold; background: transparent;")
            self.session_fn.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
            self.session_type.setStyleSheet("font-size: 11px; color: #9CA3AF;")
            
            if state.current_workflow:
                self.session_wf.setStyleSheet("font-size: 11px; color: #34D399; font-weight: bold;")
            else:
                self.session_wf.setStyleSheet("font-size: 11px; color: #9CA3AF;")
                
            self.recent_list.setStyleSheet("""
                QListWidget {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 8px;
                }
                QListWidget::item {
                    background-color: #24242B;
                    border-radius: 4px;
                    padding: 10px 14px;
                    margin-bottom: 6px;
                    color: #E5E7EB;
                }
                QListWidget::item:hover {
                    background-color: #2E2E38;
                    color: #FFFFFF;
                }
            """)

    def _on_card_clicked(self, workflow_id: str):
        state.current_workflow = workflow_id
        # Transition to Upload Page so they can select file for workflow
        navigation_service.navigate_to("upload")

    def _on_recent_item_clicked(self, item):
        path = item.text()
        logger.info("Home: Recent file path double clicked: %s", path)
        # Directly load item if valid
        from lumen.processing.image_manager import image_manager
        success, msg = image_manager.load_image(path)
        if success:
            navigation_service.navigate_to("analysis")
        else:
            logger.warning("Home: Failed to load recent item directly: %s", msg)
            # Default navigate to upload screen
            navigation_service.navigate_to("upload")
            state.current_image_path = path # Store path in state anyway
