import os
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy, QButtonGroup
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Slot, Signal
from PySide6.QtGui import QIcon, QPixmap
from lumen.core.constants import (
    SIDEBAR_EXPANDED_WIDTH, SIDEBAR_COLLAPSED_WIDTH, ANIMATION_DURATION_MS, ICONS_DIR
)
from lumen.core.logger import logger
from lumen.core.config import config
from lumen.workflows.state import state
from lumen.core.services.navigation_service import navigation_service

class SidebarWidget(QFrame):
    """Collapsible navigation sidebar on the left side of the shell."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarWidget")
        
        self.collapsed = config.sidebar_collapsed
        self.nav_buttons = {}
        self.animation = None

        self._setup_ui()
        self._init_connections()
        
        # Apply initial collapse state immediately without animation
        self.apply_collapse_state(self.collapsed)

    def _setup_ui(self):
        # Set fixed size limits based on expanded/collapsed state
        initial_width = SIDEBAR_COLLAPSED_WIDTH if self.collapsed else SIDEBAR_EXPANDED_WIDTH
        self.setFixedWidth(initial_width)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 20, 16, 20)
        self.main_layout.setSpacing(12)

        # 1. Header (Title and subtitle)
        self.header_container = QFrame()
        self.header_layout = QVBoxLayout(self.header_container)
        self.header_layout.setContentsMargins(4, 0, 4, 0)
        self.header_layout.setSpacing(4)

        self.title_label = QLabel("Lumen")
        self.title_label.setObjectName("SidebarTitle")
        
        self.subtitle_label = QLabel("AI-Assisted Biological Image Analysis")
        self.subtitle_label.setObjectName("SidebarSubtitle")
        self.subtitle_label.setWordWrap(True)

        self.header_layout.addWidget(self.title_label)
        self.header_layout.addWidget(self.subtitle_label)
        self.main_layout.addWidget(self.header_container)

        # Spacer between title and buttons
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 2. Navigation Button Group
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Nav items map (id -> (label_text, svg_filename))
        nav_items = [
            ("home", ("Home", "home.svg")),
            ("upload", ("Upload Image", "upload.svg")),
            ("analysis", ("Analysis", "analysis.svg")),
            ("results", ("Results", "results.svg")),
            ("settings", ("Settings", "settings.svg"))
        ]

        for nav_id, (label, icon_file) in nav_items:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setProperty("nav_id", nav_id)
            btn.setCursor(Qt.PointingHandCursor)
            
            # Setup icon
            icon_path = os.path.join(str(ICONS_DIR), icon_file)
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
            
            btn.setText(label)
            btn.setToolTip(label)
            btn.setProperty("class", "SidebarNavButton")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            self.button_group.addButton(btn)
            self.main_layout.addWidget(btn)
            self.nav_buttons[nav_id] = btn

        # Flexible spacing push down
        self.main_layout.addStretch(1)

        # 3. Collapse Toggle Button at the Bottom
        self.collapse_btn = QPushButton()
        self.collapse_btn.setObjectName("SidebarCollapseButton")
        self.collapse_btn.setCursor(Qt.PointingHandCursor)
        self.update_collapse_button_icon()
        self.main_layout.addWidget(self.collapse_btn)

    def _init_connections(self):
        # Handle navigation clicks
        self.button_group.buttonClicked.connect(self._on_nav_clicked)
        # Handle collapse button click
        self.collapse_btn.clicked.connect(self.toggle_collapse)

        # Listen to state changes to update active button
        state.page_changed.connect(self._on_state_page_changed)

    def _on_nav_clicked(self, button):
        nav_id = button.property("nav_id")
        navigation_service.navigate_to(nav_id)

    @Slot(str)
    def _on_state_page_changed(self, page_name: str):
        if page_name in self.nav_buttons:
            # Check the button matching current page (triggers QButtonGroup exclusion)
            self.nav_buttons[page_name].setChecked(True)

    def update_collapse_button_icon(self):
        # Use simple text arrow indicators or SVGs
        # Left chevron for expanded (collapses sidebar), right chevron for collapsed
        arrow = "«" if not self.collapsed else "»"
        self.collapse_btn.setText(arrow)
        self.collapse_btn.setToolTip("Expand Sidebar" if self.collapsed else "Collapse Sidebar")

    def toggle_collapse(self):
        """Triggers collapsible animation width transition."""
        self.collapsed = not self.collapsed
        config.sidebar_collapsed = self.collapsed
        state.sidebar_collapsed = self.collapsed

        start_width = self.width()
        end_width = SIDEBAR_COLLAPSED_WIDTH if self.collapsed else SIDEBAR_EXPANDED_WIDTH

        # Hide text immediately when collapsing to prevent clipping
        if self.collapsed:
            self.set_text_visible(False)

        # Initialize animation
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(ANIMATION_DURATION_MS)
        self.animation.setStartValue(start_width)
        self.animation.setEndValue(end_width)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # Sync maximumWidth to prevent jumping boundaries
        self.animation.valueChanged.connect(self.set_width_sync)
        self.animation.finished.connect(self._on_animation_finished)
        self.animation.start()

    def apply_collapse_state(self, is_collapsed: bool):
        """Immediately applies collapse state without transition."""
        self.collapsed = is_collapsed
        target_width = SIDEBAR_COLLAPSED_WIDTH if is_collapsed else SIDEBAR_EXPANDED_WIDTH
        self.setFixedWidth(target_width)
        self.set_text_visible(not is_collapsed)
        self.update_collapse_button_icon()

    @Slot(int)
    def set_width_sync(self, val):
        self.setMaximumWidth(val)

    def _on_animation_finished(self):
        if not self.collapsed:
            self.set_text_visible(True)
        self.update_collapse_button_icon()
        logger.debug("Sidebar: Collapse transition completed. Collapsed: %s", self.collapsed)

    def set_text_visible(self, visible: bool):
        """Shows or hides button text labels and headers."""
        # Show/Hide header details
        self.title_label.setVisible(visible)
        self.subtitle_label.setVisible(visible)

        # Show/Hide text for navigation buttons
        for nav_id, btn in self.nav_buttons.items():
            if visible:
                # Restore button text
                label, _ = ("Home" if nav_id == "home" else 
                            "Upload Image" if nav_id == "upload" else 
                            "Analysis" if nav_id == "analysis" else 
                            "Results" if nav_id == "results" else "Settings"), None
                btn.setText(label)
            else:
                # Remove text, only show icon
                btn.setText("")
