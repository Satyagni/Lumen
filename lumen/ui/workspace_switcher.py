from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Slot
from lumen.core.services.navigation_service import navigation_service
from lumen.core.services.theme_service import theme_service

class WorkspaceSwitcher(QFrame):
    """Reusable workspace switcher toggle panel for Single Explorer and Batch Explorer."""

    def __init__(self, active_view: str, parent=None):
        """
        active_view: "single" or "batch"
        """
        super().__init__(parent)
        self.setObjectName("WorkspaceSwitcherFrame")
        self.active_view = active_view

        self._setup_ui()
        self._init_connections()
        self.sync_theme(theme_service.current_theme)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.single_btn = QPushButton("Single Image Explorer")
        self.single_btn.setCheckable(True)
        self.single_btn.setChecked(self.active_view == "single")
        self.single_btn.setCursor(Qt.PointingHandCursor)

        self.batch_btn = QPushButton("Batch Results Explorer")
        self.batch_btn.setCheckable(True)
        self.batch_btn.setChecked(self.active_view == "batch")
        self.batch_btn.setCursor(Qt.PointingHandCursor)

        layout.addWidget(self.single_btn)
        layout.addWidget(self.batch_btn)
        layout.addStretch(1)

    def _init_connections(self):
        self.single_btn.clicked.connect(self._on_single_clicked)
        self.batch_btn.clicked.connect(self._on_batch_clicked)

    def _on_single_clicked(self):
        if self.active_view != "single":
            navigation_service.navigate_to("analysis")

    def _on_batch_clicked(self):
        if self.active_view != "batch":
            navigation_service.navigate_to("batch_explorer")

    def sync_theme(self, theme_name: str):
        if theme_name == "light":
            self.setStyleSheet("""
                #WorkspaceSwitcherFrame {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 6px;
                    padding: 4px;
                }
            """)
            if self.active_view == "single":
                self.single_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4F46E5;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #FFFFFF;
                        font-size: 11px;
                        font-weight: bold;
                    }
                """)
                self.batch_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #4B5563;
                        font-size: 11px;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        color: #111827;
                        background-color: #F3F4F6;
                    }
                """)
            else:
                self.single_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #4B5563;
                        font-size: 11px;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        color: #111827;
                        background-color: #F3F4F6;
                    }
                """)
                self.batch_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4F46E5;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #FFFFFF;
                        font-size: 11px;
                        font-weight: bold;
                    }
                """)
        else:
            self.setStyleSheet("""
                #WorkspaceSwitcherFrame {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 6px;
                    padding: 4px;
                }
            """)
            if self.active_view == "single":
                self.single_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #312E81;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #FFFFFF;
                        font-size: 11px;
                        font-weight: bold;
                    }
                """)
                self.batch_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #9CA3AF;
                        font-size: 11px;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        color: #FFFFFF;
                        background-color: #24242B;
                    }
                """)
            else:
                self.single_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #9CA3AF;
                        font-size: 11px;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        color: #FFFFFF;
                        background-color: #24242B;
                    }
                """)
                self.batch_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #312E81;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 16px;
                        color: #FFFFFF;
                        font-size: 11px;
                        font-weight: bold;
                    }
                """)
