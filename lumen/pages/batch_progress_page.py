import os
import time
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QProgressBar, QGridLayout, QScrollArea, QSizePolicy, QSplitter
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QCursor
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.core.services.theme_service import theme_service
from lumen.core.services.navigation_service import navigation_service
from lumen.processing.batch_manager import batch_manager

class BatchProgressPage(QWidget):
    """Execution dashboard for multi-image batch analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_images = 0
        self.results_dir = ""
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_ui_timer)
        self._setup_ui()
        self._init_connections()
        self._sync_theme()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setObjectName("PageContainer")
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # Page Header
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self.title_lbl = QLabel("Batch Analysis Pipeline")
        self.title_lbl.setObjectName("PageTitle")
        
        self.subtitle_lbl = QLabel("Monitoring folder-based microscopy image quantification")
        self.subtitle_lbl.setObjectName("PageSubtitle")

        header_layout.addWidget(self.title_lbl)
        header_layout.addWidget(self.subtitle_lbl)
        main_layout.addWidget(header_frame)

        # Divider line
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setStyleSheet("background-color: #2B2B35; max-height: 1px; border: none;")
        main_layout.addWidget(self.line)

        # 1. Main Progress Dashboard Card
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setObjectName("BatchSplitter")
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dashboard_card = QFrame()
        self.dashboard_card.setObjectName("DashboardCard")
        self.dashboard_card.setStyleSheet("""
            #DashboardCard {
                background-color: #1C1C22;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        dash_layout = QVBoxLayout(self.dashboard_card)
        dash_layout.setSpacing(16)

        # Status & Index Label
        self.status_lbl = QLabel("Initializing batch...")
        self.status_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        dash_layout.addWidget(self.status_lbl)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2B2B35;
                border-radius: 6px;
                background-color: #131317;
                text-align: center;
                color: #FFFFFF;
                font-weight: bold;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #4F46E5;
                border-radius: 5px;
            }
        """)
        dash_layout.addWidget(self.progress_bar)

        # Metrics Summary Grid (Completed, Failed, Skipped, Remaining, Backend)
        self.stats_frame = QFrame()
        grid = QGridLayout(self.stats_frame)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        lbl_style = "font-size: 11px; color: #9CA3AF; font-weight: bold; text-transform: uppercase;"
        val_style = "font-size: 14px; font-weight: bold; color: #FFFFFF;"

        # Column 1
        t_title = QLabel("Total Images:")
        t_title.setStyleSheet(lbl_style)
        self.total_val = QLabel("0")
        self.total_val.setStyleSheet(val_style)
        grid.addWidget(t_title, 0, 0)
        grid.addWidget(self.total_val, 0, 1)

        c_title = QLabel("Completed:")
        c_title.setStyleSheet(lbl_style)
        self.completed_val = QLabel("0")
        self.completed_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #34D399;") # Green
        grid.addWidget(c_title, 1, 0)
        grid.addWidget(self.completed_val, 1, 1)

        # Column 2
        f_title = QLabel("Failed:")
        f_title.setStyleSheet(lbl_style)
        self.failed_val = QLabel("0")
        self.failed_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #F87171;") # Red
        grid.addWidget(f_title, 0, 2)
        grid.addWidget(self.failed_val, 0, 3)

        s_title = QLabel("Skipped:")
        s_title.setStyleSheet(lbl_style)
        self.skipped_val = QLabel("0")
        self.skipped_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #F59E0B;") # Amber
        grid.addWidget(s_title, 1, 2)
        grid.addWidget(self.skipped_val, 1, 3)

        # Column 3
        be_title = QLabel("Backend Mode:")
        be_title.setStyleSheet(lbl_style)
        self.backend_val = QLabel("-")
        self.backend_val.setStyleSheet(val_style)
        grid.addWidget(be_title, 0, 4)
        grid.addWidget(self.backend_val, 0, 5)

        self.rem_title = QLabel("Est. Remaining:")
        self.rem_title.setStyleSheet(lbl_style)
        self.rem_val = QLabel("~0 mins")
        self.rem_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #818CF8;")
        grid.addWidget(self.rem_title, 1, 4)
        grid.addWidget(self.rem_val, 1, 5)

        # Column 4
        el_title = QLabel("Elapsed Time:")
        el_title.setStyleSheet(lbl_style)
        self.elapsed_val = QLabel("00:00:00")
        self.elapsed_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #818CF8;")
        grid.addWidget(el_title, 0, 6)
        grid.addWidget(self.elapsed_val, 0, 7)

        dash_layout.addWidget(self.stats_frame)
        self.splitter.addWidget(self.dashboard_card)

        # 2. Live Run Log Console (Aesthetics - visual feedback of files being processed)
        self.log_card = QFrame()
        self.log_card.setObjectName("LogCard")
        self.log_card.setStyleSheet("""
            #LogCard {
                background-color: #131317;
                border: 1px solid #2B2B35;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        log_vbox = QVBoxLayout(self.log_card)
        log_vbox.setSpacing(6)

        log_title = QLabel("Live Execution Log")
        log_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;")
        log_vbox.addWidget(log_title)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: transparent;")
        
        self.log_content = QWidget()
        self.log_content.setStyleSheet("background: transparent;")
        self.log_list_layout = QVBoxLayout(self.log_content)
        self.log_list_layout.setContentsMargins(0, 0, 0, 0)
        self.log_list_layout.setSpacing(4)
        self.log_list_layout.addStretch(1) # Pin entries to the top
        
        self.scroll_area.setWidget(self.log_content)
        log_vbox.addWidget(self.scroll_area)
        self.splitter.addWidget(self.log_card)

        main_layout.addWidget(self.splitter)

        # Configure splitter behavior and sizing
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, True)
        self.splitter.setSizes([200, 300])

        # 3. Actions Row (Pause & Cancel Buttons during execution)
        self.actions_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause Batch")
        self.pause_btn.setProperty("class", "SecondaryButton")
        self.pause_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.actions_layout.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Cancel Batch")
        self.cancel_btn.setProperty("class", "PrimaryButton")
        self.cancel_btn.setStyleSheet("background-color: #DC2626;") # Red cancel button
        self.cancel_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.actions_layout.addWidget(self.cancel_btn)
        self.actions_layout.addStretch(1)
        main_layout.addLayout(self.actions_layout)

        # 4. Completion Summary Card (Hidden by default, shown upon end)
        self.completion_card = QFrame()
        self.completion_card.setObjectName("CompletionCard")
        self.completion_card.setStyleSheet("""
            #CompletionCard {
                background-color: #1C1C22;
                border: 2px solid #34D399;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        comp_vbox = QVBoxLayout(self.completion_card)
        comp_vbox.setSpacing(12)

        header_comp = QHBoxLayout()
        header_comp.setSpacing(10)
        
        self.check_icon = QLabel("✓")
        self.check_icon.setStyleSheet("font-size: 24px; font-weight: bold; color: #34D399; background: transparent;")
        self.comp_title = QLabel("Batch Execution Finished")
        self.comp_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        
        header_comp.addWidget(self.check_icon)
        header_comp.addWidget(self.comp_title)
        header_comp.addStretch(1)
        comp_vbox.addLayout(header_comp)

        self.comp_desc = QLabel("All microscopy files processed. Outputs generated in output results directory.")
        self.comp_desc.setStyleSheet("font-size: 12px; color: #9CA3AF;")
        comp_vbox.addWidget(self.comp_desc)

        comp_btns_layout = QHBoxLayout()
        comp_btns_layout.setSpacing(12)

        self.review_results_btn = QPushButton("Review Results")
        self.review_results_btn.setProperty("class", "PrimaryButton")
        self.review_results_btn.setCursor(QCursor(Qt.PointingHandCursor))

        self.open_dir_btn = QPushButton("Open Results Folder")
        self.open_dir_btn.setProperty("class", "SecondaryButton")
        self.open_dir_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        self.back_btn = QPushButton("Return to Upload")
        self.back_btn.setProperty("class", "SecondaryButton")
        self.back_btn.setCursor(QCursor(Qt.PointingHandCursor))

        comp_btns_layout.addWidget(self.review_results_btn)
        comp_btns_layout.addWidget(self.open_dir_btn)
        comp_btns_layout.addWidget(self.back_btn)
        comp_btns_layout.addStretch(1)
        comp_vbox.addLayout(comp_btns_layout)

        main_layout.addWidget(self.completion_card)
        self.completion_card.setVisible(False)

        # Spacing fill (removed stretch to allow splitter to expand)
        pass

    def _init_connections(self):
        # State signals
        state.theme_changed.connect(self._sync_theme)
        
        # Connect to batch manager signals
        batch_manager.batch_started.connect(self._on_batch_started)
        batch_manager.batch_progress_updated.connect(self._on_batch_progress_updated)
        batch_manager.batch_finished.connect(self._on_batch_finished)
        batch_manager.batch_cancelled.connect(self._on_batch_cancelled)
        batch_manager.batch_paused.connect(self._on_batch_paused)
        batch_manager.batch_resumed.connect(self._on_batch_resumed)

        # Local action buttons
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.review_results_btn.clicked.connect(self._on_review_clicked)
        self.open_dir_btn.clicked.connect(self._on_open_dir_clicked)
        self.back_btn.clicked.connect(self._on_back_clicked)

    def clear_logs(self):
        """Clears the live log console widget children."""
        # Loop backwards deleting everything except the stretch space
        layout = self.log_list_layout
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def add_log_entry(self, filename: str, status: str):
        """Appends a row to the log console with colored status tags."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(8)

        # Filename
        name_lbl = QLabel(filename)
        name_lbl.setStyleSheet("font-size: 11px; font-family: monospace; color: #E5E7EB;")
        row_layout.addWidget(name_lbl)
        row_layout.addStretch(1)

        # Status Tag
        status_lbl = QLabel(status)
        tag_style = "font-size: 9px; font-weight: bold; padding: 2px 6px; border-radius: 4px; font-family: sans-serif;"
        
        if status == "SUCCESS":
            tag_style += "background-color: #065F46; color: #34D399;"
        elif status == "SKIPPED_ALREADY_EXISTS":
            tag_style += "background-color: #78350F; color: #F59E0B;"
        else:
            tag_style += "background-color: #991B1B; color: #F87171;"
            
        status_lbl.setStyleSheet(tag_style)
        row_layout.addWidget(status_lbl)

        # Insert before the layout stretch at index 0 (if any) or just append
        self.log_list_layout.insertWidget(self.log_list_layout.count() - 1, row)
        
        # Scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    @Slot(int)
    def _on_batch_started(self, total: int):
        self.total_images = total
        self.results_dir = ""  # Reset results folder path
        self.total_val.setText(str(total))
        self.completed_val.setText("0")
        self.failed_val.setText("0")
        self.skipped_val.setText("0")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(total)
        self.status_lbl.setText(f"Initializing run: 0 of {total} images processed")
        
        # Read resolved execution backend from single source of truth in batch_manager
        self.backend_val.setText(batch_manager.resolved_backend)
        
        # Calculate time estimation
        self.rem_title.setVisible(True)
        self.rem_val.setVisible(True)
        self.rem_val.setText("Estimating...")
        self.elapsed_val.setText("00:00:00")

        # Hide completion panel, show progress dash & log
        self.completion_card.setVisible(False)
        self.dashboard_card.setVisible(True)
        self.log_card.setVisible(True)
        
        # Reset cancel and pause buttons completely to prevent leakage/zombie state
        self.cancel_btn.setText("Cancel Batch")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setVisible(True)
        
        self.pause_btn.setText("Pause Batch")
        self.pause_btn.setEnabled(True)
        self.pause_btn.setVisible(True)
        
        if batch_manager.lifecycle_state == "RUNNING" and not self.ui_timer.isActive():
            self.ui_timer.start(1000)
            
        self.clear_logs()

    @Slot(int, int, str)
    def _on_batch_progress_updated(self, completed: int, failed: int, current_image_name: str):
        processed = completed + failed
        self.progress_bar.setValue(processed)
        
        # Display only actually processed successful (non-skipped) images as "Completed" in the KPI
        success_completed = max(0, completed - batch_manager.skipped_count)
        self.completed_val.setText(str(success_completed))
        self.failed_val.setText(str(failed))
        self.skipped_val.setText(str(batch_manager.skipped_count))

        # Update remaining time
        self._update_remaining_time(processed)
        
        self.status_lbl.setText(f"Processing image {processed + 1} of {self.total_images}: {current_image_name}")

        # Add log entries for completed items. We can look at the latest summary record.
        if batch_manager.summary_records:
            latest = batch_manager.summary_records[-1]
            # Verify we haven't already logged this record (keep logs aligned with summary records size)
            log_count = self.log_list_layout.count() - 1
            if log_count < len(batch_manager.summary_records):
                self.add_log_entry(latest["image_name"], latest["status"])

    @Slot(int, int, str)
    def _on_batch_finished(self, completed: int, failed: int, results_dir: str):
        self.results_dir = results_dir
        self.progress_bar.setValue(self.total_images)
        self.status_lbl.setText("Batch execution complete!")
        
        # Force a final KPI refresh using finalized source of truth from batch_manager
        # Guarantee mathematical invariant completed + failed + skipped == total_images at finish
        skipped = batch_manager.skipped_count
        success_completed = max(0, self.total_images - failed - skipped)
        self.completed_val.setText(str(success_completed))
        self.failed_val.setText(str(failed))
        self.skipped_val.setText(str(skipped))
        
        # Make sure any final records are in the log console
        log_count = self.log_list_layout.count() - 1
        while log_count < len(batch_manager.summary_records):
            latest = batch_manager.summary_records[log_count]
            self.add_log_entry(latest["image_name"], latest["status"])
            log_count += 1
            
        self.sync_ui_state()

    @Slot()
    def _on_batch_cancelled(self):
        self.status_lbl.setText("Batch cancelled by user.")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelled")
        self.pause_btn.setEnabled(False)
        self.ui_timer.stop()
        
        # Force a final KPI refresh of attempted images on cancellation before navigating
        # Do NOT rewrite total_images (total_images represents all valid discovered images)
        completed = batch_manager.completed_count
        failed = batch_manager.failed_count
        skipped = batch_manager.skipped_count
        success_completed = max(0, completed - skipped)
        self.completed_val.setText(str(success_completed))
        self.failed_val.setText(str(failed))
        self.skipped_val.setText(str(skipped))
        
        # Return to upload page
        navigation_service.navigate_to("upload")

    def _on_pause_clicked(self):
        if batch_manager.lifecycle_state == "RUNNING":
            self.pause_btn.setText("Pausing...")
            self.pause_btn.setEnabled(False)
            batch_manager.pause_batch()
        elif batch_manager.lifecycle_state == "PAUSED":
            self.pause_btn.setText("Pause Batch")
            batch_manager.resume_batch()

    @Slot()
    def _on_batch_paused(self):
        self.sync_ui_state()

    @Slot()
    def _on_batch_resumed(self):
        self.sync_ui_state()

    def _on_cancel_clicked(self):
        self.cancel_btn.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.status_lbl.setText("Finishing current image safely...")
        batch_manager.cancel_batch()

    def _on_open_dir_clicked(self):
        if not self.results_dir:
            self.results_dir = batch_manager.output_dir or state.batch_results_dir
        if self.results_dir and os.path.exists(self.results_dir):
            try:
                os.startfile(self.results_dir)
            except Exception as e:
                logger.error("BatchProgressPage: Failed to open explorer folder: %s", e)

    def _on_review_clicked(self):
        batch_manager.lifecycle_state = "REVIEWED"
        if not self.results_dir:
            self.results_dir = batch_manager.output_dir or state.batch_results_dir
        state.batch_results_dir = self.results_dir
        navigation_service.navigate_to("batch_explorer")

    def _on_back_clicked(self):
        # Reset batch states & redirect to upload page
        batch_manager.reset_batch()
        navigation_service.navigate_to("upload")

    def _update_ui_timer(self):
        if batch_manager.lifecycle_state in ("RUNNING", "PAUSED", "COMPLETED"):
            self.elapsed_val.setText(batch_manager.get_elapsed_time_hhmmss())

    def sync_ui_state(self):
        state_name = batch_manager.lifecycle_state
        if state_name == "RUNNING":
            self.dashboard_card.setVisible(True)
            self.log_card.setVisible(True)
            self.completion_card.setVisible(False)
            
            self.cancel_btn.setVisible(True)
            self.cancel_btn.setEnabled(True)
            self.cancel_btn.setText("Cancel Batch")
            
            self.pause_btn.setVisible(True)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Pause Batch")
            
            self.progress_bar.setMaximum(len(batch_manager.image_paths))
            self.total_val.setText(str(len(batch_manager.image_paths)))
            self.total_images = len(batch_manager.image_paths)
            self.backend_val.setText(batch_manager.resolved_backend)
            
            if not self.ui_timer.isActive():
                self.ui_timer.start(1000)
        elif state_name == "PAUSED":
            self.dashboard_card.setVisible(True)
            self.log_card.setVisible(True)
            self.completion_card.setVisible(False)
            
            self.cancel_btn.setVisible(True)
            self.cancel_btn.setEnabled(True)
            self.cancel_btn.setText("Cancel Batch")
            
            self.pause_btn.setVisible(True)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Resume Batch")
            
            self.progress_bar.setMaximum(len(batch_manager.image_paths))
            self.total_val.setText(str(len(batch_manager.image_paths)))
            self.total_images = len(batch_manager.image_paths)
            self.backend_val.setText(batch_manager.resolved_backend)
            
            self.ui_timer.stop()
        elif state_name == "COMPLETED":
            self.results_dir = batch_manager.output_dir or state.batch_results_dir
            self.dashboard_card.setVisible(True)
            self.log_card.setVisible(True)
            self.completion_card.setVisible(True)
            self.cancel_btn.setVisible(False)
            self.pause_btn.setVisible(False)
            
            self.ui_timer.stop()
            
            skipped = batch_manager.skipped_count
            failed = batch_manager.failed_count
            success_completed = max(0, len(batch_manager.image_paths) - failed - skipped)
            
            self.comp_desc.setText(
                f"Lumen successfully finished folder analysis run.\n\n"
                f"Completed in {batch_manager.get_completed_time_str()}\n\n"
                f"Processed: {len(batch_manager.image_paths)} images\n"
                f"• Completed: {success_completed}\n"
                f"• Failed: {failed}\n"
                f"• Skipped (Resume-safe): {skipped}\n\n"
                f"Results compiled in: {batch_manager.output_dir}"
            )
        else: # IDLE or REVIEWED or CANCELLED
            self.dashboard_card.setVisible(False)
            self.log_card.setVisible(False)
            self.completion_card.setVisible(False)
            self.cancel_btn.setVisible(False)
            self.pause_btn.setVisible(False)
            self.ui_timer.stop()

        self._update_remaining_time()

    def _update_remaining_time(self, processed=None):
        state_name = batch_manager.lifecycle_state
        if state_name == "PAUSED":
            self.rem_title.setVisible(True)
            self.rem_val.setVisible(True)
            self.rem_val.setText("ETA: Paused")
        elif state_name == "COMPLETED":
            self.rem_title.setVisible(False)
            self.rem_val.setVisible(False)
        elif state_name == "RUNNING":
            if processed is None:
                processed = batch_manager.completed_count + batch_manager.failed_count
            if self.total_images > 0:
                remaining_images = max(0, self.total_images - processed)
                if remaining_images == 0:
                    self.rem_title.setVisible(True)
                    self.rem_val.setVisible(True)
                    self.rem_val.setText("0.0 mins")
                elif processed >= 3 and len(batch_manager.image_runtimes) >= 3:
                    self.rem_title.setVisible(True)
                    self.rem_val.setVisible(True)
                    avg_time_s = sum(batch_manager.image_runtimes) / len(batch_manager.image_runtimes)
                    rem_sec = avg_time_s * remaining_images
                    rem_min = max(0.1, round(rem_sec / 60.0, 1))
                    self.rem_val.setText(f"~{rem_min} mins")
                else:
                    self.rem_title.setVisible(True)
                    self.rem_val.setVisible(True)
                    self.rem_val.setText("Estimating...")
            else:
                self.rem_title.setVisible(True)
                self.rem_val.setVisible(True)
                self.rem_val.setText("Estimating...")
        else:
            self.rem_title.setVisible(False)
            self.rem_val.setVisible(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_ui_state()
        self._update_ui_timer()
        
        # Restore log entries if any are missing
        log_count = self.log_list_layout.count() - 1
        records = batch_manager.summary_records
        if log_count < len(records):
            for i in range(max(0, log_count), len(records)):
                rec = records[i]
                self.add_log_entry(rec["image_name"], rec["status"])

    @Slot(str)
    def _sync_theme(self, theme_name: str = ""):
        theme = theme_name if theme_name else theme_service.current_theme
        
        if theme == "light":
            self.dashboard_card.setStyleSheet("""
                #DashboardCard {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 20px;
                }
            """)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #D1D5DB;
                    border-radius: 6px;
                    background-color: #F9FAFB;
                    text-align: center;
                    color: #111827;
                    font-weight: bold;
                    height: 22px;
                }
                QProgressBar::chunk {
                    background-color: #4F46E5;
                    border-radius: 5px;
                }
            """)
            self.status_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #111827;")
            
            # Labels Grid
            self.stats_frame.setStyleSheet("""
                QLabel { color: #111827; }
                QLabel[style*="9CA3AF"] { color: #4B5563; }
            """)
            
            self.log_card.setStyleSheet("""
                #LogCard {
                    background-color: #FFFFFF;
                    border: 1px solid #D1D5DB;
                    border-radius: 8px;
                    padding: 12px;
                }
            """)
            
            self.completion_card.setStyleSheet("""
                #CompletionCard {
                    background-color: #FFFFFF;
                    border: 2px solid #10B981;
                    border-radius: 8px;
                    padding: 20px;
                }
            """)
            self.comp_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #111827;")
            self.comp_desc.setStyleSheet("font-size: 12px; color: #4B5563;")
            self.splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #E5E7EB;
                }
                QSplitter::handle:hover {
                    background-color: #4F46E5;
                }
                QSplitter::handle:vertical {
                    height: 6px;
                }
            """)
            
        else:
            self.dashboard_card.setStyleSheet("""
                #DashboardCard {
                    background-color: #1C1C22;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 20px;
                }
            """)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #2B2B35;
                    border-radius: 6px;
                    background-color: #131317;
                    text-align: center;
                    color: #FFFFFF;
                    font-weight: bold;
                    height: 22px;
                }
                QProgressBar::chunk {
                    background-color: #4F46E5;
                    border-radius: 5px;
                }
            """)
            self.status_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
            
            self.log_card.setStyleSheet("""
                #LogCard {
                    background-color: #131317;
                    border: 1px solid #2B2B35;
                    border-radius: 8px;
                    padding: 12px;
                }
            """)
            
            self.completion_card.setStyleSheet("""
                #CompletionCard {
                    background-color: #1C1C22;
                    border: 2px solid #34D399;
                    border-radius: 8px;
                    padding: 20px;
                }
            """)
            self.comp_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
            self.comp_desc.setStyleSheet("font-size: 12px; color: #9CA3AF;")
            self.splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #2B2B35;
                }
                QSplitter::handle:hover {
                    background-color: #4F46E5;
                }
                QSplitter::handle:vertical {
                    height: 6px;
                }
            """)
