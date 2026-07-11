# Changelog

All notable changes to the Lumen project will be documented in this file.

---

## [0.4.0] - 2026-07-11

### Architectural Refactor, CZI Support, & Puncta Quantification Backend

This major update introduces format-agnostic image readers, memory-safe CZI file handling for Single Image Explorer, a robust Difference of Gaussians (DoG) puncta quantification backend, and a simplified professional scientific UX layout on the Analysis page.

#### Added
- **Unified Image Reader System**: Created a format-agnostic `ImageReader` abstraction with factory loading for TIFF (`TiffReader`), standard image formats (`PilReader`), and Carl Zeiss CZI (`CziReader`).
- **Memory-Safe CZI Integration**: Added memory-copying mechanisms in the C++ `libCZI` bindings to prevent access violations / native segmentation crashes when python numpy array views outlived the deallocated C++ reader.
- **Puncta Detection Backend**: Developed a complete subcellular spot detection engine (`lumen.core.puncta`) utilizing Difference of Gaussians (DoG), local maxima filtering, parent-cell assignment geometry, and intensity quantification metrics.
- **Dynamic Parameter Visibility**: The sidebar dynamically shows/hides Fluorescence and Puncta configuration panels depending on the active workflow mode.
- **Focus-Safe Widgets**: Implemented `FocusWheelComboBox`, `FocusWheelSlider`, `FocusWheelSpinBox`, and `FocusWheelDoubleSpinBox` to prevent mouse wheel events from changing parameters unless the control has active keyboard focus.

#### Changed
- **Analysis Page UX Refactor**: Replaced the cluttered multi-card panel with a simplified sidebar:
  1. Segmentation Settings (Model, Target Channel, Quality)
  2. Calibration Settings (Pixel ↔ Micron Mode)
  3. Image Preprocessing (Collapsible Drawer)
  4. Fluorescence/Puncta Settings (Dynamic Collapsible Drawer)
- **Bottom Actions Dock**: Prominently anchored the "Run Analysis" button as the primary visual action, grouping secondary buttons ("Edit Masks", "Save Analysis", "Reset Changes") directly below.
- **Right Panel Gutter**: Set a strict 16px right margin inside the sidebar's QScrollArea to prevent scrollbar overlapping and layout clipping at different screen resolutions.
- **Execution Settings Removal**: Removed GPU/CPU selectors from the page layout to rely entirely on global backend settings, simplifying the scientific user workflow.

#### Known Issues / Limitations
- **Batch CZI Uploads**: Batch processing is currently unsupported for `.czi` files.
- **Batch Fluorescence Analysis**: Batch mode does not support multi-channel fluorescence naming or quantification; attempts to run batch fluorescence analysis will fail and need appropriate policing.
- **Puncta UI Integration**: The puncta quantification backend is fully functional (with unit test coverage) but is pending complete UI controls and pipeline integration in the frontend.

---

## [0.1.0] - 2026-06-02

### Stable Cellpose Architecture Milestone

This release establishes the core deep learning backend and desktop interface for local image segmentation.

#### Added
- Local **Cellpose model inference** integration with automatic weights downloading.
- **Hardware Backend Auto-Detection**: GPU/CUDA acceleration configuration with automatic fallback to CPU.
- **Image display normalization**: 16-bit to 8-bit dynamic range stretching (using 1st/99th percentiles) with zero-signal safeguards.
- **Interactive ImageViewer**: Custom graphics viewport with high-performance panning, zoom-under-mouse, double-click fits, and mask overlays.
- **Micro-Inspection tooltips**: Hover/click feedback displaying Cell ID, area, diameter, and centroid coordinates.
- **High-throughput batch manager**: Sequential processing queue validated on large **200-image batch runs** on Windows.
- **Multi-format data exports**: Support for CSV spreadsheets, overlay PNGs, raw 16-bit TIFF label masks, and formatted A4 PDF reports.
- **Persistent SQLite configuration database** to preserve theme options and window sizes.

#### Changed
- **Qt Threading Architecture Fix**: Replaced auto-routed signal connections on background threads with explicit `Qt.QueuedConnection` parameters. This forces GUI updates (like progress bar sets and tooltips) to execute safely on the main thread, fixing the silent desktop crash.
- **Architecture Simplification**: Refactored State and Page structures to align around the AI deep learning backend, maintaining parameter extensibility for StarDist.

#### Removed
- **Fast Segmentation** classical watershed router.
- **Manual Segmentation** (legacy classical watershed editor).
- **Smart Segmentation** (classical thresholding engine).
