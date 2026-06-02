# Changelog

All notable changes to the Lumen project will be documented in this file.

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
