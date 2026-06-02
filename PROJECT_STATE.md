# Lumen Project State (Source of Truth)

This document is the technical single source of truth for the current state of the Lumen application. It provides future developers and AI agents with an instant understanding of what is stable, what has been removed, and what must not be broken during future refactoring passes.

---

## 1. Stable & Active Systems (Do-Not-Break)

The following systems are fully functional, validated by tests, and **must not be modified or broken** in subsequent phases:

- **16-bit TIFF Normalization**: `image_manager.py` loads 16-bit raw microscopy images, profiles their signals, and scales them to 8-bit displays using the 1st/99th percentile contrast-stretching range.
- **Microscopy Modality Classifier**: Heuristically profiles imported image metadata and names, routing DAPI files to `nuclei` model weights and others to `cyto` models.
- **QThread-based Deep Learning Inference**: `AnalysisWorker` evaluates local Cellpose models asynchronously.
- **GPU/CUDA Detection & CPU Fallback**: Automatically leverages NVIDIA GPUs via CUDA if available, falling back to CPU and applying performance-tuned hyperparameters.
- **Local SQLite Persistence**: `config.py` saves window geometries and layout themes.
- **Interactive ImageViewer Canvas**: Renders zoom, panning, custom mask opacity overlays, hover highlights, and click metrics tooltips.
- **Data Export Pipeline**: Generates CSV dimension logs, overlay PNGs, raw 16-bit label mask TIFFs, and PDF reports.
- **Sequential Batch Loop**: Sequential multi-image queue system in `batch_manager.py` capable of stable **200-image batch runs**.
- **Reusable WorkspaceSwitcher Component**: Shared switcher button bar at the top of `AnalysisPage` (Single Image Explorer) and `BatchResultsExplorerPage` (Batch Results Explorer) handling page navigation and theme alignment.
- **Persistent Session Architecture**: Dictionary-based state maps in `WorkspaceManager` that preserve list selection, filters, sorting, canvas zoom transform, and scroll coordinates when transitioning between workspaces.
- **Dedicated Results Page Routing**: Navigating to `"results"` resolves directly to the Results panel showing active image metrics instead of redirecting to the batch explorer.

---

## 2. Deprecated & Intentionally Removed Systems

The following systems have been **completely cleaned and deleted** from the repository to eliminate oversegmentation issues and parameter noise:
- **`fast_segmentation_router.py`** (Old Fast Classical Watershed Segmentation).
- **`smart_segmentation_engine.py`** (Old Smart CellProfiler-inspired Classical Thresholding).
- **Legacy Parameters**: All `manual_*`, `smart_*`, and `fast_*` properties, settings, UI frames, layouts, sliders, and warning widgets have been removed.

---

## 3. Recent Fixes

### Threading GUI Crash Fix (v0.1.0)
- **Problem**: Clicking "Run Analysis" on Cellpose caused an immediate, silent app crash with no traceback in the terminal.
- **Root Cause**: `AnalysisWorker` is a `QThread` instantiated on the main thread, resulting in main-thread affinity. Signals connected using default `.connect()` parameters defaulted to `DirectConnection`. When the thread emitted updates, callbacks executed directly on the worker thread, causing fatal C++ segfaults when modifying GUI widgets.
- **Resolution**: Enforced `Qt.QueuedConnection` on all callbacks connected to the worker thread in `processing_manager.py` and `batch_manager.py`. GUI slots now execute safely on the main thread.

---

## 4. Known Technical Limitations

- **No Active Manual Correction Editor**: The classical brush tool was deleted. Manual boundary splitting/merging is unavailable.
- **StarDist Hook Only**: The routing layers accept StarDist hooks, but the model library backend is not yet implemented in `AnalysisWorker`.
- **Inference CPU Overhead**: Without a CUDA-compatible GPU, Cellpose CPU fallback is slow on larger microscopy images.

---

## 5. Next Implementation Phase

### Phase 1: Vector Manual Correction Brush Workspace
- **Priority**: High (Immediate).
- **Objective**: Rebuild a vector-based paintbrush and eraser drawing tool in `ImageViewer` allowing researchers to manually draw over, split, merge, or delete mask boundaries directly on the canvas, feeding updates back to `state.analysis_results`.
