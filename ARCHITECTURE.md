# Lumen Technical Architecture Reference

This document provides a detailed explanation of Lumen's system architecture, technical design decisions, and coding guidelines.

---

## 1. Application Module Map

Lumen is divided into distinct, decoupled architectural layers to isolate GUI rendering, state tracking, and background compute tasks.

```
lumen/
├── app/                  # Main Bootstrapping & Launcher lifecycle
├── core/                 # Shared system singletons and services
│   └── services/         # Hardware backend, navigation, and theme singletons
├── storage/              # Persistent SQLite settings schema
├── workflows/            # Application state management and templates
├── processing/           # Scientific imaging, display normalization, and thread workers
├── ui/                   # Global desktop shell frame and navbar
├── pages/                # Screen-specific layouts (Upload, Analysis, Results, Settings)
└── tests/                # Automated test discover suite
```

### Major Responsibilities
- **`lumen.app.launcher`**: Entry point. Sets up the Python environment, initializes SQLite, applies user themes, and invokes `main.py` to start the QApplication.
- **`lumen.core.services.gpu_service`**: Manages compute hardware. Checks PyTorch CUDA compatibility and falls back to CPU if no NVIDIA graphics card is active.
- **`lumen.workflows.state`**: Global state engine (`AppState` singleton). Emits Qt Signals whenever state variables (e.g. `current_image_path`, `analysis_results`) change.
- **`lumen.processing.image_manager`**: Memory-mapped image cache. Handles display normalization, metadata reading, and 8-bit QImage conversion.
- **`lumen.processing.processing_manager`**: Manages execution and signal wiring of background threads.

---

## 2. Segmentation Architecture

Lumen v0.1 implements a **Cellpose-centered segmentation workflow**. Legacy classical watershed engines (Fast, Manual, and Smart Segmentation) were completely removed due to severe over-segmentation and parameter sensitivity on scientific images.

### Extensible Backend Routing
To prevent future architectural rewrites, Lumen maintains a clean routing abstraction:
```
           [Image Classification & User Presets]
                           ↓
             [Heuristic Parameter Resolution]
                           ↓
             [segmentation_method routing]
              /                         \
    AI Segmentation (Cellpose)        StarDist (Future Hook)
            ↓                             ↓
    [AnalysisWorker]               [AnalysisWorker]
            ↓                             ↓
    Evaluate Local Model          Evaluate Local Model
            \                             /
             \                           /
              ↓                         ↓
            [Mask and Cell Metrics Dictionary Output]
```
In `processing_manager.py`, the routing checks `segmentation_method`. If it matches `"AI Segmentation"`, it triggers Cellpose inference. The system raises `ValueError` for unknown methods, leaving the interface ready to accommodate StarDist or other future deep learning backends.

---

## 3. Threading Architecture & Signal Marshaling

Scientific deep learning inference (Cellpose `model.eval()`) can block CPU/GPU execution for several seconds. To prevent UI freezing, Lumen processes all evaluations inside a background `QThread` (`AnalysisWorker`).

### The Direct Connection Bug
A critical PySide6 thread-safety bug was fixed in v0.1. Because `AnalysisWorker` inherits from `QThread`, it is a `QObject` instantiated on the main GUI thread. By default, connecting its signals to slots on `AnalysisPage` widgets without specifying a connection type causes Qt to resolve the connection as a **Direct Connection** (since sender and receiver affinities both reside on the main thread).
Consequently, when the thread called `self.progress_updated.emit(10)` from inside its background run loop, it executed the slot `_on_analysis_progress` directly on the background thread. Modifying GUI widgets (`QProgressBar.setValue()`) from a non-GUI thread resulted in an immediate silent C++ segfault.

### The Queued Connection Resolution
This is resolved by explicitly enforcing `Qt.QueuedConnection` on all worker signal connections:
```python
self.active_worker.progress_updated.connect(callbacks["progress"], Qt.QueuedConnection)
self.active_worker.status_updated.connect(callbacks["status"], Qt.QueuedConnection)
self.active_worker.finished_successfully.connect(callbacks["finished"], Qt.QueuedConnection)
self.active_worker.failed.connect(callbacks["failed"], Qt.QueuedConnection)
```
Using `Qt.QueuedConnection` forces Qt to serialize the signals and post them to the main thread's event queue. The slot methods are then executed strictly on the main GUI thread, guaranteeing 100% thread-safe UI updates.

---

## 4. Batch Processing Loop

Lumen executes batch analysis sequentially using the `BatchProcessingManager` singleton.

```
       [Select Folder] 
              ↓
    [Scan Valid Extensions]
              ↓
    [Save batch_metadata.txt]
              ↓
      [Loop Next Image]
              ↓
      Check Existing?
         /        \
       Yes        No
       /            \
  [Skip / Parse]  [Spawn AnalysisWorker Thread]
       \            /
        \          /
       [Image Result]
              ↓
  [_image_result_ready Signal] (Qt.QueuedConnection)
              ↓
  [_generate_outputs_on_main_thread]
              ↓
  [Write PNG, TIFF, CSV, PDF]
              ↓
       Increment Index
```

### Safety Constraints
- **Main Thread Output Rendering**: Creating PDF reports (`QPdfWriter`, `QTextDocument`) or saving images requires the Qt graphics pipeline. This must run on the main GUI thread. The worker thread simply emits the `_image_result_ready` signal, which is connected via `Qt.QueuedConnection` to trigger file output generation safely on the main thread.
- **Resource Cleanup**: The master 2D numpy mask array is deleted from memory (`del results["masks"]`) at the end of each image iteration, preventing RAM accumulation during large 200+ image runs.

---

## 5. State & Persistence System

Lumen uses a two-tier state system:

1. **Persistent State (`lumen.core.config.config`)**: Stored in a lightweight SQLite database (`lumen.db`). Holds configurations that must survive restarts (Light/Dark theme, window geometry, last-opened folder path).
2. **Transient Session State (`lumen.workflows.state.state`)**: Kept in memory. Resets automatically on image switches to prevent parameters (e.g. mask opacity, zoom scale) from leaking between different scientific images.

---

## 6. Core Design Principles

- **Scientific Reliability First**: We prioritize standard algorithms and validated deep learning frameworks (Cellpose) over unproven classical heuristics.
- **Simple > Clever**: Maintain flat module structures, explicit signal connections, and clear, traceable logging.
- **Extensible Architecture**: Code interfaces to support future models (StarDist) and tools (vector paintbrush correction layers) without modifying core GUI components.
- **Researcher-Friendly UX**: Eliminate parameter overload. Provide intelligent defaults (auto contrast-stretching, automatic modality routing) so researchers can get high-quality segmentations on the first click.
