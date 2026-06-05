# Lumen

A modern, AI-powered desktop microscopy analysis platform focused on automated cell segmentation, workflow intelligence, and high-throughput scientific data extraction.

---

> [!NOTE]
> **Disclaimer**: Lumen v0.2 is a research-focused milestone release. Stability and workflow correctness were prioritized over UI polish.

---

## The Vision: Why Lumen Matters

Modern biological research generates massive amounts of microscopy imaging data daily, but extracting meaningful quantitative metrics remains a major bottleneck. Researchers are typically forced to choose between:
- **Generic open-source tools (Fiji/ImageJ)**, which require tedious manual configuration, manual thresholding, and have steep learning curves.
- **Fragile classical watershed algorithms**, which suffer from "marker explosion" or "marker starvation" and fail to generalize across noisy fluorescence imaging modalities.
- **Enterprise AI suites**, which are expensive, lock data into proprietary clouds, and require complex coding.

**Lumen solves this bottleneck.** It provides a local, production-grade, zero-configuration desktop workstation that automatically profiles microscopy images, routes them to state-of-the-art deep learning architectures (e.g. Cellpose), and extracts standardized, publishable scientific measurements. By combining advanced deep learning models with a responsive, high-performance desktop interface, Lumen enables researchers to run batch analysis of hundreds of multi-gigabyte microscopy images with single-click ease.

---

## Core Features (v0.2.0)

### Deep Learning & Image Handling
- **Cellpose-Powered Microscopy Segmentation**: Native integration with local Cellpose v3.0+ deep learning model evaluations.
- **Intelligent Modality & Routing**: Automatically inspects file metadata and pixel arrays on import to determine imaging modality (e.g., *Fluorescence Microscopy*, *Brightfield Microscopy*) and routes them to DAPI nuclei stains (`nuclei` model) or GFP cell bodies (`cyto` model).
- **16-bit to 8-bit Normalization**: scales 16-bit high-dynamic-range TIFF microscopy images down to 8-bit display buffers using **1st/99th percentile contrast-stretching** with division-by-zero safeguards.

### Analysis & Workspace
- **Single Image Analysis**: Perform profiling, model inference, and metadata inspection for individual microscopy images.
- **Interactive Viewer & Tooltips**: Features a zoomable, pannable graphics canvas custom-tailored for large biological image inspection with adjustable opacity mask overlays. Single-clicking any segmented cell displays quantitative dimensions (Cell ID, Area, Diameter, and Centroid) via hovering tooltips.
- **Manual Segmentation / Mask Editing**: Allows vector-drawn interactive mask editing. Researchers can manually add, delete, or merge cells directly on the visual canvas.
- **Draft → Commit Scientific Correction Workflow**: Supports saving modifications as a draft and committing scientific corrections to maintain reproducibility.
- **Save Analysis / Save to Batch**: Fully integrated saving workflow supporting custom modifications on batch images.
- **Reanalysis Support**: Detects dirty state during scientific reanalysis, prompting users to re-run or commit changes.

### Batch Pipeline & Results Explorer
- **Batch Folder Processing**: Scan folders recursively or flat for sequential batch processing of microscopy images.
- **Background Batch Execution**: Executes Cellpose inference on background threads, keeping the main GUI responsive.
- **Pause / Resume / Cancel**: Enforces strict lifecycle state machines allowing users to pause, resume, or cancel batch executions at any time.
- **Persistent Batch Progress Restoration**: Resumes interrupted batches, automatically skipping images with complete output files on disk to prevent duplicate processing.
- **Batch Results Explorer**: Grid-based gallery page enabling researchers to inspect segmentation outputs, review summary metrics, and navigate to the analysis workspace.
- **Explorer Session Persistence**: Remembers active search queries, selected indices, and list states across page transitions.
- **Comprehensive Export Pipeline**: Generates visual previews (PNG), raw 16-bit label mask files (TIFF), per-cell metrics (CSV), reproducibility parameters (Metadata TXT), run parameters (JSON Manifest), and publication-ready A4 PDF reports.

---

## Technical Improvements

- **Persistent Batch Worker Architecture**: Transitioned from a per-image worker/thread allocation lifecycle to a persistent worker and thread lifecycle, eliminating PySide6/QThread creation churn under heavy CUDA/native load.
- **Cross-Run Lifecycle Isolation**: Implemented hard startup gates and teardown barriers to guarantee consecutive runs do not contaminate each other's lifecycle or state.
- **Safe Background Execution**: Thread-safe queued signal connection system prevents C++ Qt layout conflicts during concurrent tasks.
- **Canonical Disk Restoration**: Batch Explorer resolves modified records directly from canonical outputs on disk to guarantee session persistence accuracy.
- **Dirty-State Tracking**: Employs rigorous checks comparing current analysis results against committed data to protect against unsaved corrections.
- **Progress Rehydration**: Automatically restores batch progress and rehydrates logs when navigating between pages.

---

## Technical Stack

- **Core Runtime**: Python 3.12+
- **GUI Framework**: PySide6 (Qt 6.x) for high-performance cross-platform desktop shells.
- **Deep Learning Engine**: PyTorch + Cellpose < 4.0.0
- **Scientific Computing**: NumPy, Pillow, Tifffile, SciPy.
- **Database**: SQLite3 for persistent configurations and settings.
- **Graphics Pipeline**: Qt Graphics View Framework for pixel-perfect zoom and drag rendering.

---

## Stability Status
- **Current Status**: **Production Stable (v0.2.0)**.
- Fully validated on single-channel and multi-channel 16-bit DAPI/GFP TIFF images.
- Continuous Integration verified: 60 unit tests validating display normalization, heuristics, database, manual segmentation, and persistent batch inference pipelines pass on local runs.

---

## Known Limitations (v0.2)

- **Pause → Resume Throughput**: Pause → Resume transitions may temporarily reduce throughput for the first image(s) before returning to normal processing speed.
- **Manual Mask Editor Toolbar**: Toolbar scaling can visually break or overlap on certain non-default window sizes or aspect ratios.
- **Mask Editor Layout**: Layout responsiveness under extreme aspect ratio resizes is pending final visual polish.
- **StarDist Integration**: Routing hooks exist in the routing engine, but the StarDist library backend has not yet been integrated into the thread worker.

---

## Roadmap

1. **Phase 1: Multi-Channel Visualization**: Enhance the graphics pipeline to support independent layer blending and color map assignments for multi-channel fluorescence stacks.
2. **Phase 2: StarDist Backend Integration**: Incorporate StarDist as a lightweight CPU-optimized alternative for spherical objects (like nuclei), complete with multi-model comparison interfaces.
3. **Phase 3: Advanced Export Integrations**: Support direct uploads to OME-TIFF formats and metadata syncing with OMERO databases.
