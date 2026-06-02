# Lumen Development Roadmap

This document outlines the planned development path for Lumen.

---

## 1. Immediate Priorities: v0.2.0 (Q3 2026)

### Manual Segmentation Correction MVP
- **Goal**: Reintroduce a robust manual editing workspace that allows researchers to correct model predictions.
- **Features**:
  - A clean, vector-based click-and-drag paint brush and eraser interface.
  - Split and merge operations for adjacent cells.
  - Keybindings for quick tool switching (`B` for brush, `E` for erase, `Z` for zoom).
  - Undo/redo stack.

---

## 2. Near-Term Milestones: v0.3.0 (Q4 2026)

### StarDist Backend Integration
- **Goal**: Add StarDist as a lightweight, CPU-optimized model alternative for segmenting spherical objects (like nuclei).
- **Features**:
  - StarDist model weights routing (DAPI stains).
  - Parallel backend comparison viewer (view Cellpose vs StarDist outputs side-by-side).
  - Threshold sensitivity sliders for StarDist star-convex shapes.

### Batch Review Explorer
- **Goal**: Provide a grid-based gallery workspace for high-throughput batch runs.
- **Features**:
  - Visual gallery of batch overlay previews.
  - Status flag system (Approve / Reject / Flags).
  - Interactive parameter override (double-click any thumbnail to open the file in the editor, adjust parameters, and re-run segmentation for just that image).

---

## 3. Long-Term Vision (2027+)

### Advanced Correction & Tracking Tooling
- **Goal**: Expand into time-lapse cell tracking and 3D stack segmentation.
- **Features**:
  - Frame-to-frame cell matching and lineage tracking graphs.
  - 3D stack visualization and segment interpolation across Z-slices.

### Smarter Scientific Export Formats
- **Goal**: Direct integration with professional bio-image databases.
- **Features**:
  - OME-TIFF metadata preservation.
  - Direct database upload connector for OMERO platforms.

### Workflow Heuristics & ML Auto-Tuning
- **Goal**: Automatically tune deep learning parameters based on image SNR and density profile feedback.
- **Features**:
  - SNR-driven automatic model selection.
  - Active learning loop: use manual corrections to locally fine-tune model weights.
