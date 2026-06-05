from typing import List, Dict, Any
from lumen.core.logger import logger

class WorkflowMetadata:
    def __init__(self, id_str: str, name: str, description: str, steps: List[str]):
        self.id = id_str
        self.name = name
        self.description = description
        self.steps = steps

class WorkflowManager:
    """Manages biological analysis workflow templates and definitions."""

    def __init__(self):
        self.workflows: Dict[str, WorkflowMetadata] = {
            "cell_counting": WorkflowMetadata(
                "cell_counting",
                "Cell Segmentation",
                "Detect, segment, and quantify cells or nuclei in biological images.",
                ["Load Image", "Configure Parameters", "Run Segmentation Engine", "Review and Export Metrics"]
            ),
            "fluorescence": WorkflowMetadata(
                "fluorescence",
                "Fluorescence Analysis",
                "Quantify intensity profiles across multiple channels (RGB/DAPI/FITC/TRITC).",
                ["Load Multi-channel Image", "Map Fluorescent Channels", "Measure Intensity Metrics", "Plot Output Profiles"]
            ),
            "colony": WorkflowMetadata(
                "colony",
                "Colony Analysis",
                "Identify and measure area, shape, and count of bacterial or cell colonies in culture plates.",
                ["Load Agar Plate Image", "Set Plate Border Mask", "Segment Colonies", "Quantify Spatial Distribution"]
            ),
            "custom": WorkflowMetadata(
                "custom",
                "Custom Workflow",
                "Chain custom pre-processing, thresholding, and segmentation nodes.",
                ["Define Node Graph", "Link Input/Output Anchors", "Simulate Pipeline", "Save Preset"]
            )
        }
        logger.info("WorkflowManager initialized with %d workflow templates.", len(self.workflows))

    def get_workflow(self, workflow_id: str) -> Any:
        """Retrieves metadata definition for a workflow ID."""
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[Dict[str, str]]:
        """Returns key details of all workflows for menu layouts."""
        return [
            {"id": wf.id, "name": wf.name, "description": wf.description}
            for wf in self.workflows.values()
        ]

# Global workflow manager instance
workflow_manager = WorkflowManager()
