import os
from enum import Enum
from lumen.core.logger import logger
from lumen.workflows.state import state

class BackendPreference(Enum):
    AUTO = "Auto"
    CUDA = "CUDA (GPU)"
    CPU = "CPU"

class GPUService:
    """Manages system hardware backend configuration and CUDA detection."""

    def __init__(self):
        self._backend = self._detect_backend()
        logger.info("GPUService initialized. Detected backend: %s", self._backend)
        state.current_backend = self._backend

    def resolve_execution_backend(self, preference: str) -> tuple:
        """Resolves backend preference (string or Enum) to (use_gpu: bool, resolved_name: str)."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if preference == "Use Global Setting":
            from lumen.workflows.state import state
            pref_val = state.backend_preference
        else:
            pref_val = preference

        if isinstance(pref_val, BackendPreference):
            pref_str = pref_val.value
        else:
            pref_str = str(pref_val)

        if "auto" in pref_str.lower():
            if cuda_available:
                resolved = (True, "CUDA")
            else:
                resolved = (False, "CPU")
        elif "cuda" in pref_str.lower() or "gpu" in pref_str.lower():
            if cuda_available:
                resolved = (True, "CUDA")
            else:
                logger.warning("GPUService: CUDA requested but unavailable. Falling back to CPU.")
                resolved = (False, "CPU (fallback)")
        else: # CPU
            resolved = (False, "CPU")

        logger.info("GPUService: Resolving backend. Preference: %s, CUDA available: %s, Resolved: %s", pref_str, cuda_available, resolved[1])
        return resolved

    def _detect_backend(self) -> str:
        """Determines if a CUDA GPU is available, falling back to OS checks for dev shell credibility."""
        # 1. Primary Check: Use standard PyTorch detection
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("GPUService: PyTorch CUDA check succeeded.")
                return "CUDA"
            logger.info("GPUService: PyTorch reported CPU (CUDA not active).")
        except ImportError:
            logger.debug("GPUService: PyTorch not installed in this environment.")
        except Exception as e:
            logger.warning("GPUService: Exception in PyTorch GPU check: %s", e)

        # 2. Developer Fallback: Verify if NVIDIA GPU hardware and drivers are present on the OS.
        # This keeps the shell mock status ('Backend: CUDA') accurate for supported RTX hardware
        # before the developer installs massive PyTorch wheels in their local .venv.
        if os.name == 'nt':  # Windows
            # Standard paths for Windows CUDA driver DLL
            system32_path = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', 'nvcuda.dll')
            syswow64_path = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'SysWOW64', 'nvcuda.dll')
            
            if os.path.exists(system32_path) or os.path.exists(syswow64_path):
                logger.info("GPUService: NVIDIA CUDA driver (nvcuda.dll) detected on Windows. Setting backend to CUDA.")
                return "CUDA"
                
        elif os.name == 'posix':  # Linux/macOS
            # Check for nvidia-smi command path
            import shutil
            if shutil.which("nvidia-smi"):
                logger.info("GPUService: nvidia-smi command found on POSIX. Setting backend to CUDA.")
                return "CUDA"

        return "CPU"

    @property
    def backend(self) -> str:
        """Returns active backend representation ('CPU' or 'CUDA')."""
        return self._backend

    @property
    def is_cuda_available(self) -> bool:
        """Helper to quickly check if CUDA-capable hardware is active."""
        return self._backend == "CUDA"

# Global GPU Service instance
gpu_service = GPUService()
