from typing import Dict, Type
from pathlib import Path
from lumen.core.imaging.base import ImageReader

class ImageReaderFactory:
    """Registry and factory for instantiating concrete ImageReader implementations."""
    
    _registry: Dict[str, Type[ImageReader]] = {}

    @classmethod
    def register_reader(cls, extension: str, reader_cls: Type[ImageReader]) -> None:
        """Registers a reader class for a specific file extension (case-insensitive)."""
        cls._registry[extension.lower()] = reader_cls

    @classmethod
    def get_reader(cls, file_path: str) -> ImageReader:
        """Instantiates and returns the appropriate ImageReader for the file path."""
        ext = Path(file_path).suffix.lower()
        reader_cls = cls._registry.get(ext)
        if not reader_cls:
            raise ValueError(f"No registered ImageReader found for extension '{ext}'")
        return reader_cls()
