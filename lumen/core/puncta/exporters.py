"""Tabular export interfaces for puncta quantification results.

This module houses export classes to serialize puncta analysis results 
to disk as CSV spreadsheets.
"""

from lumen.core.puncta.results import PunctaResults

class PunctaExporter:
    """Interface for exporting puncta analysis results to CSV files.
    
    This class handles rendering individual spot details and aggregated cell 
    statistics to separate spreadsheet reports.
    """

    def export_csv(self, results: PunctaResults, file_path: str) -> None:
        """Exports detailed per-punctum measurements to a CSV file.
        
        Parameters:
            results: PunctaResults object containing detailed measurements.
            file_path: Destination path for the CSV file.
            
        Raises:
            NotImplementedError: Exporter functionality is planned for future versions.
        """
        raise NotImplementedError("Puncta CSV exporter is not implemented yet.")

    def export_summary(self, results: PunctaResults, file_path: str) -> None:
        """Exports cell-level puncta summary statistics to a CSV file.
        
        Parameters:
            results: PunctaResults object containing summary statistics.
            file_path: Destination path for the CSV file.
            
        Raises:
            NotImplementedError: Exporter functionality is planned for future versions.
        """
        raise NotImplementedError("Puncta summary CSV exporter is not implemented yet.")
