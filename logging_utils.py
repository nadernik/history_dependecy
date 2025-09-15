"""
Logging Utilities for Serial Dependence Analysis
================================================

This module provides logging infrastructure to capture analysis output
to timestamped log files with both console and file output.
"""

import os
import sys
from datetime import datetime
from typing import Optional, TextIO


class AnalysisLogger:
    """Logger class to capture all analysis output to timestamped log files"""
    
    def __init__(self, log_dir: str = 'logs', enable_logging: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            enable_logging: Whether to enable logging to files
        """
        self.log_dir = log_dir
        self.enable_logging = enable_logging
        self.log_file: Optional[TextIO] = None
        self.original_stdout: Optional[TextIO] = None
        self.original_stderr: Optional[TextIO] = None
        
        if self.enable_logging:
            # Create logs directory if it doesn't exist
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                
            # Generate timestamped log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = f"serial_dependence_analysis_{timestamp}.log"
            self.log_filepath = os.path.join(self.log_dir, self.log_filename)
    
    def start_logging(self) -> None:
        """Start capturing output to log file"""
        if not self.enable_logging:
            return
            
        self.log_file = open(self.log_filepath, 'w', encoding='utf-8')
        
        # Write header
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("SERIAL DEPENDENCE ANALYSIS - COMPLETE LOG\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Log file: {self.log_filename}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()
        
        # Create a custom stdout that writes to both console and file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        class TeeOutput:
            """Output class that writes to both console and file"""
            def __init__(self, file1: TextIO, file2: TextIO):
                self.file1 = file1
                self.file2 = file2
                
            def write(self, data: str) -> int:
                """Write data to both outputs"""
                self.file1.write(data)
                self.file2.write(data)
                return len(data)
                
            def flush(self) -> None:
                """Flush both outputs"""
                self.file1.flush()
                self.file2.flush()
                
            def isatty(self) -> bool:
                """Check if output is a terminal"""
                return self.file1.isatty()
        
        # Redirect stdout and stderr to tee outputs
        sys.stdout = TeeOutput(self.original_stdout, self.log_file)
        sys.stderr = TeeOutput(self.original_stderr, self.log_file)
        
        print(f"Logging started - output will be saved to: {self.log_filepath}")
    
    def stop_logging(self) -> None:
        """Stop capturing output and restore original streams"""
        if not self.enable_logging or self.log_file is None:
            return
            
        # Write footer
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"Analysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Complete log saved to: {self.log_filepath}")
        print("=" * 80)
        
        # Restore original stdout and stderr
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr
            
        # Close log file
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            
        print(f"Logging stopped - log saved to: {self.log_filepath}")
    
    def __enter__(self) -> 'AnalysisLogger':
        """Context manager entry"""
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.stop_logging()
