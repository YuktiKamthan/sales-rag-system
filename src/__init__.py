"""
Sales RAG System
----------------
Enterprise-level RAG pipeline with OpenTelemetry monitoring and sales prediction.

Author: Yukti
Version: 1.0.0
"""

__version__ = "1.0.0"

from .data_loader import SalesDataLoader
from .telemetry import telemetry, metrics

__all__ = ['SalesDataLoader', 'telemetry', 'metrics']
