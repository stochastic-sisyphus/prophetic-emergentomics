"""
Prophetic Emergentomics: ML/DL-Driven Emergence Detection

Machine learning framework for detecting emergence in complex economic systems.
GDELT provides behavioral data; ML/DL models detect phase transitions and regime changes.

Core Philosophy:
- Economies behave like ecologies, not machines
- Emergence detection, not prediction
- Uncertainty as structure, not error
- Behavioral traces reveal what statistics miss

Architecture:
- Data Layer: GDELT events + alternative behavioral data sources
- Detection Layer: Anomaly detection, clustering, network analysis
- Output Layer: Observable-compatible visualization data

Key Components:
- GDELT Integration: Real-time global event and sentiment behavioral traces
- Anomaly Detection: Isolation forest, statistical methods for outlier detection
- Clustering: HDBSCAN, K-means for economic regime identification
- Network Analysis: Graph structures for economic interconnections
- Observable Output: JSON format compatible with Observable visualization
"""

__version__ = "0.1.0"
__author__ = "Vanessa Beck"

from emergentomics.core.config import Settings, get_settings
from emergentomics.core.models import (
    EmergenceSignal,
    EmergenceType,
    EmergenceReport,
    GDELTEvent,
    GDELTSentiment,
)
from emergentomics.gdelt.client import GDELTClient
from emergentomics.detection.anomaly import AnomalyDetector
from emergentomics.detection.clustering import ClusterAnalyzer
from emergentomics.detection.emergence import EmergenceDetector

__all__ = [
    "Settings",
    "get_settings",
    "EmergenceSignal",
    "EmergenceType",
    "EmergenceReport",
    "GDELTEvent",
    "GDELTSentiment",
    "GDELTClient",
    "AnomalyDetector",
    "ClusterAnalyzer",
    "EmergenceDetector",
]
