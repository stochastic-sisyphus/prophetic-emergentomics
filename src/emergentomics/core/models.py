"""
Core data models for Prophetic Emergentomics.

Data structures for ML/DL-driven emergence detection in complex economic systems.
GDELT provides behavioral data; ML/DL models detect emergence patterns.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================


class EmergenceType(str, Enum):
    """Types of economic emergence signals detected by ML models."""

    PHASE_TRANSITION = "phase_transition"
    TREND_ACCELERATION = "trend_acceleration"
    CLUSTER_FORMATION = "cluster_formation"
    ANOMALY_CASCADE = "anomaly_cascade"
    NETWORK_RESTRUCTURING = "network_restructuring"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    CONTAGION_PATTERN = "contagion_pattern"
    ADAPTATION_SIGNAL = "adaptation_signal"


class DataSourceType(str, Enum):
    """Types of data sources."""

    GDELT = "gdelt"  # Global event/sentiment from news
    SEARCH_TRENDS = "search_trends"
    JOB_POSTINGS = "job_postings"
    CONSUMER_SENTIMENT = "consumer_sentiment"
    MARKET_DATA = "market_data"
    SOCIAL_DISCUSSION = "social_discussion"
    FILING_RECORDS = "filing_records"


class ModelType(str, Enum):
    """ML/DL model types."""

    GNN = "graph_neural_network"
    AUTOENCODER = "autoencoder"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    REINFORCEMENT = "reinforcement_learning"
    TRANSFER = "transfer_learning"
    ENSEMBLE = "ensemble"


class SentimentPolarity(str, Enum):
    """Sentiment classification."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


# =============================================================================
# GDELT Data Models (Behavioral Data Layer)
# =============================================================================


class GeoLocation(BaseModel):
    """Geographic location data."""

    country_code: Optional[str] = None
    country_name: Optional[str] = None
    adm1_code: Optional[str] = None
    adm1_name: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class Actor(BaseModel):
    """Actor in an economic event."""

    name: Optional[str] = None
    actor_type: Optional[str] = None
    country_code: Optional[str] = None


class GDELTEvent(BaseModel):
    """An event from GDELT - behavioral trace of global information flow."""

    id: str
    timestamp: datetime
    source_url: Optional[str] = None
    source_name: Optional[str] = None

    # Event classification
    event_type: str
    themes: list[str] = Field(default_factory=list)
    economic_sector: Optional[str] = None

    # Actors
    actor1: Optional[Actor] = None
    actor2: Optional[Actor] = None

    # Location
    location: Optional[GeoLocation] = None

    # Quantitative measures from GDELT
    goldstein_scale: Optional[float] = None  # -10 to +10
    num_mentions: int = 1
    num_sources: int = 1
    num_articles: int = 1

    # Tone analysis
    tone: Optional[float] = None
    positive_score: Optional[float] = None
    negative_score: Optional[float] = None

    # Raw preservation
    raw_data: Optional[dict[str, Any]] = None


class GDELTSentiment(BaseModel):
    """Aggregated sentiment from GDELT - collective mood from news coverage."""

    id: str
    timestamp: datetime
    aggregation_period: str = "1h"

    # Scope
    scope: str = "global"
    country_code: Optional[str] = None
    sector: Optional[str] = None

    # Sentiment metrics
    overall_tone: float
    polarity: SentimentPolarity
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Component scores
    anxiety_score: float = 0.0
    optimism_score: float = 0.0
    uncertainty_score: float = 0.0

    # Volume
    article_count: int = 0
    source_diversity: int = 0

    # Trend
    sentiment_velocity: Optional[float] = None
    top_themes: list[str] = Field(default_factory=list)


# =============================================================================
# ML/DL Data Models
# =============================================================================


class TimeSeriesPoint(BaseModel):
    """A single point in an economic time series."""

    timestamp: datetime
    value: float
    source: DataSourceType
    indicator_name: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EconomicTimeSeries(BaseModel):
    """Time series for ML analysis."""

    id: str
    name: str
    source: DataSourceType
    frequency: str = "daily"
    points: list[TimeSeriesPoint] = Field(default_factory=list)

    # Computed stats
    mean: Optional[float] = None
    std: Optional[float] = None
    trend: Optional[str] = None

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML processing."""
        return np.array([p.value for p in self.points])


class EconomicNode(BaseModel):
    """Node in economic network graph for GNN."""

    id: str
    node_type: str  # sector, region, indicator, entity
    name: str
    features: list[float] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EconomicEdge(BaseModel):
    """Edge in economic network graph."""

    source_id: str
    target_id: str
    edge_type: str  # trade, correlation, supply_chain, co_mention
    weight: float = 1.0
    features: list[float] = Field(default_factory=list)
    timestamp: Optional[datetime] = None


class EconomicGraph(BaseModel):
    """Graph for GNN analysis of economic networks."""

    id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    nodes: list[EconomicNode] = Field(default_factory=list)
    edges: list[EconomicEdge] = Field(default_factory=list)

    # Graph metrics
    density: Optional[float] = None
    clustering_coefficient: Optional[float] = None

    def to_adjacency_dict(self) -> dict[str, list[str]]:
        """Convert to adjacency list."""
        adj = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            if edge.source_id in adj:
                adj[edge.source_id].append(edge.target_id)
        return adj


# =============================================================================
# Detection Results
# =============================================================================


class AnomalyScore(BaseModel):
    """Anomaly detection result."""

    timestamp: datetime
    score: float = Field(..., ge=0.0, le=1.0)
    method: str
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None


class ClusterAssignment(BaseModel):
    """Cluster assignment from unsupervised learning."""

    point_id: str
    cluster_id: int
    membership_score: float = 1.0
    distance_to_centroid: Optional[float] = None


class ClusteringResult(BaseModel):
    """Clustering analysis result."""

    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    method: str

    n_clusters: int
    assignments: list[ClusterAssignment] = Field(default_factory=list)

    # Quality metrics
    silhouette_score: Optional[float] = None
    centroids: list[list[float]] = Field(default_factory=list)
    cluster_sizes: list[int] = Field(default_factory=list)


class DimensionalityReduction(BaseModel):
    """Dimensionality reduction result (t-SNE, UMAP, PCA)."""

    id: str
    method: str
    original_dims: int
    reduced_dims: int

    coordinates: list[list[float]] = Field(default_factory=list)
    point_ids: list[str] = Field(default_factory=list)
    explained_variance: Optional[float] = None


# =============================================================================
# Emergence Signals
# =============================================================================


class EmergenceSignal(BaseModel):
    """Computationally detected emergence signal."""

    id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    # Classification
    emergence_type: EmergenceType
    description: str

    # Detection method
    detection_method: ModelType
    model_name: str

    # Strength
    signal_strength: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # Evidence
    anomaly_scores: list[AnomalyScore] = Field(default_factory=list)
    cluster_shifts: Optional[ClusteringResult] = None
    supporting_events: list[str] = Field(default_factory=list)

    # Scope
    affected_indicators: list[str] = Field(default_factory=list)
    affected_sectors: list[str] = Field(default_factory=list)
    geographic_scope: Optional[str] = None

    # Temporal
    onset_timestamp: Optional[datetime] = None
    propagation_speed: Optional[str] = None


class PhaseTransition(BaseModel):
    """Detected regime change / phase transition."""

    id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    from_regime: str
    to_regime: str
    transition_type: str  # abrupt, gradual, oscillating

    # Evidence
    order_parameter: str
    critical_value: float
    current_value: float

    confidence: float = Field(..., ge=0.0, le=1.0)


# =============================================================================
# Simulation (RL)
# =============================================================================


class EconomicState(BaseModel):
    """State for RL environment."""

    timestamp: datetime
    observation: list[float] = Field(default_factory=list)
    indicator_values: dict[str, float] = Field(default_factory=dict)
    is_terminal: bool = False


class PolicyAction(BaseModel):
    """Action in RL simulation."""

    action_type: str
    parameters: dict[str, float] = Field(default_factory=dict)
    description: str = ""


class SimulationStep(BaseModel):
    """Single RL simulation step."""

    step: int
    state: EconomicState
    action: PolicyAction
    reward: float
    next_state: EconomicState
    done: bool


# =============================================================================
# Output Reports
# =============================================================================


class EmergenceReport(BaseModel):
    """Comprehensive emergence detection report - observable output."""

    id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    analysis_period: str

    # GDELT summary
    gdelt_events_analyzed: int = 0
    gdelt_sentiment_summary: Optional[GDELTSentiment] = None

    # ML detection results
    signals_detected: list[EmergenceSignal] = Field(default_factory=list)
    phase_transitions: list[PhaseTransition] = Field(default_factory=list)

    # Clustering insights
    regime_clusters: Optional[ClusteringResult] = None
    dimensionality_reduction: Optional[DimensionalityReduction] = None

    # Network analysis
    network_metrics: dict[str, float] = Field(default_factory=dict)
    critical_nodes: list[str] = Field(default_factory=list)

    # Anomalies
    anomaly_rate: float = 0.0
    top_anomalies: list[AnomalyScore] = Field(default_factory=list)

    # Metadata
    models_used: list[str] = Field(default_factory=list)
    data_sources_used: list[DataSourceType] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_factors: list[str] = Field(default_factory=list)
