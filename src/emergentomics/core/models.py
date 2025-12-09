"""
Core data models for Prophetic Emergentomics.

These models represent the fundamental data structures flowing through
the CAG + GDELT economic intelligence pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MedallionLayer(str, Enum):
    """Data quality layers in the medallion architecture."""

    BRONZE = "bronze"  # Raw, unprocessed data
    SILVER = "silver"  # Cleaned, consolidated, enriched
    GOLD = "gold"  # Analytics-ready, LLM-synthesized


class SentimentPolarity(str, Enum):
    """Sentiment classification."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EmergenceType(str, Enum):
    """Types of economic emergence signals."""

    PHASE_TRANSITION = "phase_transition"
    TREND_ACCELERATION = "trend_acceleration"
    NARRATIVE_SHIFT = "narrative_shift"
    POLICY_SIGNAL = "policy_signal"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    GEOGRAPHIC_CLUSTERING = "geographic_clustering"
    SECTOR_CONTAGION = "sector_contagion"
    TECHNOLOGY_ADOPTION = "technology_adoption"


class OpportunityType(str, Enum):
    """Types of economic opportunities detected."""

    MARKET_TIMING = "market_timing"
    SECTOR_ROTATION = "sector_rotation"
    POLICY_ARBITRAGE = "policy_arbitrage"
    NARRATIVE_MOMENTUM = "narrative_momentum"
    GEOGRAPHIC_ADVANTAGE = "geographic_advantage"
    TECHNOLOGY_WAVE = "technology_wave"
    SENTIMENT_REVERSAL = "sentiment_reversal"


class GeoLocation(BaseModel):
    """Geographic location data."""

    country_code: Optional[str] = None
    country_name: Optional[str] = None
    adm1_code: Optional[str] = None  # State/Province
    adm1_name: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geo_precision: Optional[int] = None  # 1=country, 2=state, 3=city, 4=landmark


class Actor(BaseModel):
    """Actor in an economic event."""

    name: Optional[str] = None
    actor_type: Optional[str] = None  # GOV, BUS, NGO, etc.
    country_code: Optional[str] = None
    known_group: Optional[str] = None
    ethnic: Optional[str] = None
    religion: Optional[str] = None


class EconomicEvent(BaseModel):
    """
    An economic event extracted from GDELT or other sources.

    Represents the "who did what to whom, when, where" of economic activity.
    """

    id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(..., description="When the event occurred")
    source_url: Optional[str] = None
    source_name: Optional[str] = None

    # Event classification
    event_type: str = Field(..., description="CAMEO event code or category")
    event_description: Optional[str] = None
    themes: list[str] = Field(default_factory=list)
    economic_sector: Optional[str] = None

    # Actors
    actor1: Optional[Actor] = None
    actor2: Optional[Actor] = None

    # Location
    location: Optional[GeoLocation] = None
    locations_mentioned: list[GeoLocation] = Field(default_factory=list)

    # Quantitative measures
    goldstein_scale: Optional[float] = Field(
        None, description="GDELT Goldstein scale (-10 to +10)"
    )
    num_mentions: int = Field(default=1)
    num_sources: int = Field(default=1)
    num_articles: int = Field(default=1)

    # Tone analysis
    tone: Optional[float] = None
    positive_score: Optional[float] = None
    negative_score: Optional[float] = None
    polarity: Optional[float] = None
    activity_reference_density: Optional[float] = None
    self_group_reference_density: Optional[float] = None

    # Medallion tracking
    layer: MedallionLayer = MedallionLayer.BRONZE
    processed_at: Optional[datetime] = None

    # Raw data preservation
    raw_data: Optional[dict[str, Any]] = None


class EconomicSentiment(BaseModel):
    """
    Aggregated economic sentiment from news and media analysis.

    Captures the "mood" of economic coverage beyond raw event counts.
    """

    id: str
    timestamp: datetime
    aggregation_period: str = Field(
        default="1h", description="Time period for aggregation"
    )

    # Geographic scope
    scope: str = Field(default="global", description="global, country, region, sector")
    country_code: Optional[str] = None
    region: Optional[str] = None
    sector: Optional[str] = None

    # Sentiment metrics
    overall_tone: float = Field(..., description="Average tone (-100 to +100)")
    polarity: SentimentPolarity
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Component scores
    anxiety_score: float = Field(
        default=0.0, description="Economic anxiety indicator (0-1)"
    )
    optimism_score: float = Field(
        default=0.0, description="Economic optimism indicator (0-1)"
    )
    uncertainty_score: float = Field(
        default=0.0, description="Economic uncertainty indicator (0-1)"
    )

    # Volume metrics
    article_count: int = Field(default=0)
    source_diversity: int = Field(default=0, description="Number of unique sources")
    language_diversity: int = Field(
        default=1, description="Number of languages in coverage"
    )

    # Trend indicators
    sentiment_velocity: Optional[float] = Field(
        None, description="Rate of sentiment change"
    )
    sentiment_acceleration: Optional[float] = Field(
        None, description="Acceleration of sentiment change"
    )

    # Top themes driving sentiment
    top_themes: list[str] = Field(default_factory=list)
    top_entities: list[str] = Field(default_factory=list)

    layer: MedallionLayer = MedallionLayer.BRONZE


class TheoreticalPrediction(BaseModel):
    """
    A prediction from the Prophetic Emergentomics theoretical framework.

    Represents what the theoretical models expect to happen based on
    emergence theory, complexity economics, and adaptive system dynamics.
    """

    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Prediction content
    prediction_type: str = Field(
        ..., description="phase_transition, trend_shift, policy_impact, etc."
    )
    description: str
    hypothesis: str

    # Theoretical basis
    theoretical_framework: str = Field(
        ..., description="emergence, complexity, narrative, adaptive"
    )
    supporting_concepts: list[str] = Field(default_factory=list)

    # Quantitative expectations
    expected_direction: Optional[str] = None  # increase, decrease, volatile, stable
    expected_magnitude: Optional[str] = None  # minor, moderate, major, transformative
    expected_timeframe: Optional[str] = None  # immediate, short-term, medium-term, long-term

    # Confidence and conditions
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    preconditions: list[str] = Field(default_factory=list)
    invalidation_criteria: list[str] = Field(default_factory=list)


class EconomicContext(BaseModel):
    """
    The contextual layer for Context Augmented Generation.

    Combines theoretical predictions, real-time events, and sentiment
    to create a rich context for LLM synthesis.
    """

    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    focus_area: str = Field(
        ..., description="The economic domain or question being analyzed"
    )

    # Theoretical layer
    theoretical_predictions: list[TheoreticalPrediction] = Field(default_factory=list)
    relevant_frameworks: list[str] = Field(default_factory=list)

    # Real-time event layer
    recent_events: list[EconomicEvent] = Field(default_factory=list)
    event_clusters: list[dict[str, Any]] = Field(
        default_factory=list, description="Grouped related events"
    )

    # Sentiment layer
    sentiment_snapshot: Optional[EconomicSentiment] = None
    sentiment_trajectory: list[EconomicSentiment] = Field(default_factory=list)

    # Statistical layer
    statistical_indicators: dict[str, float] = Field(default_factory=dict)
    indicator_trends: dict[str, str] = Field(
        default_factory=dict, description="Trend direction for each indicator"
    )

    # Narrative layer
    dominant_narratives: list[str] = Field(default_factory=list)
    emerging_narratives: list[str] = Field(default_factory=list)
    narrative_conflicts: list[str] = Field(
        default_factory=list, description="Competing/contradictory narratives"
    )

    # Geographic layer
    geographic_hotspots: list[GeoLocation] = Field(default_factory=list)
    geographic_correlations: list[dict[str, Any]] = Field(default_factory=list)

    # Context quality metrics
    data_freshness_hours: float = Field(default=24.0)
    source_diversity_score: float = Field(default=0.5)
    context_completeness: float = Field(
        default=0.5, description="How complete the context is (0-1)"
    )

    layer: MedallionLayer = MedallionLayer.SILVER


class EmergenceSignal(BaseModel):
    """
    A detected signal of economic emergence.

    These are the early warnings of phase transitions, trend shifts,
    and systemic changes that traditional forecasting misses.
    """

    id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    # Signal classification
    emergence_type: EmergenceType
    description: str
    hypothesis: str = Field(
        ..., description="What this signal might indicate about economic evolution"
    )

    # Strength and confidence
    signal_strength: float = Field(
        ..., ge=0.0, le=1.0, description="How strong the signal is"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the detection"
    )
    novelty_score: float = Field(
        default=0.5, description="How novel/unprecedented this pattern is"
    )

    # Evidence
    supporting_events: list[str] = Field(
        default_factory=list, description="Event IDs supporting this signal"
    )
    supporting_sentiment: list[str] = Field(
        default_factory=list, description="Sentiment snapshot IDs"
    )
    statistical_anomalies: list[str] = Field(default_factory=list)

    # Scope
    geographic_scope: Optional[str] = None
    sector_scope: Optional[str] = None
    temporal_scope: Optional[str] = None

    # Theoretical alignment
    aligned_frameworks: list[str] = Field(
        default_factory=list, description="Which theoretical frameworks predict this"
    )
    contradicted_frameworks: list[str] = Field(default_factory=list)

    # Actionability
    monitoring_recommendations: list[str] = Field(default_factory=list)
    potential_implications: list[str] = Field(default_factory=list)

    layer: MedallionLayer = MedallionLayer.GOLD


class EconomicIntelligence(BaseModel):
    """
    The final synthesized economic intelligence output.

    This is the Gold layer output: LLM-synthesized insights that bridge
    statistical metrics with lived economic reality.
    """

    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Query context
    query: str = Field(..., description="The question or focus that generated this")
    analysis_scope: str = Field(default="general")

    # Synthesized narrative
    executive_summary: str = Field(
        ..., description="2-3 sentence synthesis of key insights"
    )
    detailed_analysis: str = Field(
        ..., description="Full LLM-generated analysis"
    )

    # Key findings
    key_insights: list[str] = Field(default_factory=list)
    blind_spots_identified: list[str] = Field(
        default_factory=list, description="What traditional metrics might miss"
    )
    narrative_vs_metrics_gap: Optional[str] = Field(
        None, description="Where lived reality diverges from statistics"
    )

    # Emergence detection
    emergence_signals: list[EmergenceSignal] = Field(default_factory=list)
    phase_transition_risk: float = Field(
        default=0.0, description="Risk of imminent phase transition"
    )

    # Forward-looking elements
    scenarios: list[dict[str, Any]] = Field(
        default_factory=list, description="Possible future trajectories"
    )
    monitoring_triggers: list[str] = Field(
        default_factory=list, description="What to watch for"
    )
    uncertainty_factors: list[str] = Field(default_factory=list)

    # Opportunities
    opportunities_detected: list[dict[str, Any]] = Field(default_factory=list)
    strategic_recommendations: list[str] = Field(default_factory=list)

    # Meta information
    context_used: Optional[str] = Field(None, description="Context ID used")
    model_used: str = Field(default="unknown")
    synthesis_confidence: float = Field(default=0.5)

    # Source attribution
    data_sources: list[str] = Field(default_factory=list)
    event_count_analyzed: int = Field(default=0)
    sentiment_snapshots_used: int = Field(default=0)

    layer: MedallionLayer = MedallionLayer.GOLD


class OpportunityAlert(BaseModel):
    """
    An actionable economic opportunity alert.

    Generated when the system detects potential opportunities before
    they become obvious to the broader market.
    """

    id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    # Opportunity classification
    opportunity_type: OpportunityType
    title: str
    description: str

    # Urgency and confidence
    urgency: str = Field(default="moderate")  # low, moderate, high, critical
    confidence: float = Field(..., ge=0.0, le=1.0)
    time_sensitivity: Optional[str] = None

    # Supporting evidence
    triggering_signals: list[str] = Field(default_factory=list)
    supporting_events: list[str] = Field(default_factory=list)
    sentiment_basis: Optional[str] = None

    # Geographic and sector scope
    geographic_focus: list[str] = Field(default_factory=list)
    sector_focus: list[str] = Field(default_factory=list)

    # Strategic guidance
    potential_actions: list[str] = Field(default_factory=list)
    risks_to_consider: list[str] = Field(default_factory=list)
    competitive_timing: Optional[str] = Field(
        None, description="How quickly others might recognize this"
    )

    # Invalidation
    invalidation_triggers: list[str] = Field(default_factory=list)
    expiration: Optional[datetime] = None

    layer: MedallionLayer = MedallionLayer.GOLD
