"""
Economic Context Builder for CAG.

Assembles multi-layer contextual information for LLM synthesis:
1. Theoretical Layer: Prophetic Emergentomics frameworks and predictions
2. Statistical Layer: Quantitative indicators and trends
3. Event Layer: Real-time events from GDELT
4. Sentiment Layer: Aggregated mood and tone
5. Narrative Layer: Dominant and emerging stories
6. Geographic Layer: Spatial patterns and correlations
"""

import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import (
    EconomicContext,
    EconomicEvent,
    EconomicSentiment,
    GeoLocation,
    MedallionLayer,
    TheoreticalPrediction,
)
from emergentomics.gdelt.collectors import (
    EconomicEventCollector,
    SentimentCollector,
    ThemeCollector,
)

logger = structlog.get_logger(__name__)


class TheoreticalFrameworkLibrary:
    """
    Library of Prophetic Emergentomics theoretical frameworks.

    These frameworks provide the theoretical grounding for CAG analysis,
    representing the "skeleton" that LLMs add meaning to.
    """

    FRAMEWORKS = {
        "emergence_theory": {
            "name": "Emergence Theory",
            "description": "Understanding how macro-economic patterns emerge from micro-economic behaviors through self-organization and phase transitions",
            "key_concepts": [
                "complex_adaptive_systems",
                "self_organization",
                "multi_scale_interactions",
                "phase_transitions",
                "emergent_properties",
            ],
            "predictions": [
                "Traditional metrics lag emergence by design",
                "Phase transitions appear sudden but have detectable precursors",
                "Self-organization creates patterns not predictable from components",
            ],
        },
        "complexity_economics": {
            "name": "Complexity Economics",
            "description": "Moving beyond equilibrium-based models to study economies as evolving, interdependent networks",
            "key_concepts": [
                "non_linear_dynamics",
                "path_dependence",
                "feedback_loops",
                "network_effects",
                "heterogeneous_agents",
            ],
            "predictions": [
                "Small events can cascade into major shifts",
                "History matters - path dependence constrains futures",
                "Positive feedback loops accelerate change beyond linear expectation",
            ],
        },
        "narrative_economics": {
            "name": "Narrative Economics",
            "description": "How collective beliefs, information flow, and economic storytelling influence market behaviors",
            "key_concepts": [
                "collective_beliefs",
                "memetic_propagation",
                "sentiment_contagion",
                "expectation_formation",
                "narrative_asymmetry",
            ],
            "predictions": [
                "Narratives can move markets independent of fundamentals",
                "Dominant narratives create self-fulfilling prophecies",
                "Narrative conflicts signal inflection points",
            ],
        },
        "adaptive_markets": {
            "name": "Adaptive Market Hypothesis",
            "description": "Markets and economic actors evolve and adapt, with efficiency varying over time",
            "key_concepts": [
                "behavioral_adaptation",
                "institutional_evolution",
                "learning_dynamics",
                "regime_switching",
                "fitness_landscapes",
            ],
            "predictions": [
                "Market efficiency varies with conditions and participants",
                "Adaptive behavior creates new patterns",
                "Evolution pressure intensifies during stress",
            ],
        },
        "prophetic_economics": {
            "name": "Prophetic Economics",
            "description": "Foresight grounded in emergence rather than historical extrapolation",
            "key_concepts": [
                "structural_inference",
                "uncertainty_as_structure",
                "non_equilibrium_modeling",
                "evolutionary_simulation",
                "epistemic_humility",
            ],
            "predictions": [
                "The future is not an extrapolation of the past",
                "Uncertainty is information, not noise",
                "Multiple trajectories exist - we navigate possibility space",
            ],
        },
    }

    @classmethod
    def get_framework(cls, name: str) -> dict[str, Any]:
        """Get a theoretical framework by name."""
        return cls.FRAMEWORKS.get(name, {})

    @classmethod
    def get_all_frameworks(cls) -> dict[str, dict[str, Any]]:
        """Get all theoretical frameworks."""
        return cls.FRAMEWORKS

    @classmethod
    def get_relevant_frameworks(
        cls,
        themes: list[str],
        event_patterns: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Determine which frameworks are most relevant given observed patterns.

        Args:
            themes: Observed themes in data
            event_patterns: Detected event patterns

        Returns:
            List of relevant framework names
        """
        relevance_scores: dict[str, float] = {}

        # Theme-to-framework mappings
        theme_mappings = {
            "emergence_theory": ["PHASE", "TRANSITION", "EMERGENCE", "SELF_ORGAN"],
            "complexity_economics": ["NETWORK", "CASCADE", "FEEDBACK", "NONLINEAR"],
            "narrative_economics": ["SENTIMENT", "BELIEF", "NARRATIVE", "EXPECT"],
            "adaptive_markets": ["ADAPT", "EVOLV", "LEARN", "REGIME"],
            "prophetic_economics": ["UNCERTAIN", "FUTURE", "FORECAST", "PREDICT"],
        }

        for framework, keywords in theme_mappings.items():
            score = 0.0
            for theme in themes:
                for keyword in keywords:
                    if keyword.upper() in theme.upper():
                        score += 1.0

            relevance_scores[framework] = score

        # Sort by relevance
        sorted_frameworks = sorted(
            relevance_scores.items(),
            key=lambda x: -x[1]
        )

        # Return frameworks with any relevance, or top 2 if none
        relevant = [f[0] for f in sorted_frameworks if f[1] > 0]
        if not relevant:
            relevant = [sorted_frameworks[0][0], sorted_frameworks[1][0]]

        return relevant


class EconomicContextBuilder:
    """
    Builds rich contextual objects for CAG analysis.

    Assembles multiple data layers into a coherent context that
    enables LLM synthesis of statistical metrics with lived reality.
    """

    def __init__(self):
        self.settings = get_settings()
        self.event_collector = EconomicEventCollector()
        self.sentiment_collector = SentimentCollector()
        self.theme_collector = ThemeCollector()
        self.framework_library = TheoreticalFrameworkLibrary()

    def _generate_context_id(self, focus_area: str) -> str:
        """Generate unique context ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{focus_area}{timestamp}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"ctx_{timestamp}_{hash_suffix}"

    def _create_theoretical_predictions(
        self,
        focus_area: str,
        relevant_frameworks: list[str],
    ) -> list[TheoreticalPrediction]:
        """Create theoretical predictions based on relevant frameworks."""
        predictions = []

        for framework_name in relevant_frameworks:
            framework = self.framework_library.get_framework(framework_name)
            if not framework:
                continue

            for i, prediction_text in enumerate(framework.get("predictions", [])):
                prediction = TheoreticalPrediction(
                    id=f"pred_{framework_name}_{i}",
                    prediction_type="theoretical",
                    description=prediction_text,
                    hypothesis=f"Based on {framework['name']}: {prediction_text}",
                    theoretical_framework=framework_name,
                    supporting_concepts=framework.get("key_concepts", []),
                    confidence=0.7,  # Theoretical predictions have moderate confidence
                )
                predictions.append(prediction)

        return predictions

    def _cluster_events(
        self,
        events: list[EconomicEvent],
        cluster_by: str = "theme",
    ) -> list[dict[str, Any]]:
        """
        Cluster related events together.

        Args:
            events: Events to cluster
            cluster_by: Clustering dimension (theme, location, time)

        Returns:
            List of event clusters
        """
        clusters: dict[str, list[EconomicEvent]] = {}

        if cluster_by == "theme":
            for event in events:
                # Use primary theme as cluster key
                primary_theme = event.themes[0] if event.themes else "UNKNOWN"
                # Simplify theme to category
                category = primary_theme.split("_")[0] if "_" in primary_theme else primary_theme
                if category not in clusters:
                    clusters[category] = []
                clusters[category].append(event)

        elif cluster_by == "location":
            for event in events:
                location = event.location.country_code if event.location else "UNKNOWN"
                if location not in clusters:
                    clusters[location] = []
                clusters[location].append(event)

        elif cluster_by == "sector":
            for event in events:
                sector = event.economic_sector or "UNKNOWN"
                if sector not in clusters:
                    clusters[sector] = []
                clusters[sector].append(event)

        # Convert to list format with metadata
        cluster_list = []
        for key, cluster_events in clusters.items():
            if len(cluster_events) >= 2:  # Only meaningful clusters
                avg_tone = sum(e.tone or 0 for e in cluster_events) / len(cluster_events)
                cluster_list.append({
                    "cluster_key": key,
                    "cluster_type": cluster_by,
                    "event_count": len(cluster_events),
                    "event_ids": [e.id for e in cluster_events],
                    "average_tone": avg_tone,
                    "time_range": {
                        "earliest": min(e.timestamp for e in cluster_events).isoformat(),
                        "latest": max(e.timestamp for e in cluster_events).isoformat(),
                    },
                })

        return sorted(cluster_list, key=lambda x: -x["event_count"])

    def _extract_narratives(
        self,
        events: list[EconomicEvent],
        sentiment: Optional[EconomicSentiment],
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Extract dominant, emerging, and conflicting narratives.

        Returns:
            Tuple of (dominant_narratives, emerging_narratives, conflicts)
        """
        # Count theme frequencies
        theme_counts: dict[str, int] = {}
        theme_tones: dict[str, list[float]] = {}

        for event in events:
            for theme in event.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
                if event.tone is not None:
                    if theme not in theme_tones:
                        theme_tones[theme] = []
                    theme_tones[theme].append(event.tone)

        # Dominant narratives = high frequency themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: -x[1])
        dominant = [
            f"{theme} (mentioned {count} times)"
            for theme, count in sorted_themes[:5]
            if count >= 3
        ]

        # Emerging narratives = themes with positive momentum
        # (simplified: themes with increasing positive sentiment)
        emerging = []
        for theme, tones in theme_tones.items():
            if len(tones) >= 3:
                recent_avg = sum(tones[-3:]) / 3
                overall_avg = sum(tones) / len(tones)
                if recent_avg > overall_avg + 1:  # Improving sentiment
                    emerging.append(f"{theme} (sentiment improving)")

        # Conflicts = themes with high variance in tone
        conflicts = []
        for theme, tones in theme_tones.items():
            if len(tones) >= 5:
                avg = sum(tones) / len(tones)
                variance = sum((t - avg) ** 2 for t in tones) / len(tones)
                if variance > 10:  # High disagreement
                    conflicts.append(f"{theme} (mixed sentiment, variance={variance:.1f})")

        return dominant, emerging[:3], conflicts[:3]

    def _identify_geographic_hotspots(
        self,
        events: list[EconomicEvent],
    ) -> list[GeoLocation]:
        """Identify geographic areas with concentrated activity."""
        location_counts: dict[str, tuple[int, Optional[GeoLocation]]] = {}

        for event in events:
            if event.location and event.location.country_code:
                code = event.location.country_code
                current_count, _ = location_counts.get(code, (0, None))
                location_counts[code] = (current_count + 1, event.location)

        # Return locations with significant activity
        hotspots = [
            loc
            for code, (count, loc) in location_counts.items()
            if count >= 5 and loc is not None
        ]

        return hotspots[:10]

    async def build_context(
        self,
        focus_area: str,
        query: Optional[str] = None,
        lookback_hours: int = 24,
        max_events: int = 100,
        countries: Optional[list[str]] = None,
        include_sentiment: bool = True,
        include_theoretical: bool = True,
    ) -> EconomicContext:
        """
        Build a comprehensive economic context for CAG analysis.

        Args:
            focus_area: The economic domain or question being analyzed
            query: Optional search query to focus data collection
            lookback_hours: How far back to collect data
            max_events: Maximum events to include
            countries: Optional country filter
            include_sentiment: Whether to collect sentiment data
            include_theoretical: Whether to include theoretical frameworks

        Returns:
            EconomicContext with all assembled layers
        """
        logger.info(
            "Building economic context",
            focus_area=focus_area,
            lookback_hours=lookback_hours,
        )

        context_id = self._generate_context_id(focus_area)

        # Collect events
        events = await self.event_collector.collect_custom_query(
            query=query or "",
            lookback_hours=lookback_hours,
            max_events=max_events,
            countries=countries,
        )

        # Extract all themes from events
        all_themes = []
        for event in events:
            all_themes.extend(event.themes)

        # Determine relevant theoretical frameworks
        relevant_frameworks = self.framework_library.get_relevant_frameworks(all_themes)

        # Build theoretical predictions
        theoretical_predictions = []
        if include_theoretical:
            theoretical_predictions = self._create_theoretical_predictions(
                focus_area, relevant_frameworks
            )

        # Cluster events
        event_clusters = self._cluster_events(events, "theme")

        # Collect sentiment if requested
        sentiment_snapshot = None
        sentiment_trajectory = []
        if include_sentiment and events:
            from emergentomics.gdelt.parsers import GDELTSentimentParser
            parser = GDELTSentimentParser()
            sentiment_snapshot = parser.aggregate_sentiment(
                events=events,
                scope="query_specific",
                aggregation_period=f"{lookback_hours}h",
            )

        # Extract narratives
        dominant_narratives, emerging_narratives, conflicts = self._extract_narratives(
            events, sentiment_snapshot
        )

        # Identify geographic hotspots
        hotspots = self._identify_geographic_hotspots(events)

        # Calculate context quality metrics
        data_freshness = 0.0
        if events:
            newest = max(e.timestamp for e in events)
            hours_old = (datetime.utcnow() - newest).total_seconds() / 3600
            data_freshness = max(0, lookback_hours - hours_old)

        source_diversity = len(set(e.source_name for e in events if e.source_name))
        completeness = min(len(events) / max_events, 1.0) if max_events > 0 else 0.0

        return EconomicContext(
            id=context_id,
            created_at=datetime.utcnow(),
            focus_area=focus_area,
            theoretical_predictions=theoretical_predictions,
            relevant_frameworks=relevant_frameworks,
            recent_events=events[:50],  # Limit stored events
            event_clusters=event_clusters,
            sentiment_snapshot=sentiment_snapshot,
            sentiment_trajectory=sentiment_trajectory,
            statistical_indicators={},  # TODO: Add external indicators
            indicator_trends={},
            dominant_narratives=dominant_narratives,
            emerging_narratives=emerging_narratives,
            narrative_conflicts=conflicts,
            geographic_hotspots=hotspots,
            geographic_correlations=[],
            data_freshness_hours=data_freshness,
            source_diversity_score=min(source_diversity / 20, 1.0),
            context_completeness=completeness,
            layer=MedallionLayer.SILVER,
        )

    async def build_comparative_context(
        self,
        focus_areas: list[str],
        lookback_hours: int = 24,
    ) -> list[EconomicContext]:
        """
        Build contexts for multiple focus areas for comparison.

        Args:
            focus_areas: List of areas to analyze
            lookback_hours: How far back to collect data

        Returns:
            List of EconomicContext objects
        """
        import asyncio

        tasks = [
            self.build_context(area, lookback_hours=lookback_hours)
            for area in focus_areas
        ]

        return await asyncio.gather(*tasks)
