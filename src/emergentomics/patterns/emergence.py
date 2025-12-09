"""
Emergence Pattern Detection for Prophetic Emergentomics.

Implements algorithms for detecting:
- Phase transitions in economic systems
- Trend accelerations and momentum shifts
- Narrative pattern evolution
- Self-organization signals
"""

from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from emergentomics.core.models import (
    EconomicContext,
    EconomicEvent,
    EconomicSentiment,
    EmergenceSignal,
    EmergenceType,
    MedallionLayer,
)

logger = structlog.get_logger(__name__)


class EmergencePatternDetector:
    """
    Detects general emergence patterns in economic data.

    Emergence patterns are macro-level behaviors that arise from
    micro-level interactions in ways that are not predictable
    from the components alone.
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Args:
            sensitivity: Detection sensitivity (0-1). Higher = more signals.
        """
        self.sensitivity = sensitivity

    def detect_self_organization(
        self,
        events: list[EconomicEvent],
        time_window_hours: int = 24,
    ) -> list[EmergenceSignal]:
        """
        Detect self-organization patterns in events.

        Self-organization appears as coordinated behavior without
        explicit coordination - multiple actors moving similarly.
        """
        if len(events) < 10:
            return []

        signals = []

        # Group events by theme and check for coordination
        theme_events: dict[str, list[EconomicEvent]] = {}
        for event in events:
            for theme in event.themes:
                if theme not in theme_events:
                    theme_events[theme] = []
                theme_events[theme].append(event)

        for theme, theme_group in theme_events.items():
            if len(theme_group) < 5:
                continue

            # Check for tone convergence (self-organization signal)
            tones = [e.tone for e in theme_group if e.tone is not None]
            if len(tones) < 3:
                continue

            avg_tone = sum(tones) / len(tones)
            variance = sum((t - avg_tone) ** 2 for t in tones) / len(tones)

            # Low variance = high coordination = self-organization
            if variance < 5 * (1 - self.sensitivity):
                signals.append(EmergenceSignal(
                    id=f"selforg_{theme[:10]}_{datetime.utcnow().strftime('%H%M%S')}",
                    detected_at=datetime.utcnow(),
                    emergence_type=EmergenceType.PHASE_TRANSITION,
                    description=f"Self-organization detected in {theme}",
                    hypothesis=(
                        f"Coordinated sentiment ({avg_tone:.1f} avg, {variance:.1f} var) "
                        f"across {len(theme_group)} events suggests emergent consensus."
                    ),
                    signal_strength=min(1 - variance / 10, 0.9),
                    confidence=min(len(theme_group) / 15, 0.8),
                    novelty_score=0.6,
                    supporting_events=[e.id for e in theme_group[:5]],
                    sector_scope=theme,
                    aligned_frameworks=["emergence_theory", "complexity_economics"],
                    layer=MedallionLayer.GOLD,
                ))

        return signals

    def detect_feedback_loops(
        self,
        sentiments: list[EconomicSentiment],
    ) -> list[EmergenceSignal]:
        """
        Detect positive feedback loops in sentiment data.

        Feedback loops appear as accelerating trends that feed on themselves.
        """
        if len(sentiments) < 3:
            return []

        signals = []

        # Sort by timestamp
        sorted_sentiments = sorted(sentiments, key=lambda s: s.timestamp)

        # Calculate sentiment velocity
        velocities = []
        for i in range(1, len(sorted_sentiments)):
            prev = sorted_sentiments[i - 1]
            curr = sorted_sentiments[i]
            velocity = curr.overall_tone - prev.overall_tone
            velocities.append(velocity)

        if len(velocities) < 2:
            return []

        # Check for acceleration (increasing velocity magnitude)
        accelerating = True
        for i in range(1, len(velocities)):
            if abs(velocities[i]) <= abs(velocities[i - 1]):
                accelerating = False
                break

        if accelerating and abs(velocities[-1]) > 2:
            direction = "positive" if velocities[-1] > 0 else "negative"
            signals.append(EmergenceSignal(
                id=f"feedback_{direction}_{datetime.utcnow().strftime('%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.TREND_ACCELERATION,
                description=f"Accelerating {direction} sentiment feedback loop",
                hypothesis=(
                    f"Sentiment velocity increasing ({velocities[-1]:.1f}/period). "
                    "May indicate self-reinforcing narrative or cascade effect."
                ),
                signal_strength=min(abs(velocities[-1]) / 5, 0.9),
                confidence=0.6,
                novelty_score=0.7,
                supporting_sentiment=[s.id for s in sorted_sentiments[-3:]],
                aligned_frameworks=["complexity_economics", "narrative_economics"],
                layer=MedallionLayer.GOLD,
            ))

        return signals


class PhaseTransitionDetector:
    """
    Detects phase transitions in economic systems.

    Phase transitions are sudden regime changes where the system
    shifts from one stable state to another.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: Detection threshold (0-1). Higher = fewer detections.
        """
        self.threshold = threshold

    def detect_regime_change(
        self,
        events: list[EconomicEvent],
        lookback_events: Optional[list[EconomicEvent]] = None,
    ) -> Optional[EmergenceSignal]:
        """
        Detect potential regime changes in event patterns.

        Looks for structural breaks in event characteristics.
        """
        if len(events) < 20:
            return None

        # Split into halves and compare
        mid = len(events) // 2
        recent = events[:mid]
        earlier = events[mid:]

        # Calculate tone statistics
        recent_tones = [e.tone for e in recent if e.tone is not None]
        earlier_tones = [e.tone for e in earlier if e.tone is not None]

        if len(recent_tones) < 5 or len(earlier_tones) < 5:
            return None

        recent_avg = sum(recent_tones) / len(recent_tones)
        earlier_avg = sum(earlier_tones) / len(earlier_tones)

        # Significant shift in average tone
        shift = abs(recent_avg - earlier_avg)
        if shift > 5 * self.threshold:
            direction = "improving" if recent_avg > earlier_avg else "deteriorating"
            return EmergenceSignal(
                id=f"regime_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.PHASE_TRANSITION,
                description=f"Potential regime change detected: {direction} sentiment",
                hypothesis=(
                    f"Sentiment shifted from {earlier_avg:.1f} to {recent_avg:.1f} "
                    f"(delta={shift:.1f}). May indicate fundamental change in "
                    "market conditions or narrative regime."
                ),
                signal_strength=min(shift / 10, 0.9),
                confidence=min(len(events) / 50, 0.8),
                novelty_score=0.8,
                supporting_events=[e.id for e in recent[:5]],
                aligned_frameworks=["adaptive_markets", "complexity_economics"],
                monitoring_recommendations=[
                    "Track if new regime stabilizes",
                    "Monitor for reversal signals",
                    "Watch for policy responses",
                ],
                potential_implications=[
                    f"Market may be entering new {direction} phase",
                    "Historical correlations may break down",
                    "Volatility likely during transition",
                ],
                layer=MedallionLayer.GOLD,
            )

        return None

    def detect_threshold_crossing(
        self,
        sentiment: EconomicSentiment,
        history: Optional[list[EconomicSentiment]] = None,
    ) -> Optional[EmergenceSignal]:
        """
        Detect crossing of critical thresholds.

        Certain threshold crossings (e.g., anxiety > 0.8) often
        precede significant market events.
        """
        thresholds = {
            "extreme_anxiety": (sentiment.anxiety_score > 0.8, "anxiety"),
            "extreme_uncertainty": (sentiment.uncertainty_score > 0.85, "uncertainty"),
            "extreme_pessimism": (sentiment.overall_tone < -8, "pessimism"),
            "extreme_optimism": (sentiment.overall_tone > 8, "optimism"),
        }

        for signal_name, (condition, metric) in thresholds.items():
            if condition:
                return EmergenceSignal(
                    id=f"threshold_{signal_name}_{datetime.utcnow().strftime('%H%M%S')}",
                    detected_at=datetime.utcnow(),
                    emergence_type=EmergenceType.PHASE_TRANSITION,
                    description=f"Critical threshold crossed: extreme {metric}",
                    hypothesis=(
                        f"Extreme {metric} level reached. Historically associated "
                        "with elevated probability of regime change or reversal."
                    ),
                    signal_strength=0.85,
                    confidence=0.7,
                    novelty_score=0.6,
                    supporting_sentiment=[sentiment.id],
                    aligned_frameworks=["adaptive_markets", "narrative_economics"],
                    monitoring_recommendations=[
                        f"Monitor for {metric} reversal",
                        "Watch for policy responses",
                        "Track real economy indicators for confirmation",
                    ],
                    layer=MedallionLayer.GOLD,
                )

        return None


class NarrativePatternAnalyzer:
    """
    Analyzes narrative patterns and their evolution.

    Based on narrative economics: stories shape economic reality.
    """

    def analyze_narrative_momentum(
        self,
        context: EconomicContext,
    ) -> dict[str, Any]:
        """
        Analyze the momentum of different narratives.

        Returns:
            Dict with narrative analysis results
        """
        result = {
            "dominant_narratives": [],
            "emerging_narratives": [],
            "fading_narratives": [],
            "narrative_conflicts": [],
            "momentum_score": 0.0,
        }

        if context.dominant_narratives:
            result["dominant_narratives"] = [
                {"narrative": n, "status": "dominant"}
                for n in context.dominant_narratives
            ]

        if context.emerging_narratives:
            result["emerging_narratives"] = [
                {"narrative": n, "status": "emerging", "momentum": "increasing"}
                for n in context.emerging_narratives
            ]
            # Higher momentum score with more emerging narratives
            result["momentum_score"] = min(len(context.emerging_narratives) / 5, 1.0)

        if context.narrative_conflicts:
            result["narrative_conflicts"] = [
                {"conflict": c, "implication": "uncertainty"}
                for c in context.narrative_conflicts
            ]

        return result

    def detect_narrative_shift(
        self,
        current_context: EconomicContext,
        previous_context: Optional[EconomicContext] = None,
    ) -> Optional[EmergenceSignal]:
        """
        Detect shifts in dominant narratives.

        Narrative shifts often precede economic shifts.
        """
        if not current_context.emerging_narratives:
            return None

        # Check if emerging narratives contradict dominant ones
        contradictions = []
        for emerging in current_context.emerging_narratives:
            for dominant in current_context.dominant_narratives:
                # Simple heuristic: opposite sentiment words
                if any(word in emerging.lower() for word in ["growth", "recovery", "boom"]):
                    if any(word in dominant.lower() for word in ["recession", "decline", "crisis"]):
                        contradictions.append((emerging, dominant))
                elif any(word in emerging.lower() for word in ["recession", "decline", "crisis"]):
                    if any(word in dominant.lower() for word in ["growth", "recovery", "boom"]):
                        contradictions.append((emerging, dominant))

        if contradictions:
            return EmergenceSignal(
                id=f"narrative_shift_{datetime.utcnow().strftime('%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.NARRATIVE_SHIFT,
                description="Emerging narratives contradicting dominant narrative",
                hypothesis=(
                    f"Conflict between emerging ({contradictions[0][0]}) and "
                    f"dominant ({contradictions[0][1]}) narratives suggests "
                    "potential narrative regime change."
                ),
                signal_strength=min(len(contradictions) / 3, 0.9),
                confidence=0.6,
                novelty_score=0.7,
                aligned_frameworks=["narrative_economics"],
                potential_implications=[
                    "Market consensus may be shifting",
                    "Expect increased volatility during transition",
                    "Early positioning opportunities may exist",
                ],
                layer=MedallionLayer.GOLD,
            )

        return None

    def detect_narrative_contagion(
        self,
        events: list[EconomicEvent],
    ) -> Optional[EmergenceSignal]:
        """
        Detect narrative contagion across sectors/regions.

        Contagion occurs when a narrative spreads rapidly across
        previously unrelated domains.
        """
        if len(events) < 20:
            return None

        # Track theme spread across sectors
        theme_sectors: dict[str, set[str]] = {}
        for event in events:
            sector = event.economic_sector or "unknown"
            for theme in event.themes:
                if theme not in theme_sectors:
                    theme_sectors[theme] = set()
                theme_sectors[theme].add(sector)

        # Find themes spreading across many sectors
        spreading_themes = [
            (theme, len(sectors))
            for theme, sectors in theme_sectors.items()
            if len(sectors) >= 3
        ]

        if spreading_themes:
            top_theme, sector_count = max(spreading_themes, key=lambda x: x[1])
            return EmergenceSignal(
                id=f"contagion_{datetime.utcnow().strftime('%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.SECTOR_CONTAGION,
                description=f"Narrative contagion: {top_theme} spreading across {sector_count} sectors",
                hypothesis=(
                    f"Theme '{top_theme}' appearing across {sector_count} distinct sectors. "
                    "Suggests cross-sector narrative propagation."
                ),
                signal_strength=min(sector_count / 5, 0.9),
                confidence=0.65,
                novelty_score=0.75,
                supporting_events=[e.id for e in events[:10]],
                aligned_frameworks=["narrative_economics", "complexity_economics"],
                layer=MedallionLayer.GOLD,
            )

        return None
