"""
Emergence detection combining multiple ML methods.

Integrates anomaly detection, clustering, and network analysis
to identify economic emergence signals. Outputs Observable-compatible
data for interactive visualization.
"""

from datetime import datetime
from typing import Any, Optional
import json
import uuid

import numpy as np

from emergentomics.core.models import (
    EmergenceSignal,
    EmergenceType,
    EmergenceReport,
    ModelType,
    DataSourceType,
    PhaseTransition,
    GDELTSentiment,
    SentimentPolarity,
)
from emergentomics.detection.anomaly import AnomalyDetector
from emergentomics.detection.clustering import ClusterAnalyzer


class EmergenceDetector:
    """
    Detects economic emergence using ML/DL methods.

    Combines:
    - Anomaly detection (isolation forest, statistical)
    - Clustering (regime identification)
    - Time series analysis (trend detection)
    - Network analysis (contagion patterns)

    All outputs are Observable-compatible for visualization.
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.7,
        cluster_shift_threshold: float = 0.3,
        trend_window: int = 7,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.cluster_shift_threshold = cluster_shift_threshold
        self.trend_window = trend_window

        self.anomaly_detector = AnomalyDetector()
        self.cluster_analyzer = ClusterAnalyzer()

    def analyze(
        self,
        data: np.ndarray,
        timestamps: Optional[list[datetime]] = None,
        feature_names: Optional[list[str]] = None,
        gdelt_events: Optional[list[dict]] = None,
    ) -> EmergenceReport:
        """
        Run full emergence analysis.

        Returns EmergenceReport with all findings.
        """
        if timestamps is None:
            timestamps = [
                datetime.utcnow().replace(hour=i % 24)
                for i in range(len(data))
            ]

        report_id = f"emergence_{uuid.uuid4().hex[:8]}"
        signals = []
        phase_transitions = []

        # 1. Anomaly Detection
        self.anomaly_detector.fit(data if data.ndim == 1 else data[:, 0])
        anomaly_results = self.anomaly_detector.detect(
            data if data.ndim == 1 else data[:, 0],
            timestamps,
            feature_names,
        )

        # Check for anomaly cascades
        anomaly_count = sum(1 for a in anomaly_results if a.is_anomaly)
        if anomaly_count >= 3:
            # Multiple anomalies = potential cascade
            signals.append(
                EmergenceSignal(
                    id=f"signal_{uuid.uuid4().hex[:8]}",
                    emergence_type=EmergenceType.ANOMALY_CASCADE,
                    description=f"Detected {anomaly_count} anomalies indicating potential cascade",
                    detection_method=ModelType.ANOMALY_DETECTION,
                    model_name="isolation_forest",
                    signal_strength=min(1.0, anomaly_count / 10),
                    confidence=0.7,
                    anomaly_scores=anomaly_results,
                    affected_indicators=feature_names or ["primary"],
                )
            )

        # 2. Clustering for regime detection
        point_ids = [f"t_{i}" for i in range(len(data))]
        if data.ndim == 1:
            # Need at least 2D for meaningful clustering
            data_2d = np.column_stack([data, np.gradient(data)])
        else:
            data_2d = data

        if len(data_2d) >= 10:
            clustering = self.cluster_analyzer.fit_predict(data_2d, point_ids)
            reduction = self.cluster_analyzer.reduce_dimensions(data_2d, point_ids)

            # Check for cluster formation/shift
            if clustering.n_clusters >= 2:
                signals.append(
                    EmergenceSignal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        emergence_type=EmergenceType.CLUSTER_FORMATION,
                        description=f"Identified {clustering.n_clusters} distinct economic regimes",
                        detection_method=ModelType.CLUSTERING,
                        model_name=clustering.method,
                        signal_strength=clustering.silhouette_score or 0.5,
                        confidence=clustering.silhouette_score or 0.5,
                        cluster_shifts=clustering,
                    )
                )
        else:
            clustering = None
            reduction = None

        # 3. Trend analysis for phase transitions
        if len(data) >= self.trend_window * 2:
            flat_data = data if data.ndim == 1 else data[:, 0]
            recent = flat_data[-self.trend_window:]
            prior = flat_data[-2 * self.trend_window : -self.trend_window]

            recent_mean = np.mean(recent)
            prior_mean = np.mean(prior)
            change_pct = abs(recent_mean - prior_mean) / (abs(prior_mean) + 1e-10)

            if change_pct > 0.2:  # 20% shift
                transition_type = "abrupt" if change_pct > 0.5 else "gradual"
                direction = "increasing" if recent_mean > prior_mean else "decreasing"

                phase_transitions.append(
                    PhaseTransition(
                        id=f"phase_{uuid.uuid4().hex[:8]}",
                        from_regime=f"stable_{direction}",
                        to_regime=f"transition_{direction}",
                        transition_type=transition_type,
                        order_parameter="primary_metric",
                        critical_value=float(prior_mean),
                        current_value=float(recent_mean),
                        confidence=min(1.0, change_pct),
                    )
                )

                signals.append(
                    EmergenceSignal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        emergence_type=EmergenceType.PHASE_TRANSITION,
                        description=f"Detected {transition_type} phase transition: {change_pct:.1%} shift",
                        detection_method=ModelType.TIME_SERIES,
                        model_name="trend_analysis",
                        signal_strength=min(1.0, change_pct * 2),
                        confidence=0.6,
                    )
                )

        # 4. Build GDELT sentiment summary if provided
        gdelt_sentiment = None
        if gdelt_events:
            tones = [e.get("tone", 0) for e in gdelt_events if e.get("tone") is not None]
            if tones:
                avg_tone = np.mean(tones)
                gdelt_sentiment = GDELTSentiment(
                    id=f"sentiment_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.utcnow(),
                    overall_tone=float(avg_tone),
                    polarity=self._tone_to_polarity(avg_tone),
                    article_count=len(gdelt_events),
                    anxiety_score=float(max(0, -avg_tone / 10)),
                    optimism_score=float(max(0, avg_tone / 10)),
                )

        # Build report
        return EmergenceReport(
            id=report_id,
            analysis_period=f"{len(data)} data points",
            gdelt_events_analyzed=len(gdelt_events) if gdelt_events else 0,
            gdelt_sentiment_summary=gdelt_sentiment,
            signals_detected=signals,
            phase_transitions=phase_transitions,
            regime_clusters=clustering,
            dimensionality_reduction=reduction,
            anomaly_rate=anomaly_count / max(len(data), 1),
            top_anomalies=[a for a in anomaly_results if a.is_anomaly][:5],
            models_used=["isolation_forest", "clustering", "trend_analysis"],
            data_sources_used=[DataSourceType.GDELT] if gdelt_events else [],
            overall_confidence=self._calc_overall_confidence(signals),
        )

    def _tone_to_polarity(self, tone: float) -> SentimentPolarity:
        """Convert GDELT tone to polarity enum."""
        if tone < -5:
            return SentimentPolarity.VERY_NEGATIVE
        elif tone < -1:
            return SentimentPolarity.NEGATIVE
        elif tone < 1:
            return SentimentPolarity.NEUTRAL
        elif tone < 5:
            return SentimentPolarity.POSITIVE
        else:
            return SentimentPolarity.VERY_POSITIVE

    def _calc_overall_confidence(self, signals: list[EmergenceSignal]) -> float:
        """Calculate overall confidence from signals."""
        if not signals:
            return 0.5
        return float(np.mean([s.confidence for s in signals]))

    def to_observable(self, report: EmergenceReport) -> dict[str, Any]:
        """
        Export EmergenceReport to Observable-compatible format.

        This is the primary output for visualization.
        """
        # Anomaly time series for line chart
        anomaly_series = [
            {
                "timestamp": a.timestamp.isoformat(),
                "score": a.score,
                "is_anomaly": a.is_anomaly,
            }
            for a in report.top_anomalies
        ]

        # Signals for card/list display
        signals_data = [
            {
                "id": s.id,
                "type": s.emergence_type.value,
                "description": s.description,
                "strength": s.signal_strength,
                "confidence": s.confidence,
                "method": s.detection_method.value,
                "timestamp": s.detected_at.isoformat(),
            }
            for s in report.signals_detected
        ]

        # Cluster scatter plot data
        scatter_data = []
        if report.regime_clusters and report.dimensionality_reduction:
            for i, coord in enumerate(report.dimensionality_reduction.coordinates):
                cluster_id = -1
                if i < len(report.regime_clusters.assignments):
                    cluster_id = report.regime_clusters.assignments[i].cluster_id
                scatter_data.append({
                    "x": coord[0] if len(coord) > 0 else 0,
                    "y": coord[1] if len(coord) > 1 else 0,
                    "cluster": cluster_id,
                    "id": report.dimensionality_reduction.point_ids[i],
                })

        # Phase transitions timeline
        transitions = [
            {
                "id": t.id,
                "from": t.from_regime,
                "to": t.to_regime,
                "type": t.transition_type,
                "confidence": t.confidence,
                "timestamp": t.detected_at.isoformat(),
            }
            for t in report.phase_transitions
        ]

        # GDELT sentiment summary
        gdelt_summary = None
        if report.gdelt_sentiment_summary:
            gs = report.gdelt_sentiment_summary
            gdelt_summary = {
                "tone": gs.overall_tone,
                "polarity": gs.polarity.value,
                "articles": gs.article_count,
                "anxiety": gs.anxiety_score,
                "optimism": gs.optimism_score,
            }

        return {
            "metadata": {
                "report_id": report.id,
                "generated_at": report.generated_at.isoformat(),
                "analysis_period": report.analysis_period,
                "overall_confidence": report.overall_confidence,
                "models_used": report.models_used,
                "gdelt_events": report.gdelt_events_analyzed,
            },
            "summary": {
                "total_signals": len(report.signals_detected),
                "phase_transitions": len(report.phase_transitions),
                "anomaly_rate": report.anomaly_rate,
                "n_clusters": report.regime_clusters.n_clusters if report.regime_clusters else 0,
            },
            "signals": signals_data,
            "anomalies": anomaly_series,
            "clusters": {
                "scatter": scatter_data,
                "n_clusters": report.regime_clusters.n_clusters if report.regime_clusters else 0,
                "silhouette": report.regime_clusters.silhouette_score if report.regime_clusters else None,
                "sizes": report.regime_clusters.cluster_sizes if report.regime_clusters else [],
            },
            "transitions": transitions,
            "gdelt": gdelt_summary,
            "network": {
                "metrics": report.network_metrics,
                "critical_nodes": report.critical_nodes,
            },
        }

    def to_json(self, report: EmergenceReport, path: Optional[str] = None) -> str:
        """Export report to JSON for Observable."""
        data = self.to_observable(report)
        json_str = json.dumps(data, indent=2)

        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str


def run_emergence_analysis(
    gdelt_events: list[dict],
    additional_data: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Convenience function to run emergence analysis on GDELT data.

    Returns Observable-compatible output dictionary.
    """
    # Extract tone time series from GDELT
    tones = []
    timestamps = []

    for event in gdelt_events:
        tone = event.get("tone")
        if tone is not None:
            tones.append(float(tone))
            ts = event.get("timestamp", event.get("date"))
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.utcnow()
            elif ts is None:
                ts = datetime.utcnow()
            timestamps.append(ts)

    if len(tones) < 10:
        return {
            "metadata": {"error": "insufficient_data", "count": len(tones)},
            "summary": {},
            "signals": [],
            "anomalies": [],
            "clusters": {},
        }

    data = np.array(tones)

    detector = EmergenceDetector()
    report = detector.analyze(
        data,
        timestamps=timestamps,
        feature_names=["gdelt_tone"],
        gdelt_events=gdelt_events,
    )

    return detector.to_observable(report)
