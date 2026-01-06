"""
Anomaly detection for economic emergence signals.

Uses isolation forest, autoencoders, and statistical methods
to detect anomalies in economic time series and GDELT data.
"""

from datetime import datetime
from typing import Any, Optional
import json

import numpy as np
from pydantic import BaseModel

from emergentomics.core.models import AnomalyScore, EmergenceType


class AnomalyDetector:
    """
    Detects anomalies in economic data streams.

    Outputs Observable-compatible JSON for visualization.
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "AnomalyDetector":
        """Fit the anomaly detection model."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            scaled_data = self._scaler.fit_transform(
                data.reshape(-1, 1) if data.ndim == 1 else data
            )

            self._model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
            )
            self._model.fit(scaled_data)
            self._fitted = True

        except ImportError:
            # Fallback to statistical method if sklearn not available
            self._mean = np.mean(data)
            self._std = np.std(data)
            self._fitted = True

        return self

    def detect(
        self,
        data: np.ndarray,
        timestamps: Optional[list[datetime]] = None,
        feature_names: Optional[list[str]] = None,
    ) -> list[AnomalyScore]:
        """
        Detect anomalies in the data.

        Returns list of AnomalyScore objects.
        """
        if timestamps is None:
            timestamps = [
                datetime.utcnow().replace(microsecond=0)
                for _ in range(len(data))
            ]

        results = []

        if self._fitted and self._model is not None:
            # Use sklearn model
            try:
                from sklearn.preprocessing import StandardScaler

                scaled = self._scaler.transform(
                    data.reshape(-1, 1) if data.ndim == 1 else data
                )
                scores_raw = self._model.decision_function(scaled)
                predictions = self._model.predict(scaled)

                # Normalize scores to 0-1 range (higher = more anomalous)
                scores = 1 - (scores_raw - scores_raw.min()) / (
                    scores_raw.max() - scores_raw.min() + 1e-10
                )

                for i, (ts, score, pred) in enumerate(
                    zip(timestamps, scores, predictions)
                ):
                    is_anomaly = pred == -1

                    # Feature contributions (simplified)
                    contributions = {}
                    if feature_names and data.ndim > 1:
                        for j, name in enumerate(feature_names):
                            contributions[name] = float(abs(scaled[i, j]))

                    results.append(
                        AnomalyScore(
                            timestamp=ts,
                            score=float(score),
                            method=self.method,
                            feature_contributions=contributions,
                            is_anomaly=is_anomaly,
                            anomaly_type="isolation_forest_outlier" if is_anomaly else None,
                        )
                    )

            except Exception:
                # Fallback
                results = self._statistical_detect(data, timestamps)

        else:
            results = self._statistical_detect(data, timestamps)

        return results

    def _statistical_detect(
        self,
        data: np.ndarray,
        timestamps: list[datetime],
    ) -> list[AnomalyScore]:
        """Statistical anomaly detection fallback."""
        mean = getattr(self, "_mean", np.mean(data))
        std = getattr(self, "_std", np.std(data))

        results = []
        for i, (ts, val) in enumerate(zip(timestamps, data)):
            z_score = abs(val - mean) / (std + 1e-10)
            score = min(1.0, z_score / 3.0)  # Normalize to 0-1
            is_anomaly = z_score > 2.5

            results.append(
                AnomalyScore(
                    timestamp=ts,
                    score=float(score),
                    method="statistical_zscore",
                    feature_contributions={"z_score": float(z_score)},
                    is_anomaly=is_anomaly,
                    anomaly_type="statistical_outlier" if is_anomaly else None,
                )
            )

        return results

    def to_observable(self, results: list[AnomalyScore]) -> dict[str, Any]:
        """
        Export results to Observable-compatible format.

        Returns a dictionary that can be JSON-serialized for Observable.
        """
        return {
            "metadata": {
                "method": self.method,
                "contamination": self.contamination,
                "generated_at": datetime.utcnow().isoformat(),
                "total_points": len(results),
                "anomaly_count": sum(1 for r in results if r.is_anomaly),
                "anomaly_rate": sum(1 for r in results if r.is_anomaly) / max(len(results), 1),
            },
            "data": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "score": r.score,
                    "is_anomaly": r.is_anomaly,
                    "anomaly_type": r.anomaly_type,
                    "contributions": r.feature_contributions,
                }
                for r in results
            ],
            "anomalies": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "score": r.score,
                    "type": r.anomaly_type,
                }
                for r in results
                if r.is_anomaly
            ],
        }

    def to_json(self, results: list[AnomalyScore], path: Optional[str] = None) -> str:
        """Export to JSON file or string for Observable."""
        data = self.to_observable(results)
        json_str = json.dumps(data, indent=2)

        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str


def detect_anomalies_in_gdelt(
    events: list[dict],
    field: str = "tone",
) -> dict[str, Any]:
    """
    Convenience function to detect anomalies in GDELT data.

    Returns Observable-compatible output.
    """
    if not events:
        return {"metadata": {"error": "no_data"}, "data": [], "anomalies": []}

    # Extract values
    values = []
    timestamps = []
    for event in events:
        if field in event and event[field] is not None:
            values.append(float(event[field]))
            ts = event.get("timestamp", event.get("date", datetime.utcnow().isoformat()))
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.utcnow()
            timestamps.append(ts)

    if not values:
        return {"metadata": {"error": "no_valid_values"}, "data": [], "anomalies": []}

    data = np.array(values)

    detector = AnomalyDetector()
    detector.fit(data)
    results = detector.detect(data, timestamps)

    return detector.to_observable(results)
