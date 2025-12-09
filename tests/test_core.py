"""Core module tests for Prophetic Emergentomics."""

import pytest
from datetime import datetime

from emergentomics.core.config import Settings, get_settings, reset_settings
from emergentomics.core.models import (
    EconomicEvent,
    EconomicSentiment,
    EconomicContext,
    EmergenceSignal,
    EmergenceType,
    MedallionLayer,
    SentimentPolarity,
)


class TestSettings:
    """Tests for configuration settings."""

    def test_default_settings(self):
        """Test that default settings load correctly."""
        reset_settings()
        settings = get_settings()

        assert settings.app_name == "Prophetic Emergentomics"
        assert settings.app_version == "0.1.0"
        assert settings.enable_gdelt_integration is True
        assert settings.enable_llm_synthesis is True

    def test_gdelt_settings(self):
        """Test GDELT settings defaults."""
        settings = get_settings()

        assert settings.gdelt.requests_per_minute == 60
        assert settings.gdelt.max_retries == 3
        assert len(settings.gdelt.economic_themes) > 0

    def test_cag_settings(self):
        """Test CAG settings defaults."""
        settings = get_settings()

        assert settings.cag.synthesis_depth in ["shallow", "moderate", "deep"]
        assert 0 <= settings.cag.emergence_threshold <= 1
        assert 0 <= settings.cag.anomaly_sensitivity <= 1


class TestModels:
    """Tests for data models."""

    def test_economic_event_creation(self):
        """Test EconomicEvent model creation."""
        event = EconomicEvent(
            id="test_event_001",
            timestamp=datetime.utcnow(),
            event_type="article",
            event_description="Test economic event",
            themes=["ECON_INFLATION", "CENTRAL_BANK"],
            economic_sector="monetary",
            tone=-2.5,
            layer=MedallionLayer.BRONZE,
        )

        assert event.id == "test_event_001"
        assert event.event_type == "article"
        assert len(event.themes) == 2
        assert event.tone == -2.5
        assert event.layer == MedallionLayer.BRONZE

    def test_economic_sentiment_creation(self):
        """Test EconomicSentiment model creation."""
        sentiment = EconomicSentiment(
            id="sentiment_001",
            timestamp=datetime.utcnow(),
            overall_tone=-3.5,
            polarity=SentimentPolarity.NEGATIVE,
            anxiety_score=0.7,
            optimism_score=0.2,
            uncertainty_score=0.6,
            article_count=100,
        )

        assert sentiment.polarity == SentimentPolarity.NEGATIVE
        assert sentiment.anxiety_score == 0.7
        assert sentiment.article_count == 100

    def test_emergence_signal_creation(self):
        """Test EmergenceSignal model creation."""
        signal = EmergenceSignal(
            id="signal_001",
            emergence_type=EmergenceType.PHASE_TRANSITION,
            description="Potential regime change detected",
            hypothesis="Market may be entering new phase",
            signal_strength=0.75,
            confidence=0.65,
            layer=MedallionLayer.GOLD,
        )

        assert signal.emergence_type == EmergenceType.PHASE_TRANSITION
        assert signal.signal_strength == 0.75
        assert signal.confidence == 0.65

    def test_medallion_layer_enum(self):
        """Test MedallionLayer enum values."""
        assert MedallionLayer.BRONZE.value == "bronze"
        assert MedallionLayer.SILVER.value == "silver"
        assert MedallionLayer.GOLD.value == "gold"

    def test_sentiment_polarity_enum(self):
        """Test SentimentPolarity enum values."""
        assert SentimentPolarity.VERY_NEGATIVE.value == "very_negative"
        assert SentimentPolarity.NEUTRAL.value == "neutral"
        assert SentimentPolarity.VERY_POSITIVE.value == "very_positive"

    def test_emergence_type_enum(self):
        """Test EmergenceType enum values."""
        assert EmergenceType.PHASE_TRANSITION.value == "phase_transition"
        assert EmergenceType.NARRATIVE_SHIFT.value == "narrative_shift"
        assert EmergenceType.SENTIMENT_DIVERGENCE.value == "sentiment_divergence"


class TestContextBuilder:
    """Tests for context building functionality."""

    def test_theoretical_framework_library(self):
        """Test theoretical framework library."""
        from emergentomics.cag.context_builder import TheoreticalFrameworkLibrary

        frameworks = TheoreticalFrameworkLibrary.get_all_frameworks()

        assert "emergence_theory" in frameworks
        assert "complexity_economics" in frameworks
        assert "narrative_economics" in frameworks
        assert "prophetic_economics" in frameworks

        emergence = TheoreticalFrameworkLibrary.get_framework("emergence_theory")
        assert emergence["name"] == "Emergence Theory"
        assert len(emergence["predictions"]) > 0

    def test_relevant_frameworks_detection(self):
        """Test detection of relevant frameworks based on themes."""
        from emergentomics.cag.context_builder import TheoreticalFrameworkLibrary

        # Themes related to narrative economics
        themes = ["SENTIMENT_ANALYSIS", "BELIEF_CHANGE", "NARRATIVE_SHIFT"]
        relevant = TheoreticalFrameworkLibrary.get_relevant_frameworks(themes)

        assert "narrative_economics" in relevant


class TestEmergenceDetector:
    """Tests for emergence detection."""

    def test_sentiment_divergence_detection(self):
        """Test sentiment divergence signal detection."""
        from emergentomics.intelligence.detector import EmergenceDetector

        detector = EmergenceDetector()

        # Create high anxiety sentiment
        sentiment = EconomicSentiment(
            id="test_sent",
            timestamp=datetime.utcnow(),
            overall_tone=0.5,  # Neutral tone
            polarity=SentimentPolarity.NEUTRAL,
            anxiety_score=0.75,  # High anxiety
            optimism_score=0.3,
            uncertainty_score=0.5,
            article_count=50,
        )

        signal = detector.detect_sentiment_divergence(sentiment)

        assert signal is not None
        assert signal.emergence_type == EmergenceType.SENTIMENT_DIVERGENCE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
