"""
GDELT Data Parsers for Bronze → Silver layer transformation.

These parsers convert raw GDELT API responses into structured
EconomicEvent and EconomicSentiment models.
"""

import hashlib
from datetime import datetime
from typing import Any, Optional

import structlog

from emergentomics.core.models import (
    Actor,
    EconomicEvent,
    EconomicSentiment,
    GeoLocation,
    MedallionLayer,
    SentimentPolarity,
)

logger = structlog.get_logger(__name__)


class GDELTEventParser:
    """
    Parser for GDELT article/event data.

    Transforms raw GDELT DOC API responses into EconomicEvent models.
    """

    @staticmethod
    def generate_event_id(article: dict[str, Any]) -> str:
        """Generate a unique ID for an event based on its content."""
        unique_string = f"{article.get('url', '')}{article.get('seendate', '')}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

    @staticmethod
    def parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse GDELT datetime formats."""
        if not date_str:
            return None

        formats = [
            "%Y%m%dT%H%M%SZ",
            "%Y%m%d%H%M%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning("Could not parse date", date_str=date_str)
        return None

    @staticmethod
    def parse_tone(tone_str: Optional[str]) -> dict[str, Optional[float]]:
        """
        Parse GDELT tone string.

        GDELT tone format: "tone,positive,negative,polarity,activity,selfref,wordcount"
        """
        result = {
            "tone": None,
            "positive_score": None,
            "negative_score": None,
            "polarity": None,
            "activity_reference_density": None,
            "self_group_reference_density": None,
        }

        if not tone_str:
            return result

        try:
            parts = tone_str.split(",")
            if len(parts) >= 1:
                result["tone"] = float(parts[0])
            if len(parts) >= 2:
                result["positive_score"] = float(parts[1])
            if len(parts) >= 3:
                result["negative_score"] = float(parts[2])
            if len(parts) >= 4:
                result["polarity"] = float(parts[3])
            if len(parts) >= 5:
                result["activity_reference_density"] = float(parts[4])
            if len(parts) >= 6:
                result["self_group_reference_density"] = float(parts[5])
        except (ValueError, IndexError) as e:
            logger.warning("Could not parse tone", tone_str=tone_str, error=str(e))

        return result

    @staticmethod
    def parse_location(article: dict[str, Any]) -> Optional[GeoLocation]:
        """Extract location from article data."""
        # GDELT provides various location fields
        if "sourcecountry" in article or "sourcelat" in article:
            return GeoLocation(
                country_code=article.get("sourcecountry"),
                latitude=article.get("sourcelat"),
                longitude=article.get("sourcelong"),
            )
        return None

    @staticmethod
    def extract_themes(article: dict[str, Any]) -> list[str]:
        """Extract themes from article."""
        themes = article.get("themes", "")
        if isinstance(themes, str):
            return [t.strip() for t in themes.split(";") if t.strip()]
        elif isinstance(themes, list):
            return themes
        return []

    @staticmethod
    def classify_economic_sector(themes: list[str]) -> Optional[str]:
        """Classify the primary economic sector based on themes."""
        sector_mappings = {
            "ECON_TRADE": "trade",
            "ECON_BANKRUPTCY": "finance",
            "ECON_STOCKS": "finance",
            "ECON_CURRENCY": "finance",
            "ECON_INFLATION": "monetary",
            "ECON_UNEMPLOYMENT": "labor",
            "ECON_MANUFACTURING": "manufacturing",
            "TAX_": "fiscal",
            "CENTRAL_BANK": "monetary",
            "TECH_": "technology",
            "AI_": "technology",
            "LABOR_": "labor",
            "SUPPLY_CHAIN": "logistics",
        }

        for theme in themes:
            for key, sector in sector_mappings.items():
                if key in theme.upper():
                    return sector

        return None

    def parse_article(self, article: dict[str, Any]) -> EconomicEvent:
        """
        Parse a single GDELT article into an EconomicEvent.

        Args:
            article: Raw article dict from GDELT DOC API

        Returns:
            EconomicEvent model
        """
        event_id = self.generate_event_id(article)
        timestamp = self.parse_datetime(article.get("seendate"))

        if timestamp is None:
            timestamp = datetime.utcnow()

        themes = self.extract_themes(article)
        tone_data = self.parse_tone(article.get("tone"))
        location = self.parse_location(article)
        sector = self.classify_economic_sector(themes)

        return EconomicEvent(
            id=event_id,
            timestamp=timestamp,
            source_url=article.get("url"),
            source_name=article.get("domain"),
            event_type="article",
            event_description=article.get("title"),
            themes=themes,
            economic_sector=sector,
            location=location,
            goldstein_scale=None,  # DOC API doesn't provide Goldstein
            num_mentions=1,
            num_sources=1,
            num_articles=1,
            tone=tone_data["tone"],
            positive_score=tone_data["positive_score"],
            negative_score=tone_data["negative_score"],
            polarity=tone_data["polarity"],
            activity_reference_density=tone_data["activity_reference_density"],
            self_group_reference_density=tone_data["self_group_reference_density"],
            layer=MedallionLayer.BRONZE,
            processed_at=datetime.utcnow(),
            raw_data=article,
        )

    def parse_articles(self, articles: list[dict[str, Any]]) -> list[EconomicEvent]:
        """Parse a list of articles into EconomicEvents."""
        events = []
        for article in articles:
            try:
                event = self.parse_article(article)
                events.append(event)
            except Exception as e:
                logger.warning("Failed to parse article", error=str(e))
                continue
        return events


class GDELTSentimentParser:
    """
    Parser for GDELT sentiment/tone data.

    Aggregates individual article tone data into EconomicSentiment snapshots.
    """

    @staticmethod
    def tone_to_polarity(tone: float) -> SentimentPolarity:
        """Convert numeric tone to polarity category."""
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

    @staticmethod
    def calculate_anxiety_score(events: list[EconomicEvent]) -> float:
        """
        Calculate economic anxiety indicator from events.

        Looks at negative sentiment density and anxiety-related themes.
        """
        if not events:
            return 0.0

        anxiety_themes = [
            "RECESSION", "BANKRUPTCY", "UNEMPLOYMENT", "LAYOFF",
            "DEBT", "CRISIS", "PANIC", "COLLAPSE", "FEAR",
        ]

        anxiety_count = 0
        negative_tone_sum = 0.0

        for event in events:
            # Check themes
            for theme in event.themes:
                if any(at in theme.upper() for at in anxiety_themes):
                    anxiety_count += 1
                    break

            # Accumulate negative tone
            if event.negative_score:
                negative_tone_sum += event.negative_score

        theme_anxiety = min(anxiety_count / len(events), 1.0)
        tone_anxiety = min(negative_tone_sum / len(events) / 10, 1.0)

        return (theme_anxiety + tone_anxiety) / 2

    @staticmethod
    def calculate_optimism_score(events: list[EconomicEvent]) -> float:
        """Calculate economic optimism indicator."""
        if not events:
            return 0.0

        optimism_themes = [
            "GROWTH", "RECOVERY", "BOOM", "SURGE", "EXPANSION",
            "OPPORTUNITY", "INNOVATION", "PROFIT", "SUCCESS",
        ]

        optimism_count = 0
        positive_tone_sum = 0.0

        for event in events:
            for theme in event.themes:
                if any(ot in theme.upper() for ot in optimism_themes):
                    optimism_count += 1
                    break

            if event.positive_score:
                positive_tone_sum += event.positive_score

        theme_optimism = min(optimism_count / len(events), 1.0)
        tone_optimism = min(positive_tone_sum / len(events) / 10, 1.0)

        return (theme_optimism + tone_optimism) / 2

    @staticmethod
    def calculate_uncertainty_score(events: list[EconomicEvent]) -> float:
        """Calculate economic uncertainty indicator."""
        if not events:
            return 0.5

        uncertainty_themes = [
            "UNCERTAIN", "VOLATIL", "RISK", "UNSTABLE", "UNCLEAR",
            "UNKNOWN", "UNPREDICTABLE", "SPECULATION",
        ]

        uncertainty_count = 0

        for event in events:
            for theme in event.themes:
                if any(ut in theme.upper() for ut in uncertainty_themes):
                    uncertainty_count += 1
                    break

        return min(uncertainty_count / len(events), 1.0)

    def aggregate_sentiment(
        self,
        events: list[EconomicEvent],
        scope: str = "global",
        country_code: Optional[str] = None,
        sector: Optional[str] = None,
        aggregation_period: str = "1h",
    ) -> EconomicSentiment:
        """
        Aggregate events into a sentiment snapshot.

        Args:
            events: List of EconomicEvents to aggregate
            scope: Geographic scope (global, country, region, sector)
            country_code: Optional country filter
            sector: Optional sector filter
            aggregation_period: Time period label

        Returns:
            EconomicSentiment snapshot
        """
        if not events:
            return EconomicSentiment(
                id=f"sentiment_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.utcnow(),
                aggregation_period=aggregation_period,
                scope=scope,
                country_code=country_code,
                sector=sector,
                overall_tone=0.0,
                polarity=SentimentPolarity.NEUTRAL,
                confidence=0.0,
                article_count=0,
            )

        # Calculate average tone
        tones = [e.tone for e in events if e.tone is not None]
        avg_tone = sum(tones) / len(tones) if tones else 0.0

        # Get unique sources and themes
        sources = set()
        all_themes: dict[str, int] = {}
        entities: dict[str, int] = {}

        for event in events:
            if event.source_name:
                sources.add(event.source_name)
            for theme in event.themes:
                all_themes[theme] = all_themes.get(theme, 0) + 1

        # Sort themes by frequency
        top_themes = sorted(all_themes.items(), key=lambda x: -x[1])[:10]

        sentiment_id = f"sentiment_{scope}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        return EconomicSentiment(
            id=sentiment_id,
            timestamp=datetime.utcnow(),
            aggregation_period=aggregation_period,
            scope=scope,
            country_code=country_code,
            sector=sector,
            overall_tone=avg_tone,
            polarity=self.tone_to_polarity(avg_tone),
            confidence=min(len(events) / 100, 1.0),  # More events = more confidence
            anxiety_score=self.calculate_anxiety_score(events),
            optimism_score=self.calculate_optimism_score(events),
            uncertainty_score=self.calculate_uncertainty_score(events),
            article_count=len(events),
            source_diversity=len(sources),
            language_diversity=1,  # TODO: Track languages
            top_themes=[t[0] for t in top_themes],
            top_entities=[],  # TODO: Entity extraction
            layer=MedallionLayer.BRONZE,
        )
