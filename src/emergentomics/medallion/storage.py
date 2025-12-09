"""
Medallion Data Storage for Prophetic Emergentomics.

Provides storage backends for each medallion layer with
support for persistence, caching, and querying.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import (
    EconomicContext,
    EconomicEvent,
    EconomicIntelligence,
    EconomicSentiment,
    EmergenceSignal,
    MedallionLayer,
    OpportunityAlert,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class DataStore(ABC, Generic[T]):
    """Abstract base class for medallion layer storage."""

    def __init__(self, layer: MedallionLayer):
        self.layer = layer
        self.settings = get_settings().medallion
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        for dir_path in [
            self.settings.bronze_dir,
            self.settings.silver_dir,
            self.settings.gold_dir,
            self.settings.cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        """Get the storage directory for this layer."""
        layer_dirs = {
            MedallionLayer.BRONZE: self.settings.bronze_dir,
            MedallionLayer.SILVER: self.settings.silver_dir,
            MedallionLayer.GOLD: self.settings.gold_dir,
        }
        return layer_dirs[self.layer]

    @abstractmethod
    async def save(self, item: T) -> str:
        """Save an item and return its ID."""
        pass

    @abstractmethod
    async def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[T]:
        """List items within a time range."""
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        pass


class JsonFileStore(DataStore[T]):
    """
    JSON file-based storage implementation.

    Simple but effective for development and moderate-scale usage.
    """

    def __init__(self, layer: MedallionLayer, item_type: str):
        super().__init__(layer)
        self.item_type = item_type
        self.index_file = self.storage_dir / f"{item_type}_index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the index from disk."""
        if self.index_file.exists():
            try:
                self._index = json.loads(self.index_file.read_text())
            except Exception as e:
                logger.warning("Failed to load index", error=str(e))
                self._index = {}

    def _save_index(self) -> None:
        """Save the index to disk."""
        try:
            self.index_file.write_text(json.dumps(self._index, indent=2))
        except Exception as e:
            logger.warning("Failed to save index", error=str(e))

    def _get_item_path(self, item_id: str) -> Path:
        """Get the file path for an item."""
        return self.storage_dir / self.item_type / f"{item_id}.json"

    async def save(self, item: Any) -> str:
        """Save an item to JSON file."""
        item_id = getattr(item, "id", str(datetime.utcnow().timestamp()))

        # Ensure directory exists
        item_dir = self.storage_dir / self.item_type
        item_dir.mkdir(parents=True, exist_ok=True)

        # Serialize item
        if hasattr(item, "model_dump"):
            data = item.model_dump(mode="json")
        elif hasattr(item, "dict"):
            data = item.dict()
        else:
            data = {"data": item}

        # Save to file
        file_path = self._get_item_path(item_id)
        file_path.write_text(json.dumps(data, indent=2, default=str))

        # Update index
        self._index[item_id] = {
            "timestamp": getattr(item, "timestamp", datetime.utcnow()).isoformat()
            if hasattr(getattr(item, "timestamp", None), "isoformat")
            else datetime.utcnow().isoformat(),
            "created_at": getattr(item, "created_at", datetime.utcnow()).isoformat()
            if hasattr(getattr(item, "created_at", None), "isoformat")
            else datetime.utcnow().isoformat(),
        }
        self._save_index()

        logger.debug("Saved item", item_id=item_id, layer=self.layer.value)
        return item_id

    async def get(self, item_id: str) -> Optional[dict[str, Any]]:
        """Get an item by ID."""
        file_path = self._get_item_path(item_id)
        if not file_path.exists():
            return None

        try:
            return json.loads(file_path.read_text())
        except Exception as e:
            logger.warning("Failed to read item", item_id=item_id, error=str(e))
            return None

    async def list(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List items within a time range."""
        items = []

        for item_id, meta in self._index.items():
            timestamp_str = meta.get("timestamp") or meta.get("created_at")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue

            item = await self.get(item_id)
            if item:
                items.append(item)

            if len(items) >= limit:
                break

        return items

    async def delete(self, item_id: str) -> bool:
        """Delete an item."""
        file_path = self._get_item_path(item_id)
        if file_path.exists():
            file_path.unlink()
            self._index.pop(item_id, None)
            self._save_index()
            return True
        return False

    async def cleanup_old(self, max_age_days: int) -> int:
        """Remove items older than max_age_days."""
        if max_age_days < 0:
            return 0  # Infinite retention

        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        deleted = 0

        for item_id in list(self._index.keys()):
            meta = self._index[item_id]
            timestamp_str = meta.get("timestamp") or meta.get("created_at")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp < cutoff:
                    await self.delete(item_id)
                    deleted += 1
            except ValueError:
                continue

        return deleted


class BronzeStore:
    """Storage for Bronze layer (raw data)."""

    def __init__(self):
        self.events = JsonFileStore[EconomicEvent](MedallionLayer.BRONZE, "events")
        self.sentiment = JsonFileStore[EconomicSentiment](MedallionLayer.BRONZE, "sentiment")

    async def save_events(self, events: list[EconomicEvent]) -> list[str]:
        """Save multiple events."""
        return [await self.events.save(e) for e in events]

    async def save_sentiment(self, sentiment: EconomicSentiment) -> str:
        """Save a sentiment snapshot."""
        return await self.sentiment.save(sentiment)

    async def get_recent_events(
        self,
        lookback_hours: int = 24,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent events."""
        start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        return await self.events.list(start_time=start_time, limit=limit)


class SilverStore:
    """Storage for Silver layer (cleaned, enriched data)."""

    def __init__(self):
        self.contexts = JsonFileStore[EconomicContext](MedallionLayer.SILVER, "contexts")

    async def save_context(self, context: EconomicContext) -> str:
        """Save an economic context."""
        return await self.contexts.save(context)

    async def get_context(self, context_id: str) -> Optional[dict[str, Any]]:
        """Get a context by ID."""
        return await self.contexts.get(context_id)

    async def get_recent_contexts(
        self,
        lookback_hours: int = 24,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent contexts."""
        start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        return await self.contexts.list(start_time=start_time, limit=limit)


class GoldStore:
    """Storage for Gold layer (analytics-ready intelligence)."""

    def __init__(self):
        self.intelligence = JsonFileStore[EconomicIntelligence](
            MedallionLayer.GOLD, "intelligence"
        )
        self.signals = JsonFileStore[EmergenceSignal](MedallionLayer.GOLD, "signals")
        self.opportunities = JsonFileStore[OpportunityAlert](
            MedallionLayer.GOLD, "opportunities"
        )

    async def save_intelligence(self, intel: EconomicIntelligence) -> str:
        """Save an intelligence report."""
        return await self.intelligence.save(intel)

    async def save_signal(self, signal: EmergenceSignal) -> str:
        """Save an emergence signal."""
        return await self.signals.save(signal)

    async def save_opportunity(self, opportunity: OpportunityAlert) -> str:
        """Save an opportunity alert."""
        return await self.opportunities.save(opportunity)

    async def get_intelligence(self, intel_id: str) -> Optional[dict[str, Any]]:
        """Get intelligence by ID."""
        return await self.intelligence.get(intel_id)

    async def get_recent_intelligence(
        self,
        lookback_hours: int = 24,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent intelligence reports."""
        start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        return await self.intelligence.list(start_time=start_time, limit=limit)

    async def get_active_opportunities(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get active opportunities."""
        all_opps = await self.opportunities.list(limit=limit)
        # Filter out expired opportunities
        now = datetime.utcnow()
        active = []
        for opp in all_opps:
            expiration = opp.get("expiration")
            if expiration:
                try:
                    exp_dt = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
                    if exp_dt < now:
                        continue
                except ValueError:
                    pass
            active.append(opp)
        return active
