import json
import logging
from typing import Dict, List, Optional, Any
from app.services.vector_service import VectorService, get_vector_service
from app.providers.defi_llama_provider import get_market_data
from app.providers.types.market_types import MarketData
from datetime import datetime, timedelta
import numpy as np
from app.utils.time import get_timestamp_range


logger = logging.getLogger("MARKET_SERVICE")
logging.basicConfig(level=logging.INFO)

class MarketService:
    """
    Service for managing market data fetching, storage, and retrieval.
    """

    def __init__(self):
        self.vector_service = get_vector_service()

    @staticmethod
    async def validate_market_data(market_data: Dict) -> Dict:
        """
        Validates and sanitizes the structure of market data.

        Args:
            market_data (Dict): The raw market data.

        Returns:
            Dict: The validated and sanitized market data.
        """
        if not isinstance(market_data, dict):
            raise ValueError(f"Market data must be a dictionary. Got: {type(market_data)}")

        if "timestamp" not in market_data or not isinstance(market_data["timestamp"], str) or not market_data["timestamp"]:
            market_data["timestamp"] = datetime.utcnow().isoformat()

        if "protocols" in market_data and not isinstance(market_data["protocols"], dict):
            market_data["protocols"] = {}

        return market_data

    @staticmethod
    async def fetch_market_data() -> Dict:
        """
        Fetches market data from the provider asynchronously.

        Returns:
            Dict: The fetched market data as a dictionary.
        """
        try:
            data = await get_market_data()

            if isinstance(data, list):
                data = data[0] if data else {}

            if isinstance(data, MarketData):
                from dataclasses import asdict
                data = asdict(data)

            return await MarketService.validate_market_data(data)
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}

    @staticmethod
    async def save_market_data(market_data: Dict) -> None:
        """
        Saves market data to Pinecone.

        Args:
            market_data (Dict): The raw market data to save.
        """
        try:
            if not isinstance(market_data, dict):
                raise ValueError("Market data must be a dictionary.")

            if "timestamp" not in market_data:
                raise ValueError("Market data is missing the 'timestamp' key.")

            key = f"market_data_{market_data['timestamp'][:19].replace('-', '_').replace(':', '_')}"
            text = f"Market data snapshot generated at {market_data['timestamp']}."

            metadata = {
                "type": "market_data",
                "timestamp": market_data["timestamp"],
                "description": "Market data snapshot for protocols.",
                "data": json.dumps(market_data)
            }

            await VectorService.save(key=key, metadata=metadata, text=text)
            logger.info(f"Market data saved successfully with key: {key}")
        except Exception as e:
            logger.error(f"Error saving market data to Pinecone: {e}")
            raise

    @staticmethod
    async def get_latest_market_data() -> Dict:
        """
        Retrieves the latest market data from Pinecone.

        Returns:
            Dict: The most recent market data.
        """
        try:
            results = await VectorService.natural_query("Latest market data", limit=1, min_score=0.5)
            logger.info(f"Natural query results: {results}")

            if results:
                return results[0]["metadata"]["data"]
            return {}
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}

