from datetime import datetime, timedelta
from typing import Tuple

def get_timestamp_range(days: int = 30) -> Tuple[str, str]:
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    return start_time.isoformat(), end_time.isoformat() 