import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_data_dir():
    """Crée un répertoire temporaire pour les tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_traffic_data():
    """Données de test pour le trafic"""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.DataFrame(
        {
            "hour": np.random.randint(0, 24, 100),
            "day_of_week": np.random.randint(0, 7, 100),
            "weather_score": np.random.uniform(0, 1, 100),
            "event_impact": np.random.uniform(0, 1, 100),
            "historical_avg": np.random.uniform(0.3, 0.9, 100),
            "traffic_level": np.random.uniform(0, 1, 100),
        }
    )


@pytest.fixture
def api_client():
    """Client de test pour l'API"""
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)
