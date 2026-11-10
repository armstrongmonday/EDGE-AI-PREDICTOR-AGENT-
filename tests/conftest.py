import asyncio

import pytest

from edgeagent.config import Settings
from edgeagent.ingestion.connectors.synthetic import SyntheticSource
from edgeagent.pipeline import Pipeline
from edgeagent.storage import Store


@pytest.fixture(scope="module")
def pipe():
    p = Pipeline(Settings(), Store(":memory:"), SyntheticSource(seed=3, n_teams=12, seasons=2))
    asyncio.run(p.ensure_ready())
    return p
