import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from dagcache.policy import configure, reset_config
from dagcache.store import _STORES


@pytest.fixture(autouse=True)
def fresh_config(tmp_path):
    reset_config()
    _STORES.clear()
    configure(db_path=str(tmp_path / "store.db"))
    yield
    reset_config()
    _STORES.clear()
