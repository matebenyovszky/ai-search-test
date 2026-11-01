import pytest
import benchmark


@pytest.fixture(autouse=True)
def fixed_benchmark_date(monkeypatch):
    # Freshness tests must remain deterministic after the real case expires.
    monkeypatch.setattr(benchmark, 'today', lambda: '2026-11-01')
