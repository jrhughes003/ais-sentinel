from datetime import date
from itertools import pairwise

from ais_sentinel.config import as_date, load_config


def test_default_config_loads_and_hashes_stably() -> None:
    cfg = load_config()
    assert cfg.region.lat_min < cfg.region.lat_max
    assert cfg.hash() == load_config().hash()


def test_splits_are_ordered_and_disjoint() -> None:
    cfg = load_config()
    spans = [tuple(as_date(d) for d in cfg.splits[k]) for k in ("train", "val", "test")]
    for (a0, a1), (b0, _b1) in pairwise(spans):
        assert a0 <= a1 < b0


def test_as_date_accepts_strings() -> None:
    assert as_date("2023-05-01") == date(2023, 5, 1)
