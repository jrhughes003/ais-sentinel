"""Command-line entry point: ``ais-sentinel <stage> [--config PATH]``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from ais_sentinel.config import Config, load_config

Stage = Callable[[Config], None]


def _stages() -> dict[str, Stage]:
    """Return the pipeline stages in execution order (imported lazily)."""
    from ais_sentinel.data import build, download
    from ais_sentinel.export import web
    from ais_sentinel.tracking import pipeline as tracking

    return {
        "download": download.stage,
        "build": build.stage,
        "track": tracking.stage,
        "export": web.stage,
    }


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the requested pipeline stage(s)."""
    stages = _stages()
    parser = argparse.ArgumentParser(prog="ais-sentinel", description=__doc__)
    parser.add_argument("stage", choices=[*stages, "run-all"])
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    selected = list(stages) if args.stage == "run-all" else [args.stage]
    for name in selected:
        logging.getLogger("ais_sentinel").info("== stage: %s (config %s)", name, cfg.hash())
        stages[name](cfg)


if __name__ == "__main__":
    main()
