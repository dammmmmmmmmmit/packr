"""python -m packr.engine                       run the two built-in demos
python -m packr.engine --scenario my.json      run one scenario from a JSON file (add --json for JSON output)
"""
from __future__ import annotations

import argparse
import json

from . import Scenario, format_report, recommend

DEMOS = {
    "potato chips": Scenario("Potato Chips", moisture_pct=2, fat_pct=33, ph=6.0, shelf_life_days=180,
                             temp_c=25, rh_pct=60),
    "strawberries": Scenario("Strawberries", moisture_pct=91, fat_pct=0.3, ph=3.5, shelf_life_days=7,
                             temp_c=5, rh_pct=90, storage="Chilled", respiration="High"),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="packr.engine")
    ap.add_argument("--scenario", help="JSON file with one scenario")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the text report")
    args = ap.parse_args(argv)

    if args.scenario:
        with open(args.scenario, encoding="utf-8") as fh:
            scenarios = [Scenario.from_dict(json.load(fh))]
    else:
        scenarios = list(DEMOS.values())
    for sc in scenarios:
        rec = recommend(sc)
        print(json.dumps(rec.to_dict(), indent=2) if args.json else format_report(rec))


if __name__ == "__main__":
    main()
