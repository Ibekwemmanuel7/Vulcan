"""
Build the pipeline's known_occurrences.csv from the occurrence register.

Only rows that are verified ("yes"), have numeric lon/lat and match the commodity
filter are exported. Everything else is listed as excluded with the reason, so the
register stays the single record of every candidate and decision.

    python scripts\\build_occurrence_csv.py --config configs\\pilot_nasarawa_keffi_li.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def build(register: Path, out: Path, commodity: str | None) -> pd.DataFrame:
    df = pd.read_csv(register, dtype=str).fillna("")
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r["verified"].strip().lower() != "yes":
            why.append("not verified")
        try:
            lon, lat = float(r["lon"]), float(r["lat"])
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                why.append("coordinates out of range")
        except ValueError:
            why.append("no coordinates")
        if commodity and commodity.lower() not in r["commodity"].lower():
            why.append(f"commodity does not contain '{commodity}'")
        reasons.append("; ".join(why))
    df["excluded_reason"] = reasons
    keep = df[df["excluded_reason"] == ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    keep[["occ_id", "name", "commodity", "lon", "lat", "evidence_type", "coord_accuracy_m"]].to_csv(out, index=False)
    df[df["excluded_reason"] != ""][["occ_id", "name", "excluded_reason"]].to_csv(
        out.with_name("occurrences_excluded.csv"), index=False)
    print(f"{len(keep)} of {len(df)} register rows exported to {out}")
    if len(df) - len(keep):
        print(f"{len(df) - len(keep)} excluded; see {out.with_name('occurrences_excluded.csv')}")
    return keep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Export verified occurrences for the pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--register", default=None, help="default: occurrence_register.csv next to positives_csv")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = cfg.path("positives_csv")
    reg = Path(args.register) if args.register else out.with_name("occurrence_register.csv")
    build(reg, out, cfg["stage2"].get("commodity_filter"))


if __name__ == "__main__":
    main()
