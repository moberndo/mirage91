#!/usr/bin/env python3
"""Filter + ICA-clean + epoch each .xdf run, build per-run HTML reports and one
combined HTML report (WP3 in reference/Plan.pdf: filtering, artifact removal,
re-referencing, plus an ERDS overview).

ICA is slow (~2min/run) - this is the bottleneck, not epoching/ERDS.

Example:
    python scripts/build_reports.py \
        --recordings-dir "data/mirage91/Cortical Peaks Challenge 2026/recordings" \
        --out reports/ --reference laplacian
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirage_eeg.epochs import DEFAULT_BASELINE, DEFAULT_TMAX, DEFAULT_TMIN, make_epochs
from mirage_eeg.erds import DEFAULT_CHANNELS, DEFAULT_FREQS
from mirage_eeg.io import find_xdf_runs, load_xdf_run
from mirage_eeg.preprocess import preprocess_run
from mirage_eeg.report import build_combined_report, build_run_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("data/mirage91/Cortical Peaks Challenge 2026/recordings"),
    )
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--reference", choices=["car", "laplacian"], default="car")
    parser.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--erds-channels", nargs="+", default=list(DEFAULT_CHANNELS))
    args = parser.parse_args()

    runs = find_xdf_runs(args.recordings_dir)
    if not runs:
        print(f"No .xdf files found under {args.recordings_dir}")
        return

    combined_inputs = []
    for path in runs:
        run_name = path.stem
        print(f"[{run_name}] loading {path.name}...")
        raw, events = load_xdf_run(path)

        print(f"[{run_name}] filtering + ICA (this can take ~1-2 min) + {args.reference} reference...")
        raw_clean, ica, ica_labels_df = preprocess_run(raw, reference=args.reference)
        print(f"[{run_name}] ICA excluded {len(ica.exclude)}/{ica.n_components_} components")

        print(f"[{run_name}] epoching (tmin={args.tmin}, tmax={args.tmax})...")
        epochs = make_epochs(raw_clean, events, tmin=args.tmin, tmax=args.tmax, baseline=DEFAULT_BASELINE)

        print(f"[{run_name}] computing ERDS + building report...")
        out_path = build_run_report(
            run_name,
            raw_clean,
            epochs,
            events,
            ica,
            ica_labels_df,
            args.reference,
            args.out,
            erds_channels=tuple(args.erds_channels),
        )
        print(f"[{run_name}] saved {out_path}")

        combined_inputs.append((run_name, epochs, ica))

    print("Building combined report...")
    combined_path = build_combined_report(
        combined_inputs, args.out, args.reference, erds_channels=tuple(args.erds_channels)
    )
    print(f"Saved combined report to {combined_path}")


if __name__ == "__main__":
    main()
