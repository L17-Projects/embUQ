#!/usr/bin/env python3

import argparse

from meso_uq.postprocess import extract_map_from_directory


def main():
    ap = argparse.ArgumentParser(description="Extract MAP sample from a Phase 3b Korali run directory")
    ap.add_argument("run_dir")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    row = extract_map_from_directory(args.run_dir, output_csv=args.output)
    print(row.to_string(index=False))


if __name__ == "__main__":
    main()
