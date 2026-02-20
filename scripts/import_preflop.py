#!/usr/bin/env python3
"""Import preflop GTO strategies from CSV files into the SQLite database.

CSV File Format:
    Each CSV file represents one "spot" — a unique combination of
    (position, action_sequence, stack_depth). The filename encodes the spot:

        {POSITION}_{ACTION_SEQUENCE}_{STACK}bb.csv

    Examples:
        SB_open_100bb.csv
        BB_vs_sb_limp_100bb.csv
        CO_vs_raise_50bb.csv
        BTN_open_40bb.csv

    CSV columns:
        hand       — Required. Hand notation (e.g. "AA", "AKs", "T7o")
        ev         — Optional. Expected value in big blinds.
        <action>   — One or more action columns. Any column that isn't
                     "hand" or "ev" is treated as an action. The column
                     name IS the action string and values are frequencies [0, 1].

    Action column names:
        fold, limp, call, raise_2.0, raise_2.5, raise_3.0, raise_all_in
        (or any other action string your solver outputs)

    Example CSV (SB_open_100bb.csv):
        hand,fold,limp,raise_2.5,ev
        AA,0.00,0.00,1.00,0.52
        KK,0.00,0.00,1.00,0.45
        T7o,0.17,0.83,0.00,-0.08
        72o,0.92,0.08,0.00,-0.42

    Frequencies per row should sum to ~1.0. Rows with all zeros are skipped.
    Actions with frequency < 0.001 are skipped (noise removal).

Usage:
    # Import a single file
    python scripts/import_preflop.py data/SB_open_100bb.csv

    # Import all CSVs in a directory
    python scripts/import_preflop.py data/preflop_csvs/

    # Import to a custom database path
    python scripts/import_preflop.py data/ --db solver/data/preflop.db

    # Dry run (parse and validate without writing)
    python scripts/import_preflop.py data/ --dry-run

    # List what's currently in the database
    python scripts/import_preflop.py --list
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Add project root to path so we can import poker_bot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_bot.solver.preflop_db import PreflopDB, STACK_BUCKETS


# Filename pattern: POSITION_ACTION-SEQUENCE_STACKbb.csv
_FILENAME_PATTERN = re.compile(
    r"^([A-Z]{2,3})_(.+?)_(\d+)bb\.csv$"
)

_VALID_POSITIONS = {"UTG", "MP", "CO", "BTN", "SB", "BB"}

_MIN_FREQUENCY = 0.001  # Skip actions below this threshold


def parse_filename(path: Path) -> tuple[str, str, str] | None:
    """Extract (position, action_sequence, stack_bucket) from filename.

    Returns None if the filename doesn't match the expected pattern.
    """
    match = _FILENAME_PATTERN.match(path.name)
    if not match:
        return None

    position = match.group(1)
    action_seq = match.group(2)
    stack = match.group(3)

    if position not in _VALID_POSITIONS:
        return None

    stack_bucket = f"{stack}bb"
    return position, action_seq, stack_bucket


def import_csv(
    csv_path: Path,
    db: PreflopDB,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Import a single CSV file into the database.

    Returns:
        Tuple of (hands_imported, actions_imported).
    """
    parsed = parse_filename(csv_path)
    if parsed is None:
        print(f"  SKIP {csv_path.name} — filename doesn't match pattern")
        return 0, 0

    position, action_seq, stack_bucket = parsed

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        if "hand" not in reader.fieldnames:
            print(f"  SKIP {csv_path.name} — missing 'hand' column")
            return 0, 0

        # All columns except 'hand' and 'ev' are action columns
        action_columns = [
            col for col in reader.fieldnames
            if col not in ("hand", "ev")
        ]

        if not action_columns:
            print(f"  SKIP {csv_path.name} — no action columns found")
            return 0, 0

        rows = []
        hands_count = 0

        for row_data in reader:
            hand = row_data["hand"].strip()
            if not hand:
                continue

            ev = float(row_data.get("ev", 0.0) or 0.0)
            hands_count += 1

            for action_col in action_columns:
                freq_str = row_data.get(action_col, "0").strip()
                try:
                    freq = float(freq_str)
                except ValueError:
                    continue

                if freq < _MIN_FREQUENCY:
                    continue

                rows.append((
                    position, action_seq, stack_bucket,
                    hand, action_col, freq, ev,
                ))

        if not dry_run and rows:
            db.insert_batch(rows)

        return hands_count, len(rows)


def import_path(
    target: Path,
    db: PreflopDB,
    dry_run: bool = False,
) -> None:
    """Import a single CSV file or all CSVs in a directory."""
    if target.is_file():
        csv_files = [target]
    elif target.is_dir():
        csv_files = sorted(target.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {target}")
            return
    else:
        print(f"Path not found: {target}")
        return

    total_hands = 0
    total_actions = 0
    total_files = 0

    for csv_path in csv_files:
        hands, actions = import_csv(csv_path, db, dry_run=dry_run)
        if hands > 0:
            tag = "[DRY RUN] " if dry_run else ""
            print(f"  {tag}{csv_path.name}: {hands} hands, {actions} action rows")
            total_hands += hands
            total_actions += actions
            total_files += 1

    print(f"\nImported {total_files} files: {total_hands} hands, {total_actions} action rows")
    if not dry_run:
        print(f"Database: {db.path} ({db.row_count()} total rows)")


def list_db(db: PreflopDB) -> None:
    """Print a summary of what's in the database."""
    spots = db.list_spots()
    if not spots:
        print("Database is empty.")
        return

    print(f"Database: {db.path} ({db.row_count()} total rows)\n")
    print(f"{'Position':<10} {'Action Sequence':<20} {'Stack':<10} {'Hands':<8}")
    print("-" * 50)
    for pos, action_seq, stack, num_hands in spots:
        print(f"{pos:<10} {action_seq:<20} {stack:<10} {num_hands:<8}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import preflop GTO strategies from CSV files into SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path", nargs="?", type=Path,
        help="CSV file or directory to import",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to the SQLite database (default: solver/data/preflop.db)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate without writing to the database",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List spots currently in the database",
    )
    args = parser.parse_args()

    db = PreflopDB(db_path=args.db)

    try:
        if args.list:
            list_db(db)
        elif args.path:
            import_path(args.path, db, dry_run=args.dry_run)
        else:
            parser.print_help()
    finally:
        db.close()


if __name__ == "__main__":
    main()
