"""
Deployment Script for Fragment Viewer (Streamlit Cloud)
=======================================================
Copies the pruned pipeline output directly into the DSFV deployment folder
expected by fragment_viewer.py when running on Streamlit Cloud.

Source (pipeline output from f05):
    {OUTPUT_DIR}/matches_pruned.db          → {DEPLOY}/data/matches_pruned.db
    {OUTPUT_DIR}/output_patches_pruned/     → {DEPLOY}/data/patches/
    {OUTPUT_DIR}/output_bbox_pruned/        → {DEPLOY}/data/bbox/output_bbox/

All paths are loaded from .env (OUTPUT_DIR, PRUNED_DB, PRUNED_PATCHES_DIR,
PRUNED_BBOX_DIR, DEPLOY_PATH).

Usage:
    python f06_deploy_pipeline.py                    # Dry run (shows what would be copied)
    python f06_deploy_pipeline.py --execute          # Copy files to deployment
    python f06_deploy_pipeline.py --execute --clean  # Wipe destination first, then copy
    python f06_deploy_pipeline.py --status           # Show current deployment state only
"""

import os
import sys
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None


# ─── Configuration (loaded from .env) ───────────────────────

def load_config():
    """Load configuration from .env file."""
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"

    env = {}
    if dotenv_values and env_path.exists():
        raw = dotenv_values(env_path)
        env = {k.lower(): v for k, v in raw.items()}
    elif env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip().lower()] = v.strip().strip('"').strip("'")

    base_path = env.get("base_path", "")
    output_dir = env.get("output_dir", "OUTPUT_faster_rcnn")
    output_base = os.path.join(base_path, output_dir)

    pruned_db = env.get("pruned_db", "matches_pruned.db")
    pruned_patches = env.get("pruned_patches_dir", "output_patches_pruned")
    pruned_bbox = env.get("pruned_bbox_dir", "output_bbox_pruned")

    deploy_path = env.get("deploy_path", "")

    return {
        "output_base": output_base,
        "source_db": os.path.join(output_base, pruned_db),
        "source_patches": os.path.join(output_base, pruned_patches),
        "source_bbox": os.path.join(output_base, pruned_bbox),
        "deploy_root": deploy_path,
        "deploy_data": os.path.join(deploy_path, "fragment-explorer", "data") if deploy_path else "",
    }


# ─── Helper functions ────────────────────────────────────────

def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def format_time(ts):
    """Format timestamp to readable string."""
    if ts is None or ts == 0:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_dir_stats(path):
    """Get file count and total size for a directory."""
    if not os.path.exists(path):
        return 0, 0
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_files += 1
                total_size += os.path.getsize(fp)
            except OSError:
                pass
    return total_files, total_size


def get_db_stats(db_path):
    """Get basic stats from a SQLite database."""
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        stats = {}
        cursor.execute("SELECT COUNT(*) FROM matches")
        stats["total_matches"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM matches WHERE match_count > 0")
        stats["positive_matches"] = cursor.fetchone()[0]

        tables = [r[0] for r in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

        if "homography_errors" in tables:
            cursor.execute("SELECT COUNT(*) FROM homography_errors WHERE is_valid = 1")
            stats["valid_homography"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT file1 AS f FROM matches
                UNION
                SELECT DISTINCT file2 AS f FROM matches
            )
        """)
        stats["unique_fragments"] = cursor.fetchone()[0]

        conn.close()
        return stats
    except Exception as e:
        return {"error": str(e)}


def copy_directory(src, dest, label, dry_run=True):
    """Copy a directory tree, preserving subfolder structure."""
    if not os.path.exists(src):
        print(f"    SKIPPED (source not found)")
        return 0, 0

    file_count, total_size = get_dir_stats(src)
    print(f"    Source:      {src}")
    print(f"    Destination: {dest}")
    print(f"    Files:       {file_count}")
    print(f"    Size:        {format_size(total_size)}")

    if not dry_run:
        if os.path.exists(dest):
            copied = 0
            for root, dirs, files in os.walk(src):
                rel_path = os.path.relpath(root, src)
                dest_dir = os.path.join(dest, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                for f in files:
                    shutil.copy2(os.path.join(root, f), os.path.join(dest_dir, f))
                    copied += 1
            print(f"    Copied {copied} files")
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copytree(src, dest)
            print(f"    Copied {file_count} files")

    return file_count, total_size


def clean_deploy_data(deploy_data_path, dry_run=True):
    """Remove existing data in the deployment folder before fresh copy."""
    targets = [
        os.path.join(deploy_data_path, "matches_pruned.db"),
        os.path.join(deploy_data_path, "patches"),
        os.path.join(deploy_data_path, "bbox"),
    ]

    for target in targets:
        if os.path.exists(target):
            if os.path.isfile(target):
                print(f"    Remove file: {os.path.basename(target)} ({format_size(os.path.getsize(target))})")
                if not dry_run:
                    os.remove(target)
            elif os.path.isdir(target):
                fc, fs = get_dir_stats(target)
                print(f"    Remove dir:  {os.path.basename(target)}/ ({fc} files, {format_size(fs)})")
                if not dry_run:
                    shutil.rmtree(target)
        else:
            print(f"    (not found): {os.path.basename(target)}")


def show_deployment_status(deploy_data):
    """Show current state of the deployment target."""
    print(f"  Path: {deploy_data}")

    if not os.path.exists(deploy_data):
        print(f"  Status: NOT DEPLOYED")
        return

    # DB
    db_path = os.path.join(deploy_data, "matches_pruned.db")
    if os.path.exists(db_path):
        stat = os.stat(db_path)
        db_stats = get_db_stats(db_path)
        print(f"\n  matches_pruned.db")
        print(f"    Size:       {format_size(stat.st_size)}")
        print(f"    Modified:   {format_time(stat.st_mtime)}")
        if db_stats and "total_matches" not in db_stats.get("error", ""):
            print(f"    Matches:    {db_stats.get('total_matches', '?')} total, "
                  f"{db_stats.get('positive_matches', '?')} positive")
            print(f"    Fragments:  {db_stats.get('unique_fragments', '?')}")
            if "valid_homography" in db_stats:
                print(f"    Homography: {db_stats['valid_homography']} valid")
    else:
        print(f"\n  matches_pruned.db: MISSING")

    # Patches
    patches_path = os.path.join(deploy_data, "patches")
    if os.path.exists(patches_path):
        fc, fs = get_dir_stats(patches_path)
        last_mod = max((os.path.getmtime(os.path.join(r, f))
                       for r, _, files in os.walk(patches_path) for f in files), default=0)
        print(f"\n  patches/")
        print(f"    Files:      {fc}")
        print(f"    Size:       {format_size(fs)}")
        print(f"    Modified:   {format_time(last_mod)}")
    else:
        print(f"\n  patches/: MISSING")

    # Bbox
    bbox_path = os.path.join(deploy_data, "bbox")
    if os.path.exists(bbox_path):
        fc, fs = get_dir_stats(bbox_path)
        last_mod = max((os.path.getmtime(os.path.join(r, f))
                       for r, _, files in os.walk(bbox_path) for f in files), default=0)
        print(f"\n  bbox/")
        print(f"    Files:      {fc}")
        print(f"    Size:       {format_size(fs)}")
        print(f"    Modified:   {format_time(last_mod)}")
    else:
        print(f"\n  bbox/: MISSING")

    # Total
    total_fc, total_fs = get_dir_stats(deploy_data)
    print(f"\n  TOTAL: {total_fc} files, {format_size(total_fs)}")


# ─── Main ────────────────────────────────────────────────────

def main():
    config = load_config()

    execute = "--execute" in sys.argv
    clean = "--clean" in sys.argv
    status_only = "--status" in sys.argv

    deploy_data = config["deploy_data"]

    print("=" * 65)
    print("  YAMHAMELACH - DEPLOYMENT TO DSFV (STREAMLIT CLOUD)")
    print("=" * 65)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Status mode ─────────────────────────────────────────
    if status_only:
        print(f"\n{'─' * 65}")
        print("  CURRENT DEPLOYMENT STATE")
        print(f"{'─' * 65}")
        show_deployment_status(deploy_data)
        print("=" * 65)
        return

    # ── Deploy mode ─────────────────────────────────────────
    print(f"  Mode: {'EXECUTE' if execute else 'DRY RUN (use --execute to apply)'}")
    if clean:
        print(f"  Clean: YES - destination will be wiped first")
    print(f"  Source: {config['output_base']}")
    print(f"  Target: {deploy_data}")

    # Validate sources
    print(f"\n{'─' * 65}")
    print("  SOURCE (f05 pruned output)")
    print(f"{'─' * 65}")

    sources = [
        ("Pruned DB",      config["source_db"],      True),
        ("Pruned patches", config["source_patches"],  True),
        ("Pruned bbox",    config["source_bbox"],     True),
    ]

    missing = []
    for label, path, required in sources:
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"  OK  {label}: {format_size(os.path.getsize(path))}")
            else:
                fc, fs = get_dir_stats(path)
                print(f"  OK  {label}: {fc} files, {format_size(fs)}")
        else:
            print(f"  --  {label}: not found at {path}")
            if required:
                missing.append(label)

    if missing:
        print(f"\n  Missing sources: {', '.join(missing)}")
        print(f"  Run f05_purnning_db_pipeline.py first to generate pruned output.")
        sys.exit(1)

    # Show current deployment state
    print(f"\n{'─' * 65}")
    print("  CURRENT DEPLOYMENT STATE")
    print(f"{'─' * 65}")
    show_deployment_status(deploy_data)

    # Clean if requested
    if clean:
        print(f"\n{'─' * 65}")
        print("  CLEANING DESTINATION")
        print(f"{'─' * 65}")
        clean_deploy_data(deploy_data, dry_run=not execute)

    # Copy
    print(f"\n{'─' * 65}")
    print("  COPYING FILES")
    print(f"{'─' * 65}")

    if execute:
        os.makedirs(deploy_data, exist_ok=True)

    # 1. Database
    print(f"\n  [matches_pruned.db]")
    src_db = config["source_db"]
    dest_db = os.path.join(deploy_data, "matches_pruned.db")
    print(f"    Source:      {src_db} ({format_size(os.path.getsize(src_db))})")
    print(f"    Destination: {dest_db}")
    if execute:
        shutil.copy2(src_db, dest_db)
        print(f"    Copied successfully")

    # 2. Patches
    print(f"\n  [patches]")
    copy_directory(config["source_patches"],
                   os.path.join(deploy_data, "patches"),
                   "Patches", dry_run=not execute)

    # 3. Bbox
    print(f"\n  [bbox/output_bbox]")
    copy_directory(config["source_bbox"],
                   os.path.join(deploy_data, "bbox", "output_bbox"),
                   "Bbox", dry_run=not execute)

    # Summary
    print(f"\n{'=' * 65}")
    if execute:
        print("  DEPLOYMENT COMPLETE")
        print(f"\n{'─' * 65}")
        print("  NEW DEPLOYMENT STATE")
        print(f"{'─' * 65}")
        show_deployment_status(deploy_data)
    else:
        print("  DRY RUN COMPLETE - No files were copied.")
        print("  Run with --execute to deploy.")
        print("  Run with --execute --clean to wipe destination first.")
        print("  Run with --status to check current deployment state.")
    print("=" * 65)


if __name__ == "__main__":
    main()
