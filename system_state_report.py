"""
System State Report for YamHamelach Pipeline
=============================================
Analyzes the current state of the project: file counts, match statistics,
folder locations (input/output), and last modification timestamps.

Usage:
    python system_state_report.py
    python system_state_report.py --base-path /path/to/data
"""

import os
import sys
import sqlite3
import glob
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None


def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_time(timestamp):
    """Format timestamp to readable datetime string."""
    if timestamp is None or timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def get_folder_stats(path, extensions=None):
    """Get statistics for a folder: file count, total size, last modified."""
    if not os.path.exists(path):
        return {"exists": False, "path": path}

    total_files = 0
    total_size = 0
    last_modified = 0
    ext_counts = defaultdict(int)

    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                stat = os.stat(fp)
                total_files += 1
                total_size += stat.st_size
                last_modified = max(last_modified, stat.st_mtime)
                ext = Path(f).suffix.lower()
                ext_counts[ext] += 1
            except OSError:
                pass

    return {
        "exists": True,
        "path": path,
        "file_count": total_files,
        "total_size": format_size(total_size),
        "total_size_bytes": total_size,
        "last_modified": format_time(last_modified) if last_modified > 0 else "N/A",
        "extensions": dict(ext_counts),
    }


def analyze_sqlite_db(db_path):
    """Analyze SQLite database for match statistics."""
    if not os.path.exists(db_path):
        return {"exists": False, "path": db_path}

    stats = {"exists": True, "path": db_path}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        stats["tables"] = tables

        # Matches table stats
        if "matches" in tables:
            cursor.execute("SELECT COUNT(*) FROM matches")
            stats["total_matches"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM matches WHERE match_count > 0")
            stats["positive_matches"] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(match_count), MAX(match_count), MIN(match_count) FROM matches WHERE match_count > 0")
            row = cursor.fetchone()
            stats["avg_match_count"] = round(row[0], 2) if row[0] else 0
            stats["max_match_count"] = row[1] or 0
            stats["min_match_count"] = row[2] or 0

            cursor.execute("SELECT COUNT(*) FROM matches WHERE is_validated = 1")
            stats["validated_matches"] = cursor.fetchone()[0]

            # Unique fragments
            cursor.execute("SELECT COUNT(DISTINCT file1) + COUNT(DISTINCT file2) FROM matches")
            stats["unique_fragments_approx"] = cursor.fetchone()[0]

        # Homography errors table stats
        if "homography_errors" in tables:
            cursor.execute("SELECT COUNT(*) FROM homography_errors")
            stats["total_homography_computed"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM homography_errors WHERE is_valid = 1")
            stats["valid_homography"] = cursor.fetchone()[0]

            cursor.execute("""
                SELECT AVG(mean_homo_err), MIN(mean_homo_err), MAX(mean_homo_err)
                FROM homography_errors WHERE is_valid = 1
            """)
            row = cursor.fetchone()
            stats["avg_homo_error"] = round(row[0], 4) if row[0] else 0
            stats["min_homo_error"] = round(row[1], 4) if row[1] else 0
            stats["max_homo_error"] = round(row[2], 4) if row[2] else 0

            # Quality distribution
            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN mean_homo_err < 5 THEN 1 END) as excellent,
                    COUNT(CASE WHEN mean_homo_err >= 5 AND mean_homo_err < 10 THEN 1 END) as good,
                    COUNT(CASE WHEN mean_homo_err >= 10 AND mean_homo_err < 20 THEN 1 END) as fair,
                    COUNT(CASE WHEN mean_homo_err >= 20 THEN 1 END) as poor
                FROM homography_errors WHERE is_valid = 1
            """)
            row = cursor.fetchone()
            stats["quality_distribution"] = {
                "excellent (<5)": row[0],
                "good (5-10)": row[1],
                "fair (10-20)": row[2],
                "poor (>20)": row[3],
            }

        # Processed pairs
        if "processed_pairs" in tables:
            cursor.execute("SELECT COUNT(*) FROM processed_pairs")
            stats["processed_pairs"] = cursor.fetchone()[0]

        # DB file info
        db_stat = os.stat(db_path)
        stats["db_size"] = format_size(db_stat.st_size)
        stats["db_last_modified"] = format_time(db_stat.st_mtime)

        conn.close()
    except Exception as e:
        stats["error"] = str(e)

    return stats


def analyze_cache_dir(cache_path, label="Cache"):
    """Analyze a cache directory (SIFT descriptors, homography errors)."""
    if not os.path.exists(cache_path):
        return {"exists": False, "path": cache_path, "label": label}

    pkl_files = glob.glob(os.path.join(cache_path, "**", "*.pkl"), recursive=True)
    bin_files = glob.glob(os.path.join(cache_path, "**", "*.bin"), recursive=True)
    all_cache = pkl_files + bin_files

    total_size = sum(os.path.getsize(f) for f in all_cache if os.path.exists(f))
    last_mod = max((os.path.getmtime(f) for f in all_cache), default=0)

    return {
        "exists": True,
        "path": cache_path,
        "label": label,
        "cached_items": len(all_cache),
        "pkl_files": len(pkl_files),
        "bin_files": len(bin_files),
        "total_size": format_size(total_size),
        "last_modified": format_time(last_mod) if last_mod > 0 else "N/A",
    }


def analyze_csv_files(base_path, env_config):
    """Analyze CSV match files."""
    csv_files = {}
    csv_keys = [
        "sift_matches", "sift_matches_w_tp", "sift_matches_w_tp_w_homo",
        "clean_sift_matches_w_tp_w_homo", "sift_matches_1000"
    ]
    for key in csv_keys:
        val = env_config.get(key, "")
        if val:
            csv_path = os.path.join(base_path, val)
            if os.path.exists(csv_path):
                stat = os.stat(csv_path)
                # Count lines
                try:
                    with open(csv_path, 'r') as f:
                        line_count = sum(1 for _ in f) - 1  # subtract header
                except:
                    line_count = -1
                csv_files[key] = {
                    "path": csv_path,
                    "size": format_size(stat.st_size),
                    "lines": line_count,
                    "last_modified": format_time(stat.st_mtime),
                }
            else:
                csv_files[key] = {"path": csv_path, "exists": False}
    return csv_files


def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def print_kv(key, value, indent=2):
    """Print key-value pair with indentation."""
    print(f"{' ' * indent}{key}: {value}")


def main():
    """Main entry point for system state report."""
    # Determine project root
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"

    print_section("YAMHAMELACH PIPELINE - SYSTEM STATE REPORT")
    print(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project root: {script_dir}")

    # Load .env configuration
    env_config = {}
    if dotenv_values and env_path.exists():
        raw = dotenv_values(env_path)
        env_config = {k.lower(): v for k, v in raw.items()}
    elif env_path.exists():
        # Fallback: basic parsing
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_config[k.strip().lower()] = v.strip().strip('"').strip("'")

    base_path = env_config.get("base_path", str(script_dir))
    model_type = env_config.get("model_type", "unknown")

    # ── 1. Project Source Files ──────────────────────────────
    print_section("1. PROJECT SOURCE FILES", "-")

    py_files = list(script_dir.glob("*.py"))
    # Core pipeline: f01_scrollPatchExtractor.py + all *_pipeline.py files
    CORE_PIPELINE_FILES = {"f01_scrollPatchExtractor.py"}
    pipeline_files = [f for f in py_files if f.name.endswith("_pipeline.py") or f.name in CORE_PIPELINE_FILES]
    other_py = [f for f in py_files if f.name not in {pf.name for pf in pipeline_files}]

    print(f"  Total Python files (root): {len(py_files)}")
    print(f"  Core pipeline files (relevant): {len(pipeline_files)}")
    for f in sorted(pipeline_files):
        mod = format_time(f.stat().st_mtime)
        print(f"    ✓ {f.name}  (last modified: {mod})")
    print(f"  Other Python files: {len(other_py)}")
    for f in sorted(other_py):
        print(f"    · {f.name}")

    # ── 2. Input/Output Folders ──────────────────────────────
    print_section("2. INPUT / OUTPUT FOLDER LOCATIONS", "-")

    # Input folders
    images_in = env_config.get("images_in", "")
    input_path = os.path.join(base_path, images_in) if images_in else ""
    local_input = str(script_dir / "input_dead_see_images")

    print("\n  [INPUT FOLDERS]")
    if input_path:
        stats = get_folder_stats(input_path)
        print_kv("Images input (from .env)", input_path, 4)
        if stats["exists"]:
            print_kv("Files", stats["file_count"], 6)
            print_kv("Size", stats["total_size"], 6)
            print_kv("Last addition", stats["last_modified"], 6)
            print_kv("Extensions", stats["extensions"], 6)
        else:
            print_kv("Status", "DIRECTORY NOT FOUND", 6)

    if os.path.exists(local_input):
        stats = get_folder_stats(local_input)
        print_kv("Local input folder", local_input, 4)
        print_kv("Files", stats.get("file_count", 0), 6)

    # Output folders
    output_base = os.path.join(base_path, f"OUTPUT_{model_type}") if base_path != str(script_dir) else ""
    patches_dir = env_config.get("patches_dir", "output_patches")
    bbox_dir = env_config.get("bbox_dir", "output_bbox")

    print("\n  [OUTPUT FOLDERS]")
    output_dirs = {
        "Patches (from .env)": os.path.join(output_base, patches_dir) if output_base else "",
        "BBox (from .env)": os.path.join(output_base, bbox_dir) if output_base else "",
        "Local patches (legacy copy)": str(script_dir / "output_patches"),
        "Local bbox (legacy copy)": str(script_dir / "output_bbox"),
    }

    for label, path in output_dirs.items():
        if path and os.path.exists(path):
            stats = get_folder_stats(path)
            print_kv(label, path, 4)
            print_kv("Files", stats["file_count"], 6)
            print_kv("Size", stats["total_size"], 6)
            print_kv("Last addition", stats["last_modified"], 6)

    # ── 2b. Deployment Target (DSFV) ────────────────────────
    deploy_path = env_config.get("deploy_path", "")
    if deploy_path:
        print_section("2b. DEPLOYMENT TARGET (DSFV)", "-")
        deploy_data = os.path.join(deploy_path, "fragment-explorer", "data")
        print_kv("Deploy root", deploy_path, 4)
        print_kv("Data folder", deploy_data, 4)

        if os.path.exists(deploy_data):
            # Deployed DB
            deployed_db = os.path.join(deploy_data, "matches_pruned.db")
            if os.path.exists(deployed_db):
                stat = os.stat(deployed_db)
                print(f"\n  [matches_pruned.db]")
                print_kv("Size", format_size(stat.st_size), 4)
                print_kv("Last modified", format_time(stat.st_mtime), 4)
                db_stats = analyze_sqlite_db(deployed_db)
                if "total_matches" in db_stats:
                    print_kv("Total matches", db_stats["total_matches"], 4)
                    print_kv("Positive matches (>0)", db_stats.get("positive_matches", 0), 4)
                if "total_homography_computed" in db_stats:
                    print_kv("Valid homography", db_stats.get("valid_homography", 0), 4)
                if "quality_distribution" in db_stats:
                    for q, count in db_stats["quality_distribution"].items():
                        print_kv(q, count, 6)

            # Deployed patches
            deployed_patches = os.path.join(deploy_data, "patches")
            if os.path.exists(deployed_patches):
                stats = get_folder_stats(deployed_patches)
                print(f"\n  [patches/]")
                print_kv("Files", stats["file_count"], 4)
                print_kv("Size", stats["total_size"], 4)
                print_kv("Last modified", stats["last_modified"], 4)

            # Deployed bbox
            deployed_bbox = os.path.join(deploy_data, "bbox")
            if os.path.exists(deployed_bbox):
                stats = get_folder_stats(deployed_bbox)
                print(f"\n  [bbox/]")
                print_kv("Files", stats["file_count"], 4)
                print_kv("Size", stats["total_size"], 4)
                print_kv("Last modified", stats["last_modified"], 4)

            # Total
            total_stats = get_folder_stats(deploy_data)
            print(f"\n  [TOTAL]")
            print_kv("Files", total_stats["file_count"], 4)
            print_kv("Size", total_stats["total_size"], 4)
            print_kv("Last deployment", total_stats["last_modified"], 4)
        else:
            print_kv("Status", "Not yet deployed (run f06_deploy_pipeline.py)", 4)

    # ── 3. Database Analysis ─────────────────────────────────
    print_section("3. DATABASE ANALYSIS (FULL)", "-")

    # Full pipeline DB (in Dropbox OUTPUT_faster_rcnn)
    remote_db = os.path.join(output_base, "matches.db") if output_base else ""

    for label, db_path in [("Pipeline DB (Dropbox)", remote_db)]:
        if db_path and os.path.exists(db_path):
            print(f"\n  [{label}]")
            db_stats = analyze_sqlite_db(db_path)
            print_kv("Path", db_stats["path"], 4)
            print_kv("Size", db_stats.get("db_size", "N/A"), 4)
            print_kv("Last modified", db_stats.get("db_last_modified", "N/A"), 4)
            print_kv("Tables", db_stats.get("tables", []), 4)

            if "total_matches" in db_stats:
                print(f"\n    Match Statistics:")
                print_kv("Total match entries", db_stats["total_matches"], 6)
                print_kv("Positive matches (>0)", db_stats.get("positive_matches", 0), 6)
                print_kv("Validated matches", db_stats.get("validated_matches", 0), 6)
                print_kv("Avg match count", db_stats.get("avg_match_count", 0), 6)
                print_kv("Max match count", db_stats.get("max_match_count", 0), 6)
                print_kv("Min match count", db_stats.get("min_match_count", 0), 6)
                print_kv("Unique fragments (approx)", db_stats.get("unique_fragments_approx", 0), 6)

            if "total_homography_computed" in db_stats:
                print(f"\n    Homography Statistics:")
                print_kv("Total computed", db_stats["total_homography_computed"], 6)
                print_kv("Valid homography", db_stats.get("valid_homography", 0), 6)
                print_kv("Avg error", db_stats.get("avg_homo_error", 0), 6)
                print_kv("Min error", db_stats.get("min_homo_error", 0), 6)
                print_kv("Max error", db_stats.get("max_homo_error", 0), 6)
                if "quality_distribution" in db_stats:
                    print(f"\n    Quality Distribution:")
                    for q, count in db_stats["quality_distribution"].items():
                        print_kv(q, count, 6)

            if "processed_pairs" in db_stats:
                print_kv("Processed pairs tracked", db_stats["processed_pairs"], 6)

    # ── 4. Cache Analysis ────────────────────────────────────
    print_section("4. CACHE DIRECTORIES", "-")

    patches_cache = env_config.get("patches_cache", "patches_key_dec_cache/")
    homo_cache = env_config.get("homography_cache", "homography_cache/")

    cache_dirs = {
        "SIFT descriptor cache (local)": str(script_dir / "local_data" / "patches_key_dec_cache"),
        "SIFT descriptor cache (.env)": os.path.join(base_path, patches_cache),
        "Homography error cache (.env)": os.path.join(base_path, homo_cache),
    }

    for label, path in cache_dirs.items():
        if path and os.path.exists(path):
            stats = analyze_cache_dir(path, label)
            print(f"\n  [{label}]")
            print_kv("Path", stats["path"], 4)
            print_kv("Cached items", stats["cached_items"], 4)
            print_kv("Total size", stats["total_size"], 4)
            print_kv("Last modified", stats["last_modified"], 4)

    # ── 5. Summary ───────────────────────────────────────────
    print_section("5. QUICK SUMMARY")
    print(f"  Model type: {model_type}")
    print(f"  Base data path: {base_path}")
    print(f"  Core pipeline files: {', '.join(f.name for f in sorted(pipeline_files))}")

    print("\n" + "=" * 60)
    print("  Report complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
