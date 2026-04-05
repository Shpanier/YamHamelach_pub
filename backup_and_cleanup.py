"""
Backup & Cleanup Script for YamHamelach Pipeline
=================================================
Moves all non-essential files into a timestamped ZIP archive,
keeping only the relevant _pipeline.py files and essential config.

Usage:
    python backup_and_cleanup.py                  # Dry run (shows what would be moved)
    python backup_and_cleanup.py --execute        # Actually moves files into ZIP
    python backup_and_cleanup.py --execute --delete  # Move to ZIP and delete originals

The script categorizes files as:
- KEEP: Essential pipeline files, config, and active data
- BACKUP: Deprecated, duplicated, or old-version files
- JUNK: Empty/accidental files to be removed
"""

import os
import sys
import zipfile
import shutil
from datetime import datetime
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────

# Files/patterns that are ESSENTIAL and must NOT be moved
KEEP_FILES = {
    # Core pipeline files
    "f01_scrollPatchExtractor.py",
    "f02_sift_matching_pipeline.py",
    "f03_Homography_matching_pipeline.py",
    "f04_plot_matching_pipeline.py",
    "f05_purnning_db_pipeline.py",
    "f06_deploy_pipeline.py",
    # Essential config and support
    "env_arguments_loader.py",
    ".env",
    ".gitignore",
    "__init__.py",
    "requirements.txt",
    "packages.txt",
    # This script and the state report
    "system_state_report.py",
    "backup_and_cleanup.py",
    # Note: matches.db is in Dropbox BASE_PATH, not local
}

# Directories to KEEP (not back up)
KEEP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
}

# Files that are pure JUNK (accidental, empty, broken)
JUNK_FILES = {
    "0",        # Accidental empty file
    "=",        # Accidental empty file
    "atches",   # Truncated "matches" filename
    ".DS_Store",
}

# File extensions that belong to the old CSV-based system (back up any at root level)
BACKUP_EXTENSIONS = {".csv"}

# Deprecated/old version files to BACK UP
BACKUP_FILES = {
    # Old CSV-based pipeline files (replaced by _pipeline.py versions)
    "f02_SIFT_matcher.py",
    "f03_Homography_matcher.py",
    "f03_homography_error_calculator.py",
    "f02a_add_tp_soft_SIFT_macher.py",
    "f03a_add_homoScore_SIFT_macher.py",
    # Debug variant of pipeline
    "f03_Homography_matching_pipeline_w_debug.py",
    # Duplicate/copy files
    "f03b_ViewHomoApp copy.py",
    "f04_plotv2_matching_pipeline.py",
    # Multiple viewer app versions
    "f03b_ViewHomoApp.py",
    "f03b_ViewHomoApp_v2.py",
    "f03b_ViewHomoApp_dropbox.py",
    # Old PAM matcher versions
    "PAM_matcher.py",
    "PAM_macher_v2.py",
    # Upload scripts
    "upload_to_dropbox_v2.py",
    "upload_to_dropbox_v3.py",
    # Utility scripts (old CSV workflow)
    "modify_csv_files.py",
    "convert_csv_to_database.py",
    "prepare_match_dataset_v2.py",
    "aux_function.py",
    # Feature extraction (potentially unused)
    "feature_extractor.py",
    "feature_model.py",
    # Training scripts
    "train_yolo.py",
    # Test files
    "test.py",
    # Visualization helper
    "f04_generate_all_matches_v5.py",
    # Typo README
    "REAMDE.MD",
    # Old CSV data
    "fragment_matches.csv",
    # Empty local DB (real DB is in Dropbox BASE_PATH/OUTPUT_faster_rcnn/)
    "matches.db",
    # Generated plot
    "myplot.png",
}

# Directories to BACK UP entirely
BACKUP_DIRS = {
    "BK",                          # Explicit backup folder
    "DSFV",                        # Empty/unused directory
    "db_manager",                  # Alternative PostgreSQL DB manager
    "postgres_docker",             # Docker config for PostgreSQL
    ".devcontainer",               # Dev container config
    ".idea",                       # IDE settings
    "input_dead_see_images",       # Empty input dir
    "readme_supplementary_images", # README images
    "tools",                       # Faster R-CNN tools
    "output_patches",              # Local patches (old copy, real output is in Dropbox BASE_PATH)
    "output_bbox",                 # Local bbox (old copy, real output is in Dropbox BASE_PATH)
    "local_data",                  # Empty local cache dir (real cache is in Dropbox BASE_PATH)
}


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent


def categorize_files(project_root):
    """Categorize all files and directories in the project root."""
    keep = []
    backup = []
    junk = []
    unknown = []

    # Process root-level files
    for item in sorted(project_root.iterdir()):
        name = item.name

        if item.is_file():
            if name in JUNK_FILES:
                junk.append(item)
            elif name in BACKUP_FILES:
                backup.append(item)
            elif item.suffix.lower() in BACKUP_EXTENSIONS:
                # All .csv files are part of the old system
                backup.append(item)
            elif name in KEEP_FILES:
                keep.append(item)
            else:
                # Unknown files - report but don't touch
                unknown.append(item)

        elif item.is_dir():
            if name in KEEP_DIRS:
                keep.append(item)
            elif name in BACKUP_DIRS:
                backup.append(item)
            else:
                unknown.append(item)

    return keep, backup, junk, unknown


def create_backup_zip(project_root, backup_items, junk_items, zip_path, delete_originals=False):
    """Create a ZIP archive with all backup and junk items."""
    total_files = 0
    total_size = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item in backup_items:
            if item.is_file():
                arcname = os.path.join("backup", item.name)
                zf.write(item, arcname)
                total_files += 1
                total_size += item.stat().st_size
                print(f"  [BACKUP] {item.name}")
            elif item.is_dir():
                for root, dirs, files in os.walk(item):
                    for f in files:
                        fp = Path(root) / f
                        arcname = os.path.join("backup", item.name, fp.relative_to(item))
                        zf.write(fp, arcname)
                        total_files += 1
                        total_size += fp.stat().st_size
                print(f"  [BACKUP] {item.name}/ ({sum(1 for _ in item.rglob('*') if _.is_file())} files)")

        for item in junk_items:
            if item.is_file():
                arcname = os.path.join("junk", item.name)
                zf.write(item, arcname)
                total_files += 1
                total_size += item.stat().st_size
                print(f"  [JUNK]   {item.name}")

    # Report ZIP stats
    zip_size = os.path.getsize(zip_path)
    print(f"\n  ZIP created: {zip_path}")
    print(f"  Files archived: {total_files}")
    print(f"  Original size: {total_size / (1024*1024):.2f} MB")
    print(f"  ZIP size: {zip_size / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {(1 - zip_size/total_size)*100:.1f}%" if total_size > 0 else "")

    # Delete originals if requested
    if delete_originals:
        print(f"\n  Deleting originals...")
        for item in backup_items + junk_items:
            if item.is_file():
                item.unlink()
                print(f"    Deleted: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"    Deleted: {item.name}/")
        print("  Originals deleted.")

    return total_files, total_size


def main():
    execute = "--execute" in sys.argv
    delete = "--delete" in sys.argv

    project_root = get_project_root()

    print("=" * 60)
    print("  YAMHAMELACH - BACKUP & CLEANUP")
    print("=" * 60)
    print(f"  Project: {project_root}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'EXECUTE' if execute else 'DRY RUN (use --execute to apply)'}")
    if delete:
        print(f"  WARNING: --delete flag set - originals will be removed!")

    keep, backup, junk, unknown = categorize_files(project_root)

    # ── Report ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  KEEP ({len(keep)} items) - Essential files, will NOT be touched:")
    print(f"{'─' * 60}")
    for item in keep:
        suffix = "/" if item.is_dir() else ""
        print(f"    ✓ {item.name}{suffix}")

    print(f"\n{'─' * 60}")
    print(f"  BACKUP ({len(backup)} items) - Will be archived to ZIP:")
    print(f"{'─' * 60}")
    for item in backup:
        suffix = "/" if item.is_dir() else ""
        print(f"    → {item.name}{suffix}")

    print(f"\n{'─' * 60}")
    print(f"  JUNK ({len(junk)} items) - Accidental/broken files:")
    print(f"{'─' * 60}")
    for item in junk:
        print(f"    ✗ {item.name}")

    if unknown:
        print(f"\n{'─' * 60}")
        print(f"  UNKNOWN ({len(unknown)} items) - Not categorized, left as-is:")
        print(f"{'─' * 60}")
        for item in unknown:
            suffix = "/" if item.is_dir() else ""
            print(f"    ? {item.name}{suffix}")

    # ── Execute ──────────────────────────────────────────
    if execute and (backup or junk):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"backup_{timestamp}.zip"
        zip_path = project_root / zip_name

        print(f"\n{'=' * 60}")
        print(f"  CREATING BACKUP ARCHIVE")
        print(f"{'=' * 60}")

        total_files, total_size = create_backup_zip(
            project_root, backup, junk, zip_path, delete_originals=delete
        )

        print(f"\n{'=' * 60}")
        print(f"  CLEANUP COMPLETE")
        print(f"  Archive: {zip_name}")
        if not delete:
            print(f"  Note: Original files were NOT deleted.")
            print(f"  Run with --execute --delete to also remove originals.")
        print(f"{'=' * 60}")

    elif not execute:
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN COMPLETE - No files were modified.")
        print(f"  Run with --execute to create the backup ZIP.")
        print(f"  Run with --execute --delete to also remove originals.")
        print(f"{'=' * 60}")
    else:
        print(f"\n  Nothing to back up or clean.")


if __name__ == "__main__":
    main()
