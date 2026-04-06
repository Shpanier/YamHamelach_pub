"""
Database Pruning Script for Fragment Matches
Reduces database size by keeping only essential data
"""

import sqlite3
import pickle
import os


def prune_database(input_db, output_db, keep_top_n=5000, min_match_count=10, max_homo_error=100):
    """
    Create a smaller version of the database keeping only the best matches.

    Args:
        input_db: Path to original database
        output_db: Path for pruned database
        keep_top_n: Keep only top N matches by homography error
        min_match_count: Minimum match count to include
        max_homo_error: Maximum homography error to include
    """

    # Remove existing output database if it exists
    if os.path.exists(output_db):
        os.remove(output_db)
        print(f"🗑️ Removed existing database: {output_db}")

    # Connect to both databases
    conn_in = sqlite3.connect(input_db)
    conn_out = sqlite3.connect(output_db)

    # Create tables in output database
    conn_out.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            file1 TEXT,
            file2 TEXT,
            match_count INTEGER,
            is_validated INTEGER DEFAULT 0,
            matches_data BLOB
        )
    ''')

    conn_out.execute('''
        CREATE TABLE IF NOT EXISTS homography_errors (
            id INTEGER PRIMARY KEY,
            match_id INTEGER,
            mean_homo_err REAL,
            std_homo_err REAL,
            max_homo_err REAL,
            min_homo_err REAL,
            median_homo_err REAL,
            is_valid INTEGER,
            len_homo_err INTEGER,
            FOREIGN KEY (match_id) REFERENCES matches(id)
        )
    ''')

    # Query to get the best matches (skip rows already pruned by f02/f03)
    query = """
    SELECT
        m.id,
        m.file1,
        m.file2,
        m.match_count,
        m.is_validated,
        m.matches_data,
        h.mean_homo_err,
        h.std_homo_err,
        h.max_homo_err,
        h.min_homo_err,
        h.median_homo_err,
        h.is_valid,
        h.len_homo_err
    FROM matches m
    LEFT JOIN homography_errors h ON m.id = h.match_id
    WHERE m.match_count >= ?
        AND (h.mean_homo_err IS NULL OR h.mean_homo_err <= ?)
        AND h.is_valid = 1
        AND m.pruned_at_stage IS NULL
    ORDER BY
        CASE WHEN h.mean_homo_err IS NULL THEN 999999 ELSE h.mean_homo_err END ASC
    LIMIT ?
    """

    cursor_in = conn_in.execute(query, (min_match_count, max_homo_error, keep_top_n))

    inserted_count = 0
    total_size_saved = 0

    for row in cursor_in:
        # Option 1: Keep matches_data (larger file)
        # Insert into matches table with matches_data
        conn_out.execute("""
            INSERT INTO matches (id, file1, file2, match_count, is_validated, matches_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row[0], row[1], row[2], row[3], row[4], row[5]))

        # Option 2: Remove matches_data to save space (much smaller file)
        # Uncomment this and comment above to save significant space
        # conn_out.execute("""
        #     INSERT INTO matches (id, file1, file2, match_count, is_validated)
        #     VALUES (?, ?, ?, ?, ?)
        # """, (row[0], row[1], row[2], row[3], row[4]))

        # Insert into homography_errors table
        if row[6] is not None:  # If homography data exists
            conn_out.execute("""
                INSERT INTO homography_errors 
                (match_id, mean_homo_err, std_homo_err, max_homo_err, 
                 min_homo_err, median_homo_err, is_valid, len_homo_err)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (row[0], row[6], row[7], row[8], row[9], row[10], row[11], row[12]))

        inserted_count += 1

        # Estimate size saved (matches_data is typically the largest field)
        if row[5]:  # matches_data
            total_size_saved += len(row[5])

    # Create indices for faster queries
    conn_out.execute("CREATE INDEX IF NOT EXISTS idx_matches_files ON matches(file1, file2)")
    conn_out.execute("CREATE INDEX IF NOT EXISTS idx_matches_count ON matches(match_count)")
    conn_out.execute("CREATE INDEX IF NOT EXISTS idx_homo_error ON homography_errors(mean_homo_err)")
    conn_out.execute("CREATE INDEX IF NOT EXISTS idx_homo_match ON homography_errors(match_id)")

    # Commit and close
    conn_out.commit()
    conn_in.close()
    conn_out.close()

    # Get file sizes
    original_size = os.path.getsize(input_db) / (1024 ** 3)  # GB
    new_size = os.path.getsize(output_db) / (1024 ** 3)  # GB

    print(f"✅ Database pruning complete!")
    print(f"📊 Original size: {original_size:.2f} GB")
    print(f"📊 New size: {new_size:.2f} GB")
    print(f"📊 Size reduction: {(1 - new_size / original_size) * 100:.1f}%")
    print(f"📊 Matches kept: {inserted_count}")
    print(f"📊 Estimated space saved from blob data: {total_size_saved / (1024 ** 3):.2f} GB")

    return inserted_count


def prune_images(matches_db, image_input_dir, image_output_dir , type_="patches"):
    """
    Copy only the images that are referenced in the pruned database.
    """
    import shutil

    conn = sqlite3.connect(matches_db)

    # Get all unique filenames from the database
    query = """
    SELECT DISTINCT file1 FROM matches
    UNION
    SELECT DISTINCT file2 FROM matches
    """

    cursor = conn.execute(query)
    filenames = [row[0] for row in cursor]
    conn.close()

    print(f"Found {len(filenames)} unique images in database")

    # Create output directory if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for filename in filenames:
        # Try to find and copy the image
        # Extract folder structure from filename
        parts = filename.split('-')
        folder = '-'.join(parts[:3]).split('_')[0]
        if type_ == "patches":

            source_path = os.path.join(image_input_dir, folder, filename)
        else:
            source_path = os.path.join(image_input_dir, folder + ".jpg")

        # If not found in folder, try root directory
        print("source_path: " , source_path)
        if not os.path.exists(source_path):
            source_path = os.path.join(image_input_dir, filename)

        if os.path.exists(source_path):
            # Maintain folder structure in output
            if len(parts) >= 3:
                output_folder = os.path.join(image_output_dir, folder)
                os.makedirs(output_folder, exist_ok=True)
                dest_path = os.path.join(output_folder, filename)
            else:
                dest_path = os.path.join(image_output_dir, filename)

            shutil.copy2(source_path, dest_path)
            copied_count += 1
        else:
            missing_count += 1
            print(f"⚠️ Missing: {filename}")

    print(f"✅ Image pruning complete!")
    print(f"📊 Images copied: {copied_count}")
    print(f"📊 Images missing: {missing_count}")

    return copied_count


def load_config(db_name: str = None):
    """Load configuration from .env file with DB profile support."""
    from db_profile import resolve_profile
    from pathlib import Path
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None

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

    profile = resolve_profile(db_name)
    output_base = profile.output_base

    patches_dir = env.get("patches_dir", "output_patches")
    bbox_dir = env.get("bbox_dir", "output_bbox")
    pruned_db = env.get("pruned_db", "matches_pruned.db")
    pruned_patches = env.get("pruned_patches_dir", "output_patches_pruned")
    pruned_bbox = env.get("pruned_bbox_dir", "output_bbox_pruned")

    return {
        "db_profile": profile.name,
        "db_label": profile.label,
        "input_db": os.path.join(output_base, "matches.db"),
        "output_db": os.path.join(output_base, pruned_db),
        "input_patches": os.path.join(output_base, patches_dir),
        "output_patches": os.path.join(output_base, pruned_patches),
        "input_bbox": os.path.join(output_base, bbox_dir),
        "output_bbox": os.path.join(output_base, pruned_bbox),
        "min_match_count": int(env.get("early_prune_min_matches", "10")),
        "max_homo_error": float(env.get("early_prune_max_homo_error", "200")),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Database Pruning Script")
    parser.add_argument("--db", type=str, default=None,
                        help="Database profile name (e.g. '180', '354')")
    args = parser.parse_args()

    config = load_config(db_name=args.db)
    print(f"Database profile: {config['db_profile']} ({config['db_label']})")

    print("🔄 Starting database pruning...")
    print(f"  Input DB:  {config['input_db']}")
    print(f"  Output DB: {config['output_db']}")

    prune_database(
        config["input_db"],
        config["output_db"],
        keep_top_n=4000,
        min_match_count=config["min_match_count"],
        max_homo_error=config["max_homo_error"],
    )

    # Prune images to match database
    print("\n🔄 Starting image pruning...")
    print("Pruning patches...")
    prune_images(config["output_db"], config["input_patches"], config["output_patches"], type_="patches")

    print("\nPruning bbox images...")
    prune_images(config["output_db"], config["input_bbox"], config["output_bbox"], type_="bbox")

    print("\n✅ All pruning complete!")
    print("Use the '_pruned' versions for deployment")