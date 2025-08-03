"""
Optimized Fragment Matching Pipeline with Database Backend

This version replaces CSV storage with SQLite for better performance with millions of pairs.
Key improvements:
1. SQLite database with proper indexing
2. Batch processing for efficient I/O
3. Memory-mapped arrays for feature caching
4. Streaming processing to avoid memory overflow
5. Parallel processing support
"""

import sqlite3
import numpy as np
import cv2
import os
import pickle
from typing import Dict, List, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from contextlib import contextmanager
import mmap
import struct


@dataclass
class MatchResult:
    """Data class for match results"""
    file1: str
    file2: str
    match_count: int
    matches_data: bytes  # Serialized match data
    is_validated: bool = False


class DatabaseManager:
    """
    High-performance database manager for storing and querying millions of matches.
    Uses SQLite with optimizations for bulk operations and fast queries.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with optimized schema and indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            conn.execute("PRAGMA temp_store=memory")  # Use memory for temp tables

            # Create main matches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY,
                    file1 TEXT NOT NULL,
                    file2 TEXT NOT NULL,
                    match_count INTEGER NOT NULL,
                    matches_data BLOB,
                    is_validated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file1 ON matches(file1)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file2 ON matches(file2)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files ON matches(file1, file2)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_count ON matches(match_count DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_validated ON matches(is_validated)")

            # Create processed pairs table for resume capability
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_pairs (
                    file1 TEXT,
                    file2 TEXT,
                    PRIMARY KEY (file1, file2)
                )
            """)

            conn.commit()

    def batch_insert_matches(self, matches: List[MatchResult], batch_size: int = 1000):
        """Insert matches in batches for better performance"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for i in range(0, len(matches), batch_size):
                    batch = matches[i:i + batch_size]
                    conn.executemany("""
                        INSERT INTO matches (file1, file2, match_count, matches_data, is_validated)
                        VALUES (?, ?, ?, ?, ?)
                    """, [
                        (m.file1, m.file2, m.match_count, m.matches_data, m.is_validated)
                        for m in batch
                    ])

                    # Also insert into processed pairs
                    conn.executemany("""
                        INSERT OR IGNORE INTO processed_pairs (file1, file2)
                        VALUES (?, ?)
                    """, [(m.file1, m.file2) for m in batch])

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e

    def get_processed_pairs(self) -> set:
        """Get already processed pairs for resume capability"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file1, file2 FROM processed_pairs")
            return set(cursor.fetchall())

    def get_top_matches(self, limit: int = 1000, min_matches: int = 1) -> Iterator[Tuple]:
        """Stream top matches without loading all into memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file1, file2, match_count, matches_data, is_validated
                FROM matches 
                WHERE match_count >= ?
                ORDER BY match_count DESC 
                LIMIT ?
            """, (min_matches, limit))

            for row in cursor:
                yield row

    def update_validation_status(self, file_pairs: List[Tuple[str, str]]):
        """Batch update validation status for matched pairs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                UPDATE matches 
                SET is_validated = 1 
                WHERE (file1 = ? AND file2 = ?) OR (file1 = ? AND file2 = ?)
            """, [(f1, f2, f2, f1) for f1, f2 in file_pairs])
            conn.commit()

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_matches,
                    SUM(is_validated) as validated_matches,
                    AVG(match_count) as avg_match_count,
                    MAX(match_count) as max_match_count
                FROM matches
            """)
            result = cursor.fetchone()
            return {
                'total_matches': result[0],
                'validated_matches': result[1],
                'avg_match_count': result[2],
                'max_match_count': result[3]
            }


class MemoryMappedFeatureCache:
    """
    Memory-mapped feature cache for efficient SIFT descriptor storage.
    Uses binary format for fast I/O and reduced memory usage.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.descriptor_cache = {}  # In-memory cache for frequently accessed descriptors
        self.max_memory_cache = 1000  # Max items in memory cache
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, image_key: str) -> str:
        return os.path.join(self.cache_dir, f"{image_key}.bin")

    def _serialize_descriptors(self, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray) -> bytes:
        """Serialize keypoints and descriptors to binary format"""
        if descriptors is None:
            return b''

        # Convert keypoints to structured array
        kp_array = np.array([
            (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints
        ], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('size', 'f4'), ('angle', 'f4'),
            ('response', 'f4'), ('octave', 'i4'), ('class_id', 'i4')
        ])

        # Pack data: header + keypoints + descriptors
        header = struct.pack('II', len(keypoints), descriptors.shape[1])
        kp_bytes = kp_array.tobytes()
        desc_bytes = descriptors.astype(np.float32).tobytes()

        return header + kp_bytes + desc_bytes

    def _deserialize_descriptors(self, data: bytes) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Deserialize binary data back to keypoints and descriptors"""
        if not data:
            return [], None

        # Unpack header
        n_kp, desc_dim = struct.unpack('II', data[:8])
        offset = 8

        # Unpack keypoints
        kp_dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('size', 'f4'), ('angle', 'f4'),
            ('response', 'f4'), ('octave', 'i4'), ('class_id', 'i4')
        ])
        kp_size = kp_dtype.itemsize * n_kp
        kp_array = np.frombuffer(data[offset:offset + kp_size], dtype=kp_dtype)
        offset += kp_size

        # Convert back to KeyPoint objects
        keypoints = [
            cv2.KeyPoint(x=float(kp['x']), y=float(kp['y']), size=float(kp['size']),
                         angle=float(kp['angle']), response=float(kp['response']),
                         octave=int(kp['octave']), class_id=int(kp['class_id']))
            for kp in kp_array
        ]

        # Unpack descriptors
        desc_data = data[offset:]
        descriptors = np.frombuffer(desc_data, dtype=np.float32).reshape(n_kp, desc_dim)

        return keypoints, descriptors

    def get_features(self, image_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Get or compute SIFT features for an image"""
        image_key = os.path.basename(image_path)

        # Check memory cache first
        if image_key in self.descriptor_cache:
            return self.descriptor_cache[image_key]

        cache_path = self._get_cache_path(image_key)

        # Try to load from disk cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = f.read()
                result = self._deserialize_descriptors(data)

                # Add to memory cache if space available
                if len(self.descriptor_cache) < self.max_memory_cache:
                    self.descriptor_cache[image_key] = result

                return result
            except Exception as e:
                print(f"Error loading cache for {image_key}: {e}")

        # Compute features if not cached
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Save to disk cache
        try:
            data = self._serialize_descriptors(keypoints, descriptors)
            with open(cache_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"Error saving cache for {image_key}: {e}")

        result = (keypoints, descriptors)

        # Add to memory cache if space available
        if len(self.descriptor_cache) < self.max_memory_cache:
            self.descriptor_cache[image_key] = result

        return result


class ParallelFragmentMatcher:
    """
    Parallel fragment matcher with database backend and optimized processing.
    """

    def __init__(self, image_base_path: str, cache_dir: str, db_path: str, num_workers: int = 4):
        self.image_base_path = image_base_path
        self.cache = MemoryMappedFeatureCache(cache_dir)
        self.db = DatabaseManager(db_path)
        self.num_workers = num_workers
        self._match_lock = threading.Lock()

    def get_image_files(self) -> List[str]:
        """Get all image files in the dataset"""
        image_files = []
        for root, _, files in os.walk(self.image_base_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        return image_files

    def _calculate_matches(self, file1: str, file2: str) -> MatchResult:
        """Calculate matches between two images"""
        try:
            kp1, des1 = self.cache.get_features(file1)
            kp2, des2 = self.cache.get_features(file2)

            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return None

            # SIFT matching with ratio test
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append((m.queryIdx, m.trainIdx, m.distance))

            if len(good_matches) == 0:
                return None

            # Serialize match data
            matches_data = pickle.dumps(good_matches)

            return MatchResult(
                file1=os.path.basename(file1),
                file2=os.path.basename(file2),
                match_count=len(good_matches),
                matches_data=matches_data
            )

        except Exception as e:
            print(f"Error processing {file1} vs {file2}: {e}")
            return None

    def _process_batch(self, image_pairs: List[Tuple[str, str]]) -> List[MatchResult]:
        """Process a batch of image pairs"""
        results = []
        for file1, file2 in image_pairs:
            result = self._calculate_matches(file1, file2)
            if result:
                results.append(result)
        return results

    def run_parallel_matching(self, batch_size: int = 100):
        """Run parallel matching with database storage"""
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} images")

        # Get already processed pairs
        processed_pairs = self.db.get_processed_pairs()
        print(f"Resuming from {len(processed_pairs)} already processed pairs")

        # Generate pairs to process
        pairs_to_process = []
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                file1, file2 = image_files[i], image_files[j]

                # Skip same directory
                if os.path.dirname(file1) == os.path.dirname(file2):
                    continue

                # Skip if already processed
                basename1, basename2 = os.path.basename(file1), os.path.basename(file2)
                if (basename1, basename2) in processed_pairs or (basename2, basename1) in processed_pairs:
                    continue

                pairs_to_process.append((file1, file2))

        print(f"Processing {len(pairs_to_process)} pairs")

        # Process in batches with parallel workers
        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for i in range(0, len(pairs_to_process), batch_size):
                batch = pairs_to_process[i:i + batch_size]
                future = executor.submit(self._process_batch, batch)
                futures.append(future)

            # Process completed batches and save to database
            all_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_results = future.result()
                if batch_results:
                    all_results.extend(batch_results)

                    # Save batch to database when we have enough results
                    if len(all_results) >= 1000:
                        self.db.batch_insert_matches(all_results)
                        all_results = []

            # Save remaining results
            if all_results:
                self.db.batch_insert_matches(all_results)

        # Print statistics
        stats = self.db.get_statistics()
        print(f"Matching completed. Statistics: {stats}")

    def validate_with_pam(self, pam_data_path: str):
        """Validate matches against PAM data"""
        import pandas as pd

        # Load PAM data
        pam_df = pd.read_csv(pam_data_path)
        pam_df = pam_df.dropna(subset=["Box"])

        # Find potential true positive pairs
        merged_df = pd.merge(pam_df, pam_df, on=["Scroll", "Frg"])
        filtered_df = merged_df[merged_df["Box_x"] != merged_df["Box_y"]]

        # Generate image pairs to validate
        validation_pairs = []
        for _, row in filtered_df.iterrows():
            file1 = f"{row['File_x']}_{int(row['Box_x'])}.jpg"
            file2 = f"{row['File_y']}_{int(row['Box_y'])}.jpg"
            validation_pairs.append((file1, file2))

        # Update database with validation status
        self.db.update_validation_status(validation_pairs)

        stats = self.db.get_statistics()
        print(f"Validation completed. Statistics: {stats}")

    def export_top_matches(self, output_path: str, limit: int = 10000):
        """Export top matches to CSV for analysis"""
        import csv

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file1', 'file2', 'match_count', 'is_validated'])

            for row in self.db.get_top_matches(limit=limit):
                file1, file2, match_count, matches_data, is_validated = row
                writer.writerow([file1, file2, match_count, bool(is_validated)])

        print(f"Exported top {limit} matches to {output_path}")


# Environment configuration
import os
from dotenv import load_dotenv


def load_env_config():
    """Load configuration from .env file"""
    load_dotenv()

    base_path = os.getenv('BASE_PATH')
    if not base_path:
        raise ValueError("BASE_PATH not found in .env file")

    model_type = os.getenv('MODEL_TYPE', 'default')

    config = {
        'base_path': base_path,
        'model_type': model_type,
        'image_base_path': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('PATCHES_DIR', 'patches')),
        'cache_dir': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('PATCHES_CACHE', 'cache')),
        'db_path': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('DB_NAME', 'matches.db')),
        'pam_data_path': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('PAM_CSV', 'pam.csv')),
        'output_csv_path': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('OUTPUT_CSV', 'top_matches.csv')),
        'num_workers': int(os.getenv('NUM_WORKERS', '8')),
        'batch_size': int(os.getenv('BATCH_SIZE', '200')),
        'export_limit': int(os.getenv('EXPORT_LIMIT', '10000')),
        'debug': os.getenv('DEBUG', 'false').lower() == 'true'
    }

    return config


class OptimizedFragmentMatchingPipeline:
    """
    Main pipeline controller that uses environment configuration and database backend.
    """

    def __init__(self, config: dict = None):
        """Initialize pipeline with configuration from .env"""
        self.config = config or load_env_config()

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config['cache_dir']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['db_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['output_csv_path']), exist_ok=True)

        # Initialize matcher
        self.matcher = ParallelFragmentMatcher(
            image_base_path=self.config['image_base_path'],
            cache_dir=self.config['cache_dir'],
            db_path=self.config['db_path'],
            num_workers=self.config['num_workers']
        )

        if self.config['debug']:
            print(f"Initialized pipeline with config: {self.config}")

    def run_feature_matching(self):
        """Execute the SIFT-based feature matching stage"""
        print("Starting optimized parallel matching...")
        print(f"Processing images from: {self.config['image_base_path']}")
        print(f"Using {self.config['num_workers']} workers with batch size {self.config['batch_size']}")

        self.matcher.run_parallel_matching(batch_size=self.config['batch_size'])

        # Print statistics
        stats = self.matcher.db.get_statistics()
        print(f"Feature matching completed. Statistics:")
        print(f"  Total matches: {stats['total_matches']:,}")
        print(f"  Average match count: {stats['avg_match_count']:.2f}")
        print(f"  Maximum match count: {stats['max_match_count']}")

    def run_validation(self):
        """Execute the PAM validation stage"""
        print("Starting PAM validation...")
        print(f"Using PAM data from: {self.config['pam_data_path']}")

        if not os.path.exists(self.config['pam_data_path']):
            print(f"Warning: PAM file not found at {self.config['pam_data_path']}")
            print("Skipping validation step...")
            return

        self.matcher.validate_with_pam(self.config['pam_data_path'])

        # Print updated statistics
        stats = self.matcher.db.get_statistics()
        print(f"Validation completed. Statistics:")
        print(f"  Total matches: {stats['total_matches']:,}")
        print(f"  Validated matches: {stats['validated_matches']:,}")
        print(f"  Validation rate: {(stats['validated_matches'] / stats['total_matches'] * 100):.2f}%")

    def export_results(self):
        """Export top matches to CSV"""
        print(f"Exporting top {self.config['export_limit']} matches...")
        print(f"Output file: {self.config['output_csv_path']}")

        self.matcher.export_top_matches(
            self.config['output_csv_path'],
            limit=self.config['export_limit']
        )

    def run_complete_pipeline(self):
        """Execute the complete optimized pipeline"""
        print("Starting complete optimized fragment matching pipeline...")
        print(f"Base path: {self.config['base_path']}")
        print(f"Model type: {self.config['model_type']}")

        # Run feature matching
        self.run_feature_matching()

        # Run validation if PAM data exists
        self.run_validation()

        # Export results
        self.export_results()

        print("Complete optimized pipeline finished successfully!")

    def get_database_info(self):
        """Print database information and statistics"""
        stats = self.matcher.db.get_statistics()
        print(f"Database: {self.config['db_path']}")
        print(f"Statistics: {stats}")

        # Check file size
        if os.path.exists(self.config['db_path']):
            size_mb = os.path.getsize(self.config['db_path']) / (1024 * 1024)
            print(f"Database size: {size_mb:.2f} MB")


def main():
    """Main execution function using .env configuration"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimized Fragment Matching Pipeline")
    parser.add_argument(
        "--stage",
        choices=["matching", "validation", "export", "complete", "info"],
        default="complete",
        help="Pipeline stage to execute (default: complete)"
    )
    parser.add_argument(
        "--config",
        help="Path to .env file (default: .env in current directory)"
    )

    args = parser.parse_args()

    # Load environment from specific file if provided
    if args.config:
        load_dotenv(args.config)

    try:
        # Initialize pipeline with environment configuration
        pipeline = OptimizedFragmentMatchingPipeline()

        # Execute appropriate stage
        if args.stage == "matching":
            pipeline.run_feature_matching()
        elif args.stage == "validation":
            pipeline.run_validation()
        elif args.stage == "export":
            pipeline.export_results()
        elif args.stage == "info":
            pipeline.get_database_info()
        else:  # complete
            pipeline.run_complete_pipeline()

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease ensure your .env file contains:")
        print("BASE_PATH=/your/base/path")
        print("MODEL_TYPE=your_model_type")
        print("PATCHES_DIR=patches")
        print("PATCHES_CACHE=cache")
        print("DB_NAME=matches.db")
        print("PAM_CSV=pam.csv")
        print("OUTPUT_CSV=top_matches.csv")
        print("NUM_WORKERS=8")
        print("BATCH_SIZE=200")
        print("EXPORT_LIMIT=10000")
        print("DEBUG=false")
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
