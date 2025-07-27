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
# """
# Fragment Matching Pipeline
#
# This module provides a comprehensive pipeline for matching image fragments/patches using
# SIFT (Scale-Invariant Feature Transform) feature detection and validation against ground
# truth data. It combines feature-based matching with metadata validation to identify
# genuine fragment matches in archaeological or document analysis workflows.
#
# The pipeline implements two main stages:
# 1. Feature Matching: Extracts SIFT features and performs brute-force matching between patches
# 2. Validation: Cross-references results with PAM (Patch Annotation Metadata) to identify
#    true positive matches based on scroll/fragment relationships
#
# Key Components:
# - DescriptorCacheManager: Handles caching of SIFT features to disk for efficiency
# - NaiveImageMatcher: Performs SIFT-based feature matching between image pairs
# - FragmentMatcher: Orchestrates the complete feature matching pipeline
# - PamProcessor: Validates matches against ground truth metadata and generates final results
#
# The matching process uses SIFT features with brute-force matching and Lowe's ratio test
# to filter high-quality matches. Results are cross-validated against metadata to identify
# matches between different boxes of the same scroll/fragment combination.
#
# Dependencies:
#     - cv2 (OpenCV): Computer vision operations and SIFT feature extraction
#     - numpy: Numerical array operations
#     - pandas: Data manipulation for CSV processing
#     - pickle: Serialization for caching feature data
#     - csv: Reading/writing match results
#     - tqdm: Progress tracking for long-running operations
#
# Usage:
#     # Stage 1: Feature matching
#     python fragment_matching_pipeline.py --stage=matching
#
#     # Stage 2: Validation against PAM data
#     python fragment_matching_pipeline.py --stage=validation
#
#     # Run complete pipeline
#     python fragment_matching_pipeline.py --stage=complete
#
# Performance Notes:
#     - SIFT features are cached to disk to avoid recomputation
#     - Only patches from different directories are compared in initial matching
#     - CSV processing uses chunking for large datasets to manage memory usage
#     - Progress can be resumed from checkpoints in both stages
# """
#
# import csv
# import itertools
# import os
# import pickle
# import sys
# from typing import Dict, List, Tuple
#
# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# from env_arguments_loader import load_env_arguments
#
# # Increase the CSV field size limit to handle large match data
# csv.field_size_limit(sys.maxsize)
#
#
# class DescriptorCacheManager:
#     """
#     Manages caching of SIFT descriptors and keypoints to disk for efficient reuse.
#
#     This class handles the storage and retrieval of computed SIFT features, eliminating
#     the need to recompute expensive feature extraction for the same images. Features
#     are serialized using pickle and stored with the image filename as the key.
#
#     The cache stores both keypoints (with their geometric properties) and descriptors
#     (feature vectors) in a structured format that can be efficiently loaded.
#
#     Attributes:
#         cache_dir (str): Directory path where cache files are stored
#
#     Cache File Format:
#         Each image gets a .pkl file containing:
#         {
#             "keypoints": [(pt, size, angle, response, octave, class_id), ...],
#             "descriptors": numpy.ndarray of shape (n_keypoints, 128)
#         }
#     """
#
#     def __init__(self, cache_dir):
#         """
#         Initialize the cache manager with a specified cache directory.
#
#         Args:
#             cache_dir (str): Path to the directory where cache files will be stored.
#                            Directory will be created if it doesn't exist.
#
#         Example:
#             >>> cache_mgr = DescriptorCacheManager("/tmp/sift_cache")
#         """
#         self.cache_dir = cache_dir
#         if not os.path.exists(self.cache_dir):
#             os.makedirs(self.cache_dir)
#
#     def _get_cache_file_path(self, image_key: str) -> str:
#         """
#         Generate the file path for a specific image's cache file.
#
#         Args:
#             image_key (str): Base filename of the image (used as cache key)
#
#         Returns:
#             str: Full path to the cache file for this image
#
#         Example:
#             >>> cache_mgr._get_cache_file_path("patch_123.jpg")
#             "/tmp/sift_cache/patch_123.jpg.pkl"
#         """
#         return os.path.join(self.cache_dir, f"{image_key}.pkl")
#
#     def _is_cached(self, image_key: str) -> bool:
#         """
#         Check if cached data exists for a specific image.
#
#         Args:
#             image_key (str): Base filename of the image
#
#         Returns:
#             bool: True if cache file exists, False otherwise
#
#         Example:
#             >>> cache_mgr._is_cached("patch_123.jpg")
#             True
#         """
#         return os.path.exists(self._get_cache_file_path(image_key))
#
#     def _load_cache(self, image_key: str) -> Dict:
#         """
#         Load cached SIFT data for an image.
#
#         Args:
#             image_key (str): Base filename of the image
#
#         Returns:
#             Dict or None: Dictionary containing keypoints and descriptors,
#                          or None if cache file doesn't exist or is corrupted
#
#         Dictionary Structure:
#             {
#                 "keypoints": List of serialized keypoint tuples,
#                 "descriptors": numpy.ndarray of SIFT descriptors
#             }
#
#         Example:
#             >>> data = cache_mgr._load_cache("patch_123.jpg")
#             >>> data["descriptors"].shape
#             (156, 128)  # 156 keypoints, 128-dim descriptors
#         """
#         cache_file = self._get_cache_file_path(image_key)
#         if os.path.exists(cache_file):
#             with open(cache_file, "rb") as f:
#                 return pickle.load(f)
#         return None
#
#     def _save_cache(self, image_key: str, data: Dict):
#         """
#         Save computed SIFT data to a cache file.
#
#         Args:
#             image_key (str): Base filename of the image
#             data (Dict): Dictionary containing keypoints and descriptors to cache
#
#         Side Effects:
#             Creates or overwrites the cache file for this image
#
#         Example:
#             >>> data = {"keypoints": serialized_kp, "descriptors": desc_array}
#             >>> cache_mgr._save_cache("patch_123.jpg", data)
#         """
#         cache_file = self._get_cache_file_path(image_key)
#         with open(cache_file, "wb") as f:
#             pickle.dump(data, f)
#
#     def _serialize_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[Tuple]:
#         """
#         Convert OpenCV KeyPoint objects to serializable tuples.
#
#         OpenCV KeyPoint objects cannot be directly pickled, so this method
#         extracts their essential properties into tuples that can be saved.
#
#         Args:
#             keypoints (List[cv2.KeyPoint]): List of OpenCV keypoint objects
#
#         Returns:
#             List[Tuple]: List of tuples containing keypoint properties:
#                         (point_coords, size, angle, response, octave, class_id)
#
#         Example:
#             >>> kp_list = [cv2.KeyPoint(x=10, y=20, size=5, ...)]
#             >>> serialized = cache_mgr._serialize_keypoints(kp_list)
#             >>> serialized[0]
#             ((10.0, 20.0), 5.0, -1.0, 0.1, 0, -1)
#         """
#         return [
#             (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
#             for kp in keypoints
#         ]
#
#     def _deserialize_keypoints(self, keypoints_data: List[Tuple]) -> List[cv2.KeyPoint]:
#         """
#         Convert serialized keypoint tuples back to OpenCV KeyPoint objects.
#
#         Args:
#             keypoints_data (List[Tuple]): List of serialized keypoint tuples
#
#         Returns:
#             List[cv2.KeyPoint]: List of reconstructed OpenCV KeyPoint objects
#
#         Example:
#             >>> tuples = [((10.0, 20.0), 5.0, -1.0, 0.1, 0, -1)]
#             >>> keypoints = cache_mgr._deserialize_keypoints(tuples)
#             >>> keypoints[0].pt
#             (10.0, 20.0)
#         """
#         return [
#             cv2.KeyPoint(
#                 x=pt[0][0],
#                 y=pt[0][1],
#                 size=pt[1],
#                 angle=pt[2],
#                 response=pt[3],
#                 octave=pt[4],
#                 class_id=pt[5],
#             )
#             for pt in keypoints_data
#         ]
#
#     def process_image(self, file_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
#         """
#         Process an image to extract SIFT features, using cache when available.
#
#         This method first checks if SIFT features for the image are already cached.
#         If cached data exists, it loads and returns it. Otherwise, it computes
#         SIFT features from scratch and caches the results for future use.
#
#         Args:
#             file_path (str): Full path to the image file to process
#
#         Returns:
#             Tuple[List[cv2.KeyPoint], np.ndarray]: Tuple containing:
#                 - List of detected keypoints
#                 - Array of SIFT descriptors (shape: n_keypoints x 128)
#
#         Raises:
#             ValueError: If the image file cannot be loaded
#
#         Example:
#             >>> keypoints, descriptors = cache_mgr.process_image("patch.jpg")
#             >>> len(keypoints)
#             156
#             >>> descriptors.shape
#             (156, 128)
#         """
#         image_key = os.path.basename(file_path)
#
#         if self._is_cached(image_key):
#             # Load cached data if available
#             cached_data = self._load_cache(image_key)
#             if cached_data:
#                 keypoints = self._deserialize_keypoints(
#                     cached_data["keypoints"]
#                 )
#                 descriptors = cached_data["descriptors"]
#                 return keypoints, descriptors
#
#         # If not cached, compute SIFT features and cache the result
#         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise ValueError(f"Could not load image: {file_path}")
#
#         sift = cv2.SIFT_create()
#         keypoints, descriptors = sift.detectAndCompute(img, None)
#
#         # Save the computed data to the cache
#         self._save_cache(
#             image_key,
#             {
#                 "keypoints": self._serialize_keypoints(keypoints),
#                 "descriptors": descriptors,
#             },
#         )
#
#         return keypoints, descriptors
#
#
# class NaiveImageMatcher:
#     """
#     Performs feature matching between image pairs using SIFT features and brute-force matching.
#
#     This class implements a straightforward approach to image matching by:
#     1. Extracting SIFT descriptors from both images
#     2. Using brute-force matcher to find nearest neighbors
#     3. Applying Lowe's ratio test to filter high-quality matches
#
#     The matcher uses a ratio threshold of 0.75, which is a standard value
#     that provides a good balance between match precision and recall.
#
#     Attributes:
#         descriptor_cache (DescriptorCacheManager): Cache manager for SIFT features
#     """
#
#     def __init__(self, descriptor_cache: DescriptorCacheManager):
#         """
#         Initialize the image matcher with a descriptor cache.
#
#         Args:
#             descriptor_cache (DescriptorCacheManager): Cache manager for storing/loading
#                                                      SIFT features
#
#         Example:
#             >>> cache = DescriptorCacheManager("/tmp/cache")
#             >>> matcher = NaiveImageMatcher(cache)
#         """
#         self.descriptor_cache = descriptor_cache
#
#     def calc_matches(self, file1: str, file2: str) -> List[cv2.DMatch]:
#         """
#         Calculate SIFT feature matches between two images using Lowe's ratio test.
#
#         This method performs the following steps:
#         1. Extract SIFT descriptors from both images (via cache)
#         2. Use brute-force matcher to find 2 nearest neighbors for each descriptor
#         3. Apply Lowe's ratio test: accept match if distance_1 < 0.75 * distance_2
#         4. Return list of good matches
#
#         The ratio test helps filter out ambiguous matches where the closest and
#         second-closest matches are very similar, indicating low discriminability.
#
#         Args:
#             file1 (str): Path to the first image file
#             file2 (str): Path to the second image file
#
#         Returns:
#             List[cv2.DMatch]: List of good matches that passed the ratio test.
#                              Each DMatch contains queryIdx, trainIdx, and distance.
#
#         Note:
#             Returns empty list if matching fails due to insufficient features
#             or other errors.
#
#         Example:
#             >>> matches = matcher.calc_matches("patch1.jpg", "patch2.jpg")
#             >>> len(matches)
#             23
#             >>> matches[0].distance
#             45.7  # Euclidean distance between descriptors
#         """
#         # Get descriptors and keypoints for both images
#         kp1, des1 = self.descriptor_cache.process_image(file1)
#         kp2, des2 = self.descriptor_cache.process_image(file2)
#
#         try:
#             good_matches = []
#             bf = cv2.BFMatcher()
#             # BFMatcher stands for Brute-Force Matcher. It compares each descriptor
#             # from des1 with all the descriptors from des2.
#             matches = bf.knnMatch(des1, des2, k=2)
#
#             # Apply ratio test to filter out good matches
#             for m, n in matches:
#                 if m.distance < 0.75 * n.distance:
#                     good_matches.append(m)
#
#         except Exception as e:
#             print(f"Error occurred while filtering matches: {e}")
#             return []
#
#         return good_matches
#
#
# class FragmentMatcher:
#     """
#     Orchestrates the complete fragment matching pipeline for image patch datasets.
#
#     This class manages the entire process of matching image fragments:
#     1. Discovers all image files in the dataset directory
#     2. Generates all possible image pairs (excluding same-directory pairs)
#     3. Performs SIFT-based matching for each pair
#     4. Saves results to CSV with resume capability
#     5. Tracks progress and handles large datasets efficiently
#
#     The matcher assumes that patches from the same original image are stored
#     in the same subdirectory, and only compares patches across different
#     subdirectories to find potential matches between different source images.
#
#     Attributes:
#         image_base_path (str): Root directory containing image subdirectories
#         matcher (NaiveImageMatcher): Matcher instance for performing comparisons
#     """
#
#     def __init__(self, image_base_path: str, cache_dir: str):
#         """
#         Initialize the fragment matcher with dataset and cache paths.
#
#         Args:
#             image_base_path (str): Root directory containing image files/subdirectories
#             cache_dir (str): Directory for caching SIFT features
#
#         Example:
#             >>> fm = FragmentMatcher("/data/patches", "/tmp/sift_cache")
#         """
#         self.image_base_path = image_base_path
#         self.matcher = NaiveImageMatcher(DescriptorCacheManager(cache_dir))
#
#     def get_image_files(self) -> List[str]:
#         """
#         Recursively discover all JPEG image files in the base directory.
#
#         Walks through all subdirectories of the base path and collects
#         full paths to all .jpg files found.
#
#         Returns:
#             List[str]: List of full paths to all discovered image files
#
#         Example:
#             >>> fm.get_image_files()
#             ['/data/patches/img1/patch_1.jpg', '/data/patches/img1/patch_2.jpg',
#              '/data/patches/img2/patch_1.jpg', ...]
#         """
#         image_files = []
#         for root, _, files in os.walk(self.image_base_path):
#             for file in files:
#                 if file.endswith(".jpg"):  # Assuming patches are in .jpg format
#                     image_files.append(os.path.join(root, file))
#         return image_files
#
#     def _get_processed_pairs(self, success_csv: str) -> set:
#         """
#         Read existing CSV results to determine which image pairs have been processed.
#
#         This enables resume capability by tracking which comparisons have already
#         been completed and stored in the results file.
#
#         Args:
#             success_csv (str): Path to the CSV file containing previous results
#
#         Returns:
#             set: Set of tuples (file1, file2) representing processed pairs
#
#         Example:
#             >>> processed = fm._get_processed_pairs("results.csv")
#             >>> ("patch1.jpg", "patch2.jpg") in processed
#             True
#         """
#         processed_pairs = set()
#         if os.path.exists(success_csv):
#             with open(success_csv, mode="r") as file:
#                 reader = csv.DictReader(file)
#                 for row in reader:
#                     processed_pairs.add((row["file1"], row["file2"]))
#         return processed_pairs
#
#     def calculate_distances(
#             self, image_files: List[str], success_csv: str, debug: bool = False
#     ) -> None:
#         """
#         Calculate and save matching distances for all valid image pairs.
#
#         This method performs the core matching computation:
#         1. Generates all possible pairs from the image list
#         2. Filters out pairs from the same directory
#         3. Skips already processed pairs (resume capability)
#         4. Computes SIFT matches for each remaining pair
#         5. Saves results with match count and detailed match data
#
#         Only pairs with at least one good match are saved to the CSV file.
#         Progress is tracked with a progress bar for long-running operations.
#
#         Args:
#             image_files (List[str]): List of all image file paths to compare
#             success_csv (str): Path to output CSV file for results
#             debug (bool): Enable debug output (currently unused)
#
#         Side Effects:
#             - Creates or appends to the CSV results file
#             - Updates progress bar during processing
#             - Flushes results to disk after each successful match
#
#         CSV Output Format:
#             file1,file2,distance,matches
#             patch1.jpg,patch2.jpg,23,"[(0,5,45.7), (1,12,38.2), ...]"
#
#         Example:
#             >>> fm.calculate_distances(image_list, "results.csv")
#             Processing Patches: 100%|██████████| 1000/1000 [05:23<00:00, 3.09it/s]
#         """
#         total_iterations = sum(
#             range(1, len(image_files))
#         )  # Total number of comparisons
#         processed_pairs = self._get_processed_pairs(success_csv)
#
#         with open(success_csv, mode="a", newline="") as file:
#             fieldnames = ["file1", "file2", "distance", "matches"]
#             writer = csv.DictWriter(file, fieldnames=fieldnames)
#
#             # Write header only if the file is newly created
#             if not processed_pairs:
#                 writer.writeheader()
#
#             with tqdm(
#                     total=total_iterations,
#                     desc="Processing Patches",
#                     disable=False,
#             ) as pbar:
#                 for i, j in itertools.combinations(range(len(image_files)), 2):
#                     image_path1 = image_files[i]
#                     image_path2 = image_files[j]
#
#                     dirname1 = os.path.dirname(image_path1)
#                     dirname2 = os.path.dirname(image_path2)
#
#                     base_name1 = os.path.basename(image_path1)
#                     base_name2 = os.path.basename(image_path2)
#
#                     # Different patches can't be from the same directory
#                     if dirname1 == dirname2:
#                         pbar.update(1)  # Update progress bar
#                         continue
#
#                     # Check if the pair has already been processed
#                     if (image_path1, image_path2) in processed_pairs or (
#                             image_path2,
#                             image_path1,
#                     ) in processed_pairs:
#                         pbar.update(1)  # Update progress bar
#                         continue  # Skip this pair
#
#                     # Calculate matches for this pair
#                     pbar.update(1)
#                     good_matches = self.matcher.calc_matches(
#                         image_path1, image_path2
#                     )
#
#                     if len(good_matches) <= 0:
#                         continue
#
#                     # Write match details to the CSV
#                     writer.writerow(
#                         {
#                             "file1": base_name1,
#                             "file2": base_name2,
#                             "distance": len(good_matches),
#                             "matches": [
#                                 (m.queryIdx, m.trainIdx, m.distance)
#                                 for m in good_matches
#                             ],
#                         }
#                     )
#
#                     # Flush to ensure data is written to the file immediately
#                     file.flush()
#
#     def run(self, success_csv, debug=False):
#         """
#         Execute the complete fragment matching pipeline.
#
#         This is the main entry point that orchestrates the entire matching process:
#         1. Discovers all image files in the dataset
#         2. Performs pairwise matching with progress tracking
#         3. Saves results to the specified CSV file
#
#         Args:
#             success_csv (str): Path to output CSV file for storing match results
#             debug (bool): Enable debug mode (passed to calculate_distances)
#
#         Side Effects:
#             - Prints completion message with output file path
#             - Creates the output CSV file with match results
#
#         Example:
#             >>> fm = FragmentMatcher("/data/patches", "/tmp/cache")
#             >>> fm.run("fragment_matches.csv")
#             Results written to fragment_matches.csv
#         """
#         image_files = self.get_image_files()
#         self.calculate_distances(image_files, success_csv, debug=debug)
#         print(f"Results written to {success_csv}")
#
#
# class PamProcessor:
#     """
#     Processes and validates SIFT matches against PAM (Patch Annotation Metadata) ground truth data.
#
#     This class takes the raw SIFT matching results and cross-references them with metadata
#     about scroll/fragment relationships to identify true positive matches. It focuses on
#     finding matches between different boxes of the same scroll/fragment combination, which
#     represent genuine fragment matches.
#
#     The validation process involves:
#     1. Loading PAM metadata containing scroll, fragment, and box information
#     2. Identifying potential true positive pairs (same scroll/fragment, different boxes)
#     3. Cross-referencing with SIFT match results to mark validated matches
#     4. Generating a final sorted output with match confidence scores
#
#     Attributes:
#         pam_csv_file (str): Path to the PAM metadata CSV file
#         image_base_path (str): Root directory containing patch images
#         sift_matches_file (str): Path to the SIFT matching results CSV
#     """
#
#     def __init__(self, image_base_path: str, pam_csv_file: str, sift_matches_file: str):
#         """
#         Initialize the PAM processor with required file paths.
#
#         Args:
#             image_base_path (str): Root directory containing patch image files
#             pam_csv_file (str): Path to PAM metadata CSV file
#             sift_matches_file (str): Path to SIFT matching results CSV file
#
#         Raises:
#             AssertionError: If any of the required files/directories don't exist
#
#         Example:
#             >>> processor = PamProcessor("/data/patches", "/data/pam.csv", "/data/matches.csv")
#         """
#         self.pam_csv_file = pam_csv_file
#         assert os.path.exists(self.pam_csv_file), (
#             f"PAM CSV file not found: {self.pam_csv_file}"
#         )
#
#         self.image_base_path = image_base_path
#         assert os.path.exists(self.image_base_path), (
#             f"Image base path not found: {self.image_base_path}"
#         )
#
#         self.sift_matches_file = sift_matches_file
#         assert os.path.exists(self.sift_matches_file), (
#             f"SIFT matches file not found: {self.sift_matches_file}"
#         )
#
#     def read_pam_csv(self) -> pd.DataFrame:
#         """
#         Read the PAM metadata CSV file into a pandas DataFrame.
#
#         Returns:
#             pd.DataFrame: DataFrame containing PAM metadata with columns for
#                          Scroll, Fragment (Frg), File, Box, and other metadata
#
#         Example:
#             >>> df = processor.read_pam_csv()
#             >>> df.columns
#             Index(['Scroll', 'Frg', 'File', 'Box', ...])
#         """
#         return pd.read_csv(self.pam_csv_file)
#
#     def read_sift_csv_chunks(self, chunk_size: int = 100000) -> pd.io.parsers.readers.TextFileReader:
#         """
#         Read the SIFT matches CSV file in chunks to manage memory usage.
#
#         Args:
#             chunk_size (int): Number of rows to read per chunk (default: 100000)
#
#         Returns:
#             pd.io.parsers.readers.TextFileReader: Chunked CSV reader for processing
#                                                  large SIFT match files
#
#         Example:
#             >>> for chunk in processor.read_sift_csv_chunks():
#             ...     # Process each chunk of SIFT matches
#             ...     process_chunk(chunk)
#         """
#         return pd.read_csv(self.sift_matches_file, chunksize=chunk_size)
#
#     def get_matched_pam_files(self) -> pd.DataFrame:
#         """
#         Identify potential true positive matches from PAM metadata.
#
#         This method finds pairs of patches that come from the same scroll and fragment
#         but different boxes, which represent genuine fragment matches that should be
#         validated against SIFT results.
#
#         The process:
#         1. Remove rows with NaN Box values
#         2. Self-join PAM data on Scroll and Fragment columns
#         3. Filter for pairs with different Box values
#         4. Return potential match pairs for validation
#
#         Returns:
#             pd.DataFrame: DataFrame with columns [Scroll, Frg, File_x, Box_x, File_y, Box_y]
#                          representing potential true positive match pairs
#
#         Example:
#             >>> matches = processor.get_matched_pam_files()
#             >>> matches.head()
#                Scroll  Frg    File_x  Box_x    File_y  Box_y
#             0      1    2  scroll1_2      1  scroll1_2      3
#             1      1    2  scroll1_2      1  scroll1_2      5
#         """
#         df = self.read_pam_csv()
#
#         # Remove rows where Box is NaN
#         df = df.dropna(subset=["Box"])
#
#         # Perform an inner join on the Scroll and Frg columns
#         merged_df = pd.merge(df, df, on=["Scroll", "Frg"])
#
#         # Filter the results where the Box values are different
#         filtered_df = merged_df[merged_df["Box_x"] != merged_df["Box_y"]]
#
#         # Select only the necessary columns: file1, box1, file2, box2
#         result_df = filtered_df[
#             ["Scroll", "Frg", "File_x", "Box_x", "File_y", "Box_y"]
#         ]
#         return result_df
#
#     def get_image_path(self, file_name: str, box: int) -> str:
#         """
#         Generate the standardized image filename based on file and box information.
#
#         Args:
#             file_name (str): Base filename from PAM metadata
#             box (int): Box number for the specific patch
#
#         Returns:
#             str: Standardized image filename in format "{file_name}_{box}.jpg"
#
#         Example:
#             >>> processor.get_image_path("scroll1_frg2", 3)
#             "scroll1_frg2_3.jpg"
#         """
#         return f"{file_name}_{int(box)}.jpg"
#
#     def process_and_save_matches(self, output_filename: str, top_n: int = -1) -> None:
#         """
#         Process SIFT matches against PAM metadata and save validated results.
#
#         This method performs the core validation process:
#         1. Identifies potential true positive pairs from PAM metadata
#         2. Processes SIFT matches in chunks to manage memory
#         3. Marks matches that correspond to true positive pairs
#         4. Sorts results by match confidence (distance/score)
#         5. Saves final validated results to output file
#
#         The process uses chunked reading for the SIFT matches to handle large datasets
#         efficiently, and includes a temporary file mechanism to manage intermediate results.
#
#         Args:
#             output_filename (str): Path for the final validated results CSV file
#             top_n (int): Number of top matches to include in final output.
#                         If -1, includes all matches (default: -1)
#
#         Side Effects:
#             - Creates a temporary file during processing
#             - Writes the final sorted results to output_filename
#             - Prints progress updates and completion message
#
#         Output CSV Format:
#             The output file contains all original SIFT match columns plus a "Match" column
#             indicating validation status (1 for validated true positives, 0 otherwise).
#
#         Example:
#             >>> processor.process_and_save_matches("validated_matches.csv", top_n=1000)
#             Processing Matches: 100%|██████████| 500/500 [02:15<00:00, 3.70it/s]
#             Match found: scroll1_2_1.jpg - scroll1_2_3.jpg
#             Processing final sorting...
#             Processed and sorted results saved to: validated_matches.csv
#         """
#         matches_df = self.get_matched_pam_files()
#
#         # Create a temporary file for the intermediate results
#         temp_output = output_filename + ".temp"
#
#         # Process and save matches with validation
#         with open(temp_output, "w") as f:
#             header_written = False
#             for chunk in self.read_sift_csv_chunks():
#                 # Add Match column to track validation status
#                 chunk["Match"] = 0
#
#                 # Check each potential true positive pair against current chunk
#                 for index, row in tqdm(
#                         matches_df.iterrows(),
#                         total=matches_df.shape[0],
#                         desc="Processing Matches",
#                 ):
#                     image_path1 = self.get_image_path(
#                         row["File_x"], row["Box_x"]
#                     )
#                     image_path2 = self.get_image_path(
#                         row["File_y"], row["Box_y"]
#                     )
#
#                     # Check if both image paths are present in the current chunk
#                     # (in either order since matching is bidirectional)
#                     matching_rows = chunk[
#                         (
#                                 (chunk["file1"] == image_path1)
#                                 & (chunk["file2"] == image_path2)
#                         )
#                         | (
#                                 (chunk["file2"] == image_path1)
#                                 & (chunk["file1"] == image_path2)
#                         )
#                         ].index
#
#                     if len(matching_rows) > 0:
#                         chunk.loc[matching_rows, "Match"] = 1
#                         print(f"Match found: {image_path1} - {image_path2}")
#
#                 # Save the chunk to the temporary output file
#                 if not header_written:
#                     chunk.to_csv(f, index=False)
#                     header_written = True
#                 else:
#                     chunk.to_csv(f, index=False, header=False, mode="a")
#
#         # Read temporary file, sort results, and save final output
#         print("Processing final sorting...")
#         df = pd.read_csv(temp_output)
#
#         # Sort by the third column (distance/score) in descending order
#         df_sorted = df.sort_values(by=df.columns[2], ascending=False)
#
#         # Select the top N rows if specified
#         if top_n > 0:
#             df_sorted_top = df_sorted.head(top_n)
#         else:
#             df_sorted_top = df_sorted
#
#         # Save the final sorted results
#         df_sorted_top.to_csv(output_filename, index=False)
#
#         # Clean up the temporary file
#         os.remove(temp_output)
#
#         print(f"Processed and sorted results saved to: {output_filename}")
#
#
# class FragmentMatchingPipeline:
#     """
#     Main pipeline controller that orchestrates both feature matching and validation stages.
#
#     This class provides a unified interface for running the complete fragment matching
#     workflow, from initial SIFT-based feature matching through final validation against
#     ground truth metadata.
#
#     The pipeline can be run in different modes:
#     - Feature matching only: Generates initial SIFT matches
#     - Validation only: Processes existing SIFT matches against PAM data
#     - Complete pipeline: Runs both stages sequentially
#
#     Attributes:
#         args: Configuration arguments loaded from environment
#         patches_dir (str): Directory containing patch images
#         patch_cache_dir (str): Directory for SIFT feature cache
#         pam_csv_path (str): Path to PAM metadata file
#         sift_matches_path (str): Path to SIFT matches output file
#         validated_matches_path (str): Path to final validated matches file
#     """
#
#     def __init__(self, args):
#         """
#         Initialize the pipeline with configuration arguments.
#
#         Args:
#             args: Configuration object containing paths and settings
#
#         Example:
#             >>> args = load_env_arguments()
#             >>> pipeline = FragmentMatchingPipeline(args)
#         """
#         self.args = args
#
#         # Set up all required paths
#         self.patches_dir = os.path.join(args.base_path, "OUTPUT_" + args.model_type, args.patches_dir)
#         self.patch_cache_dir = os.path.join(args.base_path, "OUTPUT_" + args.model_type,  args.patches_cache)
#         self.pam_csv_path = os.path.join(args.base_path, "OUTPUT_" + args.model_type, args.csv_in)
#         self.sift_matches_path = os.path.join(args.base_path, "OUTPUT_" + args.model_type, args.sift_matches)
#         self.validated_matches_path = os.path.join(args.base_path,"OUTPUT_" + args.model_type,  args.sift_matches_w_tp)
#
#     def run_feature_matching(self) -> None:
#         """
#         Execute the SIFT-based feature matching stage of the pipeline.
#
#         This stage:
#         1. Discovers all patch images in the dataset
#         2. Performs pairwise SIFT feature matching
#         3. Saves raw match results to CSV
#         4. Uses caching for efficiency and supports resume capability
#
#         Side Effects:
#             - Creates SIFT feature cache files
#             - Generates raw match results CSV file
#             - Prints progress and completion status
#
#         Example:
#             >>> pipeline.run_feature_matching()
#             Processing Patches: 100%|██████████| 5000/5000 [15:23<00:00, 5.41it/s]
#             Results written to /data/sift_matches.csv
#         """
#         print("Starting SIFT-based feature matching stage...")
#
#
#         # Initialize and run the fragment matcher
#         matcher = FragmentMatcher(self.patches_dir, self.patch_cache_dir)
#         matcher.run(success_csv=self.sift_matches_path, debug=self.args.debug)
#
#         print("Feature matching stage completed.")
#
#     def run_validation(self) -> None:
#         """
#         Execute the PAM validation stage of the pipeline.
#
#         This stage:
#         1. Loads PAM metadata and SIFT match results
#         2. Identifies potential true positive pairs from metadata
#         3. Cross-validates SIFT matches against ground truth
#         4. Generates final sorted and validated results
#
#         Side Effects:
#             - Creates validated match results CSV file
#             - Prints validation progress and match discoveries
#             - Removes temporary processing files
#
#         Example:
#             >>> pipeline.run_validation()
#             Processing Matches: 100%|██████████| 1500/1500 [03:45<00:00, 6.67it/s]
#             Match found: scroll1_2_1.jpg - scroll1_2_3.jpg
#             Processing final sorting...
#             Processed and sorted results saved to: /data/validated_matches.csv
#         """
#         print("Starting PAM validation stage...")
#
#         # Initialize and run the PAM processor
#         processor = PamProcessor(
#             self.patches_dir,
#             self.pam_csv_path,
#             self.sift_matches_path
#         )
#         processor.process_and_save_matches(self.validated_matches_path, top_n=-1)
#
#         print("Validation stage completed.")
#
#     def run_complete_pipeline(self) -> None:
#         """
#         Execute the complete fragment matching pipeline.
#
#         Runs both feature matching and validation stages sequentially,
#         providing a complete end-to-end solution for fragment matching
#         with ground truth validation.
#
#         Side Effects:
#             - All effects from both run_feature_matching() and run_validation()
#             - Prints overall pipeline status messages
#
#         Example:
#             >>> pipeline.run_complete_pipeline()
#             Starting complete fragment matching pipeline...
#             Starting SIFT-based feature matching stage...
#             [... feature matching progress ...]
#             Feature matching stage completed.
#             Starting PAM validation stage...
#             [... validation progress ...]
#             Validation stage completed.
#             Complete pipeline finished successfully.
#         """
#         print("Starting complete fragment matching pipeline...")
#
#         # Run feature matching stage
#         self.run_feature_matching()
#
#         # Run validation stage
#         self.run_validation()
#
#         print("Complete pipeline finished successfully.")
#
#
# def main():
#     """
#     Main execution function with support for different pipeline stages.
#
#     Loads configuration from environment variables and runs the appropriate
#     pipeline stage based on command line arguments or default behavior.
#
#     Supported execution modes:
#     - Feature matching only: python fragment_matching_pipeline.py --stage=matching
#     - Validation only: python fragment_matching_pipeline.py --stage=validation
#     - Complete pipeline: python fragment_matching_pipeline.py --stage=complete
#     - Default (complete): python fragment_matching_pipeline.py
#
#     The function handles argument parsing, pipeline initialization, and
#     stage execution based on the specified mode.
#     """
#     import argparse
#
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Fragment Matching Pipeline")
#     parser.add_argument(
#         "--stage",
#         choices=["matching", "validation", "complete"],
#         default="complete",
#         help="Pipeline stage to execute (default: complete)"
#     )
#     cmd_args = parser.parse_args()
#
#     # Load environment configuration
#     env_args = load_env_arguments()
#
#     # Initialize pipeline
#     pipeline = FragmentMatchingPipeline(env_args)
#
#     # Execute appropriate stage
#     if cmd_args.stage == "matching":
#         pipeline.run_feature_matching()
#     elif cmd_args.stage == "validation":
#         pipeline.run_validation()
#     else:  # complete
#         pipeline.run_complete_pipeline()
#
#
# if __name__ == "__main__":
#     """
#     Main execution block for the fragment matching pipeline.
#
#     This block handles the primary execution flow when the script is run directly.
#     It supports various execution modes through command line arguments and provides
#     a complete workflow for archaeological or document fragment matching tasks.
#
#     The pipeline is designed for scenarios where:
#     - Image fragments need to be matched based on visual similarity
#     - Ground truth metadata is available for validation
#     - Large datasets require efficient processing with caching and resume capability
#     - Results need to be ranked and filtered for manual review
#
#     Configuration is loaded from environment variables, allowing for flexible
#     deployment across different systems and datasets without code modification.
#     """
#     main()