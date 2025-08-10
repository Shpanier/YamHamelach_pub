"""
Database-Integrated Homography Error Calculator

This module extends the original homography error computation to work directly with
the SQLite database from the fragment matching pipeline. It processes SIFT match data,
computes homography matrices and projection errors, and stores results in the database.

The integrated pipeline:
1. Reads SIFT match pairs from the matches database table
2. Computes homography matrices and projection errors
3. Caches error arrays to disk for reuse
4. Calculates statistical metrics (sum, mean, std, count)
5. Updates database with homography error statistics
6. Provides comprehensive analysis and export capabilities

Dependencies:
    - cv2 (OpenCV): Homography estimation
    - numpy: Numerical computations
    - sqlite3: Database operations
    - pickle: Serialization for caching
    - tqdm: Progress tracking
    - python-dotenv: Environment variable loading
"""

import csv
import math
import os
import pickle
import sqlite3
import sys
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
from contextlib import contextmanager
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


@dataclass
class HomographyResult:
    """Data class for storing homography computation results."""
    match_id: int
    sum_homo_err: float
    len_homo_err: int
    mean_homo_err: float
    std_homo_err: float
    max_homo_err: float
    min_homo_err: float
    median_homo_err: float
    is_valid: bool
    computation_time: float


class HomographyDatabaseManager:
    """
    Database manager extended with homography error storage and querying.

    This class extends the original DatabaseManager to include homography-specific
    tables and operations while maintaining compatibility with the existing schema.
    """

    def __init__(self, db_path: str):
        """Initialize database manager with homography tables."""
        self.db_path = db_path
        self.init_homography_tables()

    def init_homography_tables(self):
        """
        Create or update database schema for homography error storage.

        Creates a new table for homography errors linked to the matches table,
        and adds indexes for efficient querying.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Ensure WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")

            # Create homography errors table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS homography_errors (
                    match_id INTEGER PRIMARY KEY,
                    sum_homo_err REAL NOT NULL,
                    len_homo_err INTEGER NOT NULL,
                    mean_homo_err REAL NOT NULL,
                    std_homo_err REAL NOT NULL,
                    max_homo_err REAL,
                    min_homo_err REAL,
                    median_homo_err REAL,
                    is_valid INTEGER NOT NULL,
                    computation_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_homo_match_id ON homography_errors(match_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_homo_mean_err ON homography_errors(mean_homo_err)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_homo_valid ON homography_errors(is_valid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_homo_mean_valid ON homography_errors(mean_homo_err, is_valid)")

            # Create processed tracking table for homography computation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS homography_processed (
                    match_id INTEGER PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add homography quality columns to matches table if they don't exist
            # Check if columns exist first
            cursor = conn.execute("PRAGMA table_info(matches)")
            existing_columns = [row[1] for row in cursor.fetchall()]

            if 'has_homography' not in existing_columns:
                conn.execute("ALTER TABLE matches ADD COLUMN has_homography INTEGER DEFAULT 0")

            if 'homography_quality' not in existing_columns:
                conn.execute("ALTER TABLE matches ADD COLUMN homography_quality REAL")

            conn.commit()
            print(f"Homography tables initialized in database: {self.db_path}")

    def get_unprocessed_matches(self, limit: int = None) -> Iterator[Tuple]:
        """
        Get matches that haven't had homography errors computed yet.

        Args:
            limit: Maximum number of matches to return

        Yields:
            Tuple: (id, file1, file2, match_count, matches_data)
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT m.id, m.file1, m.file2, m.match_count, m.matches_data
                FROM matches m
                LEFT JOIN homography_processed hp ON m.id = hp.match_id
                WHERE hp.match_id IS NULL
                ORDER BY m.match_count DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            for row in cursor:
                yield row

    def batch_insert_homography_errors(self, results: List[HomographyResult], batch_size: int = 1000):
        """
        Insert homography error results in batches.

        Args:
            results: List of HomographyResult objects
            batch_size: Number of records per transaction
        """
        total_inserted = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for i in range(0, len(results), batch_size):
                    batch = results[i:i + batch_size]

                    # Insert homography errors
                    conn.executemany("""
                        INSERT OR REPLACE INTO homography_errors 
                        (match_id, sum_homo_err, len_homo_err, mean_homo_err, 
                         std_homo_err, max_homo_err, min_homo_err, median_homo_err, 
                         is_valid, computation_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        (r.match_id, r.sum_homo_err, r.len_homo_err, r.mean_homo_err,
                         r.std_homo_err, r.max_homo_err, r.min_homo_err, r.median_homo_err,
                         r.is_valid, r.computation_time)
                        for r in batch
                    ])

                    # Mark as processed
                    conn.executemany("""
                        INSERT OR IGNORE INTO homography_processed (match_id) VALUES (?)
                    """, [(r.match_id,) for r in batch])

                    # Update matches table with homography status
                    conn.executemany("""
                        UPDATE matches 
                        SET has_homography = 1, 
                            homography_quality = ?
                        WHERE id = ?
                    """, [(r.mean_homo_err if r.is_valid else None, r.match_id) for r in batch])

                    total_inserted += len(batch)

                conn.execute("COMMIT")
                print(f"Successfully inserted {total_inserted} homography results")

            except Exception as e:
                conn.execute("ROLLBACK")
                print(f"Database error, rolled back transaction: {e}")
                raise e

    def get_homography_statistics(self) -> Dict:
        """Get comprehensive statistics about homography errors."""
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_computed,
                    SUM(is_valid) as valid_count,
                    AVG(CASE WHEN is_valid = 1 THEN mean_homo_err END) as avg_error,
                    MIN(CASE WHEN is_valid = 1 THEN mean_homo_err END) as min_error,
                    MAX(CASE WHEN is_valid = 1 THEN mean_homo_err END) as max_error,
                    AVG(computation_time) as avg_computation_time
                FROM homography_errors
            """)
            result = cursor.fetchone()

            # Count matches pending homography computation
            cursor2 = conn.execute("""
                SELECT COUNT(*) FROM matches m
                LEFT JOIN homography_processed hp ON m.id = hp.match_id
                WHERE hp.match_id IS NULL
            """)
            pending = cursor2.fetchone()[0]

            return {
                'total_computed': result[0] or 0,
                'valid_count': result[1] or 0,
                'avg_error': result[2] or 0.0,
                'min_error': result[3] or 0.0,
                'max_error': result[4] or 0.0,
                'avg_computation_time': result[5] or 0.0,
                'pending_computation': pending
            }

    def get_best_matches_with_homography(self, limit: int = 1000, max_error: float = 10.0) -> Iterator[Tuple]:
        """
        Get best matches based on homography error threshold.

        Args:
            limit: Maximum number of results
            max_error: Maximum mean homography error to include

        Yields:
            Tuple: (file1, file2, match_count, mean_homo_err, is_validated)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT m.file1, m.file2, m.match_count, h.mean_homo_err, m.is_validated
                FROM matches m
                JOIN homography_errors h ON m.id = h.match_id
                WHERE h.is_valid = 1 AND h.mean_homo_err <= ?
                ORDER BY h.mean_homo_err ASC
                LIMIT ?
            """, (max_error, limit))

            for row in cursor:
                yield row


class ErrorCacheManager:
    """Enhanced cache manager for homography projection error arrays."""

    def __init__(self, cache_dir: str = "error_cache"):
        """Initialize cache manager with specified directory."""
        self.cache_dir = cache_dir
        self.stats = {'hits': 0, 'misses': 0}
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, file1: str, file2: str) -> str:
        """Generate standardized cache key for image pair."""
        # Use sorted order to ensure consistency
        f1, f2 = sorted([os.path.basename(file1), os.path.basename(file2)])
        return f"{f1}_{f2}"

    def _get_cache_file_path(self, file1: str, file2: str) -> str:
        """Generate cache file path for image pair."""
        cache_key = self._get_cache_key(file1, file2)
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def load_cache(self, file1: str, file2: str) -> Optional[np.ndarray]:
        """Load cached error array for image pair."""
        cache_file = self._get_cache_file_path(file1, file2)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.stats['hits'] += 1
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache {cache_file}: {e}")
                os.remove(cache_file)  # Remove corrupted cache
        self.stats['misses'] += 1
        return None

    def save_cache(self, file1: str, file2: str, errors: np.ndarray):
        """Save computed error array to cache."""
        cache_file = self._get_cache_file_path(file1, file2)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(errors, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving cache {cache_file}: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total * 100 if total > 0 else 0
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'cache_size_mb': self._get_cache_size_mb()
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)


class FeatureCache:
    """Simple feature cache for SIFT descriptors."""

    def __init__(self, cache_dir: str):
        """Initialize feature cache."""
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.max_memory_cache = 1000
        os.makedirs(cache_dir, exist_ok=True)

    def get_features(self, image_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Get or compute SIFT features for an image."""
        image_key = os.path.basename(image_path)

        # Check memory cache
        if image_key in self.memory_cache:
            return self.memory_cache[image_key]

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{image_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    kp_data, descriptors = pickle.load(f)
                    # Reconstruct KeyPoint objects
                    keypoints = [cv2.KeyPoint(x=p['x'], y=p['y'], size=p['size'],
                                             angle=p['angle'], response=p['response'],
                                             octave=p['octave'], class_id=p['class_id'])
                                for p in kp_data]
                    result = (keypoints, descriptors)

                    # Add to memory cache if space
                    if len(self.memory_cache) < self.max_memory_cache:
                        self.memory_cache[image_key] = result
                    return result
            except Exception as e:
                print(f"Cache error for {image_key}: {e}")

        # Compute features
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            return [], None

        # Save to disk cache
        try:
            kp_data = [{'x': kp.pt[0], 'y': kp.pt[1], 'size': kp.size,
                       'angle': kp.angle, 'response': kp.response,
                       'octave': kp.octave, 'class_id': kp.class_id}
                      for kp in keypoints]
            with open(cache_file, 'wb') as f:
                pickle.dump((kp_data, descriptors), f)
        except Exception as e:
            print(f"Error caching features for {image_key}: {e}")

        result = (keypoints, descriptors)

        # Add to memory cache
        if len(self.memory_cache) < self.max_memory_cache:
            self.memory_cache[image_key] = result

        return result


class HomographyErrorCalculator:
    """Computes homography matrices and projection errors for matched image pairs."""

    def __init__(self, feature_cache: FeatureCache, error_cache: ErrorCacheManager,
                 image_base_path: str):
        """Initialize homography error calculator."""
        self.feature_cache = feature_cache
        self.error_cache = error_cache
        self.image_base_path = image_base_path

    def _reconstruct_matches(self, matches_data: bytes) -> List[Tuple[int, int, float]]:
        """Reconstruct match data from serialized format."""
        try:
            return pickle.loads(matches_data)
        except Exception as e:
            print(f"Error deserializing matches: {e}")
            return []

    def _compute_homography_and_errors(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[Tuple[int, int, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute homography matrix and projection errors."""
        if len(matches) < 4:
            return None, np.array([])

        # Extract matched points
        ptsA = np.array([kp1[m[0]].pt for m in matches], dtype="float32")
        ptsB = np.array([kp2[m[1]].pt for m in matches], dtype="float32")

        # Compute homography using RANSAC
        try:
            H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
            if H is None:
                return None, np.array([])
        except cv2.error as e:
            print(f"Homography computation error: {e}")
            return None, np.array([])

        # Compute projection errors
        errors = []
        for i, (queryIdx, trainIdx, _) in enumerate(matches):
            # Get original points
            pt_a = ptsA[i]
            pt_b = ptsB[i]

            # Convert to homogeneous coordinates
            pt_b_h = np.array([pt_b[0], pt_b[1], 1.0])

            # Project point
            projected = H @ pt_b_h

            # Check for degenerate case
            if abs(projected[2]) < 1e-8:
                errors.append(float('inf'))
                continue

            # Normalize
            projected_normalized = projected[:2] / projected[2]

            # Calculate error
            error = np.linalg.norm(projected_normalized - pt_a)
            errors.append(error)

        return H, np.array(errors)

    def calculate_errors_for_match(
        self,
        match_id: int,
        file1: str,
        file2: str,
        matches_data: bytes
    ) -> HomographyResult:
        """Calculate homography errors for a single match."""
        start_time = time.time()

        # Build full paths
        full_path1 = os.path.join(self.image_base_path, file1.split('-')[0] + '-' + file1.split('-')[1] + '-' + file1.split('-')[2].split('_')[0], file1)
        full_path2 = os.path.join(self.image_base_path, file2.split('-')[0] + '-' + file2.split('-')[1] + '-' + file2.split('-')[2].split('_')[0], file2)

        if os.path.exists(full_path1) and os.path.exists(full_path2):

            # Check cache first
            cached_errors = self.error_cache.load_cache(file1, file2)

            if cached_errors is None:
                # Reconstruct matches
                matches = self._reconstruct_matches(matches_data)

                if not matches:
                    return HomographyResult(
                        match_id=match_id,
                        sum_homo_err=-1,
                        len_homo_err=0,
                        mean_homo_err=-1,
                        std_homo_err=-1,
                        max_homo_err=-1,
                        min_homo_err=-1,
                        median_homo_err=-1,
                        is_valid=False,
                        computation_time=time.time() - start_time
                    )

                try:
                    # Get features
                    kp1, des1 = self.feature_cache.get_features(full_path1)
                    kp2, des2 = self.feature_cache.get_features(full_path2)

                    if des1 is None or des2 is None:
                        return HomographyResult(
                            match_id=match_id,
                            sum_homo_err=-1,
                            len_homo_err=0,
                            mean_homo_err=-1,
                            std_homo_err=-1,
                            max_homo_err=-1,
                            min_homo_err=-1,
                            median_homo_err=-1,
                            is_valid=False,
                            computation_time=time.time() - start_time
                        )

                    # Compute homography and errors
                    H, errors = self._compute_homography_and_errors(kp1, kp2, matches)

                    if len(errors) > 0:
                        # Save to cache
                        self.error_cache.save_cache(file1, file2, errors)
                    else:
                        errors = np.array([])

                except Exception as e:
                    print(f"Error processing {file1} vs {file2}: {e}")
                    errors = np.array([])
            else:
                errors = cached_errors

            # Calculate statistics
            if len(errors) > 0:
                # Filter out infinite values
                finite_errors = errors[np.isfinite(errors)]

                if len(finite_errors) > 0:
                    return HomographyResult(
                        match_id=match_id,
                        sum_homo_err=float(np.sum(finite_errors)),
                        len_homo_err=len(finite_errors),
                        mean_homo_err=float(np.mean(finite_errors)),
                        std_homo_err=float(np.std(finite_errors, ddof=1)) if len(finite_errors) > 1 else 0.0,
                        max_homo_err=float(np.max(finite_errors)),
                        min_homo_err=float(np.min(finite_errors)),
                        median_homo_err=float(np.median(finite_errors)),
                        is_valid=True,
                        computation_time=time.time() - start_time
                    )

        # Return invalid result
        return HomographyResult(
            match_id=match_id,
            sum_homo_err=-1,
            len_homo_err=0,
            mean_homo_err=-1,
            std_homo_err=-1,
            max_homo_err=-1,
            min_homo_err=-1,
            median_homo_err=-1,
            is_valid=False,
            computation_time=time.time() - start_time
        )


class ParallelHomographyProcessor:
    """Parallel processor for homography error computation."""

    def __init__(
        self,
        db_path: str,
        image_base_path: str,
        feature_cache_dir: str,
        error_cache_dir: str,
        num_workers: int = 4
    ):
        """Initialize parallel processor."""
        self.db = HomographyDatabaseManager(db_path)
        self.num_workers = num_workers

        # Create caches
        self.feature_cache = FeatureCache(feature_cache_dir)
        self.error_cache = ErrorCacheManager(error_cache_dir)

        # Create calculator
        self.calculator = HomographyErrorCalculator(
            self.feature_cache,
            self.error_cache,
            image_base_path
        )

    def _process_batch(self, matches: List[Tuple]) -> List[HomographyResult]:
        """Process a batch of matches."""
        results = []
        for match_id, file1, file2, match_count, matches_data in matches:
            try:
                result = self.calculator.calculate_errors_for_match(
                    match_id, file1, file2, matches_data
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing match {match_id}: {e}")
                # Add failed result
                results.append(HomographyResult(
                    match_id=match_id,
                    sum_homo_err=-1,
                    len_homo_err=0,
                    mean_homo_err=-1,
                    std_homo_err=-1,
                    max_homo_err=-1,
                    min_homo_err=-1,
                    median_homo_err=-1,
                    is_valid=False,
                    computation_time=0
                ))
        return results

    def run_parallel_processing(self, batch_size: int = 100, limit: int = None):
        """
        Process all unprocessed matches in parallel.

        Args:
            batch_size: Number of matches per batch
            limit: Maximum number of matches to process
        """
        start_time = time.time()

        # Get matches to process
        matches_to_process = list(self.db.get_unprocessed_matches(limit))
        total_matches = len(matches_to_process)

        if total_matches == 0:
            print("No unprocessed matches found!")
            return

        print(f"Processing {total_matches:,} matches with {self.num_workers} workers")

        # Process in parallel batches
        all_results = []
        processed_count = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit batches
            futures = []
            for i in range(0, total_matches, batch_size):
                batch = matches_to_process[i:i + batch_size]
                future = executor.submit(self._process_batch, batch)
                futures.append(future)

            print(f"Submitted {len(futures)} batches")

            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_results = future.result()
                all_results.extend(batch_results)
                processed_count += len(batch_results)

                # Save to database periodically
                if len(all_results) >= 1000:
                    self.db.batch_insert_homography_errors(all_results)
                    all_results = []

            # Save remaining results
            if all_results:
                self.db.batch_insert_homography_errors(all_results)

        # Report statistics
        elapsed_time = time.time() - start_time
        stats = self.db.get_homography_statistics()
        cache_stats = self.error_cache.get_cache_stats()

        print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
        print(f"Processing rate: {processed_count / elapsed_time:.1f} matches/second")
        print(f"\nHomography Statistics:")
        print(f"  Total computed: {stats['total_computed']:,}")
        print(f"  Valid results: {stats['valid_count']:,}")
        print(f"  Average error: {stats['avg_error']:.2f}")
        print(f"  Min error: {stats['min_error']:.2f}")
        print(f"  Max error: {stats['max_error']:.2f}")
        print(f"  Pending computation: {stats['pending_computation']:,}")
        print(f"\nCache Statistics:")
        print(f"  Cache hits: {cache_stats['hits']:,}")
        print(f"  Cache misses: {cache_stats['misses']:,}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"  Cache size: {cache_stats['cache_size_mb']:.1f} MB")

    def export_results_to_csv(self, output_path: str, limit: int = 10000, max_error: float = 10.0):
        """Export best matches with homography errors to CSV."""
        import csv

        print(f"Exporting matches with max error {max_error} to: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['file1', 'file2', 'match_count', 'mean_homo_err', 'is_validated'])

            # Export results
            exported_count = 0
            for row in self.db.get_best_matches_with_homography(limit, max_error):
                writer.writerow(row)
                exported_count += 1

        print(f"Exported {exported_count} matches to CSV")


def load_config():
    """Load configuration from environment variables."""
    load_dotenv()

    base_path = os.getenv('BASE_PATH')
    if not base_path:
        raise ValueError("BASE_PATH not found in .env file")

    model_type = os.getenv('MODEL_TYPE', 'default')

    config = {
        'base_path': base_path,
        'model_type': model_type,
        'db_path': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('DB_NAME', 'matches.db')),
        'image_base_path': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('PATCHES_DIR', 'patches')),
        'feature_cache_dir': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('PATCHES_CACHE', 'cache')),
        'error_cache_dir': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('ERROR_CACHE', 'error_cache')),
        'output_csv': os.path.join(base_path, f"OUTPUT_{model_type}", os.getenv('HOMO_CSV', 'homography_matches.csv')),
        'num_workers': int(os.getenv('NUM_WORKERS', '8')),
        'batch_size': int(os.getenv('BATCH_SIZE', '100')),
        'process_limit': int(os.getenv('PROCESS_LIMIT', '0')) or None,
        'max_error': float(os.getenv('MAX_HOMO_ERROR', '10.0')),
        'export_limit': int(os.getenv('EXPORT_LIMIT', '10000'))
    }

    return config


def main():
    """Main execution function for database-integrated homography processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Database-Integrated Homography Error Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["process", "export", "stats", "all"],
        default="all",
        help="Operation mode (default: all)"
    )

    parser.add_argument(
        "--config",
        help="Path to .env configuration file"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        load_dotenv(args.config)

    try:
        config = load_config()

        print("=" * 60)
        print("DATABASE-INTEGRATED HOMOGRAPHY ERROR CALCULATOR")
        print("=" * 60)
        print(f"Database: {config['db_path']}")
        print(f"Image path: {config['image_base_path']}")
        print(f"Workers: {config['num_workers']}")
        print("=" * 60)

        # Initialize processor
        processor = ParallelHomographyProcessor(
            db_path=config['db_path'],
            image_base_path=config['image_base_path'],
            feature_cache_dir=config['feature_cache_dir'],
            error_cache_dir=config['error_cache_dir'],
            num_workers=config['num_workers']
        )

        if args.mode in ["process", "all"]:
            print("\nSTARTING HOMOGRAPHY COMPUTATION")
            print("-" * 60)
            processor.run_parallel_processing(
                batch_size=config['batch_size'],
                limit=config['process_limit']
            )

        if args.mode in ["export", "all"]:
            print("\nEXPORTING RESULTS TO CSV")
            print("-" * 60)
            processor.export_results_to_csv(
                output_path=config['output_csv'],
                limit=config['export_limit'],
                max_error=config['max_error']
            )

        if args.mode in ["stats", "all"]:
            print("\nDATABASE STATISTICS")
            print("-" * 60)
            stats = processor.db.get_homography_statistics()
            for key, value in stats.items():
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")

        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nRequired environment variables:")
        print("  BASE_PATH=/path/to/data")
        print("  MODEL_TYPE=experiment_name")

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        print("The process can be resumed by running the same command again.")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()