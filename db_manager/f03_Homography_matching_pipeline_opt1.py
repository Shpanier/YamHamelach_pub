"""
Database-Integrated Homography Error Calculator with Enhanced Progress Tracking

This module extends the original homography error computation to work directly with
the SQLite database from the fragment matching pipeline. It processes SIFT match data,
computes homography matrices and projection errors, and stores results in the database.

The integrated pipeline:
1. Reads SIFT match pairs from the matches database table
2. SKIPS already computed homography calculations
3. Computes homography matrices and projection errors for new matches only
4. Caches error arrays to disk for reuse
5. Calculates statistical metrics (sum, mean, std, count)
6. Updates database with homography error statistics
7. Provides comprehensive analysis and export capabilities

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
from typing import List, Tuple, Dict, Optional, Iterator, Set
from dataclasses import dataclass
from contextlib import contextmanager
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta

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


class ProgressTracker:
    """Track and display progress with ETA estimation."""

    def __init__(self, total: int, desc: str = "Processing"):
        """Initialize progress tracker."""
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.desc = desc
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms minimum

    def update(self, n: int = 1):
        """Update progress count."""
        self.processed += n
        current_time = time.time()

        # Only update display if enough time has passed
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            self._display_progress()

    def _display_progress(self):
        """Display progress bar with statistics."""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.processed) / rate if rate > 0 else 0

        # Format ETA
        if eta > 3600:
            eta_str = f"{eta/3600:.1f}h"
        elif eta > 60:
            eta_str = f"{eta/60:.1f}m"
        else:
            eta_str = f"{eta:.0f}s"

        # Calculate percentage
        percentage = (self.processed / self.total * 100) if self.total > 0 else 0

        # Create progress bar
        bar_length = 40
        filled = int(bar_length * self.processed / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)

        # Display
        print(f"\r{self.desc}: |{bar}| {percentage:.1f}% "
              f"[{self.processed:,}/{self.total:,}] "
              f"Rate: {rate:.1f}/s ETA: {eta_str}    ", end='', flush=True)

    def finish(self):
        """Mark progress as complete."""
        self.processed = self.total
        self._display_progress()
        print()  # New line after completion


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
        # Cache for already processed match IDs
        self._processed_cache = None
        self._cache_lock = threading.Lock()

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

            # Add homography quality columns to matches table if they don't exist
            # Check if columns exist first
            cursor = conn.execute("PRAGMA table_info(matches)")
            existing_columns = [row[1] for row in cursor.fetchall()]

            if 'has_homography' not in existing_columns:
                conn.execute("ALTER TABLE matches ADD COLUMN has_homography INTEGER DEFAULT 0")

            if 'homography_quality' not in existing_columns:
                conn.execute("ALTER TABLE matches ADD COLUMN homography_quality REAL")

            conn.commit()
            print(f"✓ Homography tables initialized in database: {os.path.basename(self.db_path)}")

    def load_processed_ids_to_cache(self) -> Set[int]:
        """Load all processed match IDs into memory for fast lookup."""
        with self._cache_lock:
            if self._processed_cache is None:
                print("Loading processed match IDs into cache...", end='', flush=True)
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT match_id FROM homography_errors")
                    self._processed_cache = set(row[0] for row in cursor)
                print(f" ✓ Loaded {len(self._processed_cache):,} IDs")
            return self._processed_cache

    def is_match_processed(self, match_id: int) -> bool:
        """Check if a match has already been processed (fast cached version)."""
        if self._processed_cache is None:
            self.load_processed_ids_to_cache()
        return match_id in self._processed_cache

    def add_to_processed_cache(self, match_ids: List[int]):
        """Add newly processed match IDs to the cache."""
        with self._cache_lock:
            if self._processed_cache is None:
                self._processed_cache = set()
            self._processed_cache.update(match_ids)

    def get_unprocessed_matches(self, limit: int = None) -> Iterator[Tuple]:
        """
        Get matches that haven't had homography errors computed yet.

        This now uses a more efficient LEFT JOIN approach to find unprocessed matches.

        Args:
            limit: Maximum number of matches to return

        Yields:
            Tuple: (id, file1, file2, match_count, matches_data)
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT m.id, m.file1, m.file2, m.match_count, m.matches_data
                FROM matches m
                LEFT JOIN homography_errors he ON m.id = he.match_id
                WHERE he.match_id IS NULL
                ORDER BY m.match_count DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            for row in cursor:
                yield row

    def get_unprocessed_matches_batch(self, batch_size: int = 1000, offset: int = 0) -> List[Tuple]:
        """
        Get a batch of unprocessed matches with pagination support.

        Args:
            batch_size: Number of matches to return
            offset: Number of matches to skip

        Returns:
            List of match tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT m.id, m.file1, m.file2, m.match_count, m.matches_data
                FROM matches m
                LEFT JOIN homography_errors he ON m.id = he.match_id
                WHERE he.match_id IS NULL
                ORDER BY m.match_count DESC
                LIMIT ? OFFSET ?
            """, (batch_size, offset))

            return cursor.fetchall()

    def count_unprocessed_matches(self) -> int:
        """Get the total count of unprocessed matches."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) 
                FROM matches m
                LEFT JOIN homography_errors he ON m.id = he.match_id
                WHERE he.match_id IS NULL
            """)
            return cursor.fetchone()[0]

    def batch_insert_homography_errors(self, results: List[HomographyResult], batch_size: int = 1000):
        """
        Insert homography error results in batches.

        Args:
            results: List of HomographyResult objects
            batch_size: Number of records per transaction
        """
        if not results:
            return

        total_inserted = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for i in range(0, len(results), batch_size):
                    batch = results[i:i + batch_size]

                    # Insert homography errors (using INSERT OR IGNORE to skip duplicates)
                    conn.executemany("""
                        INSERT OR IGNORE INTO homography_errors 
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

                    # Update matches table with homography status
                    conn.executemany("""
                        UPDATE matches 
                        SET has_homography = 1, 
                            homography_quality = ?
                        WHERE id = ?
                    """, [(r.mean_homo_err if r.is_valid else None, r.match_id) for r in batch])

                    total_inserted += len(batch)

                conn.execute("COMMIT")

                # Update the in-memory cache
                self.add_to_processed_cache([r.match_id for r in results])

            except Exception as e:
                conn.execute("ROLLBACK")
                print(f"\n❌ Database error, rolled back transaction: {e}")
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
            pending = self.count_unprocessed_matches()

            # Count total matches
            cursor3 = conn.execute("SELECT COUNT(*) FROM matches")
            total_matches = cursor3.fetchone()[0]

            return {
                'total_matches': total_matches,
                'total_computed': result[0] or 0,
                'valid_count': result[1] or 0,
                'avg_error': result[2] or 0.0,
                'min_error': result[3] or 0.0,
                'max_error': result[4] or 0.0,
                'avg_computation_time': result[5] or 0.0,
                'pending_computation': pending,
                'completion_percentage': ((result[0] or 0) / total_matches * 100) if total_matches > 0 else 0
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
                 image_base_path: str, db_manager: HomographyDatabaseManager):
        """Initialize homography error calculator."""
        self.feature_cache = feature_cache
        self.error_cache = error_cache
        self.image_base_path = image_base_path
        self.db_manager = db_manager

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
    ) -> Optional[HomographyResult]:
        """
        Calculate homography errors for a single match.

        Returns None if already processed, HomographyResult otherwise.
        """
        # SKIP if already processed
        if self.db_manager.is_match_processed(match_id):
            return None

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
    """Parallel processor for homography error computation with enhanced progress tracking."""

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

        # Create calculator with DB manager reference
        self.calculator = HomographyErrorCalculator(
            self.feature_cache,
            self.error_cache,
            image_base_path,
            self.db  # Pass DB manager for checking processed status
        )

        # Progress tracking
        self.progress_lock = threading.Lock()
        self.total_processed = 0
        self.total_valid = 0
        self.total_invalid = 0

    def _process_batch(self, batch_data: Tuple[List[Tuple], int, int]) -> Tuple[List[HomographyResult], int, int, int]:
        """
        Process a batch of matches with progress tracking.

        Returns: (results, processed_count, valid_count, invalid_count)
        """
        matches, batch_idx, total_batches = batch_data
        results = []
        skipped = 0
        valid = 0
        invalid = 0

        # Create a mini progress bar for this batch
        batch_desc = f"Batch {batch_idx}/{total_batches}"

        for match_id, file1, file2, match_count, matches_data in matches:
            # Double-check if already processed (in case of race conditions)
            if self.db.is_match_processed(match_id):
                skipped += 1
                continue

            try:
                result = self.calculator.calculate_errors_for_match(
                    match_id, file1, file2, matches_data
                )
                if result is not None:  # None means it was already processed
                    results.append(result)
                    if result.is_valid:
                        valid += 1
                    else:
                        invalid += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"\n⚠️  Error processing match {match_id}: {e}")
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
                invalid += 1

        return results, len(results), valid, invalid

    def run_parallel_processing(self, batch_size: int = 100, limit: int = None):
        """
        Process all unprocessed matches in parallel with comprehensive progress tracking.

        Args:
            batch_size: Number of matches per batch
            limit: Maximum number of matches to process
        """
        start_time = time.time()

        print("\n" + "="*70)
        print(" HOMOGRAPHY ERROR CALCULATION")
        print("="*70)

        # Load processed IDs into cache for fast lookup
        print("\n📊 Initialization:")
        self.db.load_processed_ids_to_cache()

        # Count unprocessed matches
        print("  Counting unprocessed matches...", end='', flush=True)
        unprocessed_count = self.db.count_unprocessed_matches()
        print(f" {unprocessed_count:,} found")

        if unprocessed_count == 0:
            print("\n✅ All homography calculations are complete!")
            stats = self.db.get_homography_statistics()
            print(f"  Total matches: {stats['total_matches']:,}")
            print(f"  Valid results: {stats['valid_count']:,}")
            print(f"  Average error: {stats['avg_error']:.2f}")
            return

        # Determine total to process
        if limit and limit < unprocessed_count:
            total_to_process = limit
            print(f"  Processing limited to: {limit:,} matches")
        else:
            total_to_process = unprocessed_count
            print(f"  Processing all: {total_to_process:,} matches")

        print(f"  Workers: {self.num_workers}")
        print(f"  Batch size: {batch_size}")

        # Prepare batches
        print("\n📦 Preparing batches...")
        batches = []
        offset = 0
        batch_idx = 1

        # Progress bar for batch preparation
        prep_pbar = tqdm(total=total_to_process, desc="  Preparing", unit="matches")

        while offset < total_to_process:
            current_batch_size = min(batch_size, total_to_process - offset)
            batch = self.db.get_unprocessed_matches_batch(current_batch_size, offset)

            if not batch:
                break

            batches.append((batch, batch_idx, 0))  # Will update total_batches later
            batch_idx += 1
            offset += len(batch)
            prep_pbar.update(len(batch))

        prep_pbar.close()

        # Update total batches count
        total_batches = len(batches)
        batches = [(b[0], b[1], total_batches) for b in batches]

        print(f"  ✓ Created {total_batches} batches")

        # Process in parallel
        print("\n🚀 Processing matches:")
        print("-"*70)

        all_results = []
        total_computed = 0
        total_valid = 0
        total_invalid = 0

        # Main progress bar
        main_pbar = tqdm(
            total=total_to_process,
            desc="Overall Progress",
            unit="matches",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        # Batch progress bar
        batch_pbar = tqdm(
            total=total_batches,
            desc="Batches Complete",
            unit="batch",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]'
        )

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batches
            futures_to_batch = {
                executor.submit(self._process_batch, batch_data): batch_data
                for batch_data in batches
            }

            # Process completed batches
            for future in as_completed(futures_to_batch):
                batch_results, processed, valid, invalid = future.result()

                # Update counters
                all_results.extend(batch_results)
                total_computed += processed
                total_valid += valid
                total_invalid += invalid

                # Update progress bars
                main_pbar.update(processed)
                batch_pbar.update(1)

                # Update main progress bar postfix with statistics
                main_pbar.set_postfix({
                    'Valid': f'{total_valid:,}',
                    'Invalid': f'{total_invalid:,}',
                    'Cache': f'{self.error_cache.stats["hits"]}/{self.error_cache.stats["hits"]+self.error_cache.stats["misses"]}'
                })

                # Save to database periodically
                if len(all_results) >= 1000:
                    batch_pbar.set_description("Saving to database...")
                    self.db.batch_insert_homography_errors(all_results)
                    all_results = []
                    batch_pbar.set_description("Batches Complete")

        # Close progress bars
        main_pbar.close()
        batch_pbar.close()

        # Save remaining results
        if all_results:
            print("\n💾 Saving final results to database...", end='', flush=True)
            self.db.batch_insert_homography_errors(all_results)
            print(" ✓")

        # Calculate and display final statistics
        elapsed_time = time.time() - start_time
        stats = self.db.get_homography_statistics()
        cache_stats = self.error_cache.get_cache_stats()

        print("\n" + "="*70)
        print(" PROCESSING COMPLETED")
        print("="*70)

        # Time statistics
        print(f"\n⏱️  Time Statistics:")
        print(f"  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"  Processing rate: {total_computed / elapsed_time:.1f} matches/second" if elapsed_time > 0 else "N/A")

        # Processing statistics
        print(f"\n📊 Processing Results:")
        print(f"  Newly computed: {total_computed:,}")
        print(f"  Valid results: {total_valid:,} ({total_valid/total_computed*100:.1f}%)" if total_computed > 0 else "  Valid results: 0")
        print(f"  Invalid results: {total_invalid:,} ({total_invalid/total_computed*100:.1f}%)" if total_computed > 0 else "  Invalid results: 0")

        # Database statistics
        print(f"\n📈 Database Statistics:")
        print(f"  Total matches: {stats['total_matches']:,}")
        print(f"  Total computed: {stats['total_computed']:,}")
        print(f"  Completion: {stats['completion_percentage']:.1f}%")

        # Progress bar for completion
        completion_bar_length = 50
        filled = int(completion_bar_length * stats['completion_percentage'] / 100)
        bar = '█' * filled + '░' * (completion_bar_length - filled)
        print(f"  Progress: |{bar}| {stats['completion_percentage']:.1f}%")

        print(f"  Remaining: {stats['pending_computation']:,}")

        # Error statistics
        if stats['valid_count'] > 0:
            print(f"\n📐 Error Statistics (valid matches only):")
            print(f"  Average error: {stats['avg_error']:.2f} pixels")
            print(f"  Min error: {stats['min_error']:.2f} pixels")
            print(f"  Max error: {stats['max_error']:.2f} pixels")

        # Cache statistics
        print(f"\n💾 Cache Performance:")
        print(f"  Cache hits: {cache_stats['hits']:,}")
        print(f"  Cache misses: {cache_stats['misses']:,}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"  Cache size: {cache_stats['cache_size_mb']:.1f} MB")

        print("\n" + "="*70)

    def export_results_to_csv(self, output_path: str, limit: int = 10000, max_error: float = 10.0):
        """Export best matches with homography errors to CSV with progress tracking."""
        print(f"\n📁 Exporting matches to CSV")
        print(f"  Output file: {output_path}")
        print(f"  Max error threshold: {max_error} pixels")
        print(f"  Max results: {limit:,}")

        print("\n  Querying database...", end='', flush=True)

        # Get count first for progress bar
        results = list(self.db.get_best_matches_with_homography(limit, max_error))
        total_results = len(results)
        print(f" found {total_results:,} matches")

        if total_results == 0:
            print("  ⚠️  No matches found with specified criteria")
            return

        print("  Writing CSV file...")

        # Progress bar for export
        with tqdm(total=total_results, desc="  Exporting", unit="rows") as pbar:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(['file1', 'file2', 'match_count', 'mean_homo_err', 'is_validated'])
                pbar.update(0)

                # Write data
                for row in results:
                    writer.writerow(row)
                    pbar.update(1)

        print(f"\n✅ Successfully exported {total_results:,} matches to {os.path.basename(output_path)}")


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
        'process_limit': int(os.getenv('PROCESS_LIMIT', '../0')) or None,
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

        print("\n" + "="*70)
        print(" DATABASE-INTEGRATED HOMOGRAPHY ERROR CALCULATOR")
        print("="*70)
        print(f"  Database: {os.path.basename(config['db_path'])}")
        print(f"  Model type: {config['model_type']}")
        print(f"  Workers: {config['num_workers']}")
        print(f"  Batch size: {config['batch_size']}")

        # Initialize processor
        processor = ParallelHomographyProcessor(
            db_path=config['db_path'],
            image_base_path=config['image_base_path'],
            feature_cache_dir=config['feature_cache_dir'],
            error_cache_dir=config['error_cache_dir'],
            num_workers=config['num_workers']
        )

        if args.mode in ["process", "all"]:
            processor.run_parallel_processing(
                batch_size=config['batch_size'],
                limit=config['process_limit']
            )

        if args.mode in ["export", "all"]:
            processor.export_results_to_csv(
                output_path=config['output_csv'],
                limit=config['export_limit'],
                max_error=config['max_error']
            )

        if args.mode in ["stats", "all"]:
            print("\n📊 DATABASE STATISTICS")
            print("-" * 70)
            stats = processor.db.get_homography_statistics()

            # Create a visual representation
            completion = stats['completion_percentage']
            bar_length = 40
            filled = int(bar_length * completion / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"  Completion: |{bar}| {completion:.1f}%")
            print(f"  Total matches: {stats['total_matches']:,}")
            print(f"  Computed: {stats['total_computed']:,}")
            print(f"  Valid: {stats['valid_count']:,}")
            print(f"  Pending: {stats['pending_computation']:,}")

            if stats['valid_count'] > 0:
                print(f"\n  Error Statistics:")
                print(f"    Average: {stats['avg_error']:.2f} pixels")
                print(f"    Range: {stats['min_error']:.2f} - {stats['max_error']:.2f} pixels")

        print("\n✅ All operations completed successfully!")
        print("="*70 + "\n")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nRequired environment variables:")
        print("  BASE_PATH=/path/to/data")
        print("  MODEL_TYPE=experiment_name")

    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        print("="*70)
        print("ℹ️  The process can be resumed by running the same command again.")
        print("   Already computed homography errors will be skipped automatically.")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()