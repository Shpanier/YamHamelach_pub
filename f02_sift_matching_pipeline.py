"""
Fragment Matching Pipeline

This module provides a comprehensive pipeline for matching image fragments/patches using
SIFT (Scale-Invariant Feature Transform) feature detection and validation against ground
truth data. It combines feature-based matching with metadata validation to identify
genuine fragment matches in archaeological or document analysis workflows.

The pipeline implements two main stages:
1. Feature Matching: Extracts SIFT features and performs brute-force matching between patches
2. Validation: Cross-references results with PAM (Patch Annotation Metadata) to identify
   true positive matches based on scroll/fragment relationships

Key Components:
- DescriptorCacheManager: Handles caching of SIFT features to disk for efficiency
- NaiveImageMatcher: Performs SIFT-based feature matching between image pairs
- FragmentMatcher: Orchestrates the complete feature matching pipeline
- PamProcessor: Validates matches against ground truth metadata and generates final results

The matching process uses SIFT features with brute-force matching and Lowe's ratio test
to filter high-quality matches. Results are cross-validated against metadata to identify
matches between different boxes of the same scroll/fragment combination.

Dependencies:
    - cv2 (OpenCV): Computer vision operations and SIFT feature extraction
    - numpy: Numerical array operations
    - pandas: Data manipulation for CSV processing
    - pickle: Serialization for caching feature data
    - csv: Reading/writing match results
    - tqdm: Progress tracking for long-running operations

Usage:
    # Stage 1: Feature matching
    python fragment_matching_pipeline.py --stage=matching

    # Stage 2: Validation against PAM data
    python fragment_matching_pipeline.py --stage=validation

    # Run complete pipeline
    python fragment_matching_pipeline.py --stage=complete

Performance Notes:
    - SIFT features are cached to disk to avoid recomputation
    - Only patches from different directories are compared in initial matching
    - CSV processing uses chunking for large datasets to manage memory usage
    - Progress can be resumed from checkpoints in both stages
"""

import csv
import itertools
import os
import pickle
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from env_arguments_loader import load_env_arguments

# Increase the CSV field size limit to handle large match data
csv.field_size_limit(sys.maxsize)


class DescriptorCacheManager:
    """
    Manages caching of SIFT descriptors and keypoints to disk for efficient reuse.

    This class handles the storage and retrieval of computed SIFT features, eliminating
    the need to recompute expensive feature extraction for the same images. Features
    are serialized using pickle and stored with the image filename as the key.

    The cache stores both keypoints (with their geometric properties) and descriptors
    (feature vectors) in a structured format that can be efficiently loaded.

    Attributes:
        cache_dir (str): Directory path where cache files are stored

    Cache File Format:
        Each image gets a .pkl file containing:
        {
            "keypoints": [(pt, size, angle, response, octave, class_id), ...],
            "descriptors": numpy.ndarray of shape (n_keypoints, 128)
        }
    """

    def __init__(self, cache_dir):
        """
        Initialize the cache manager with a specified cache directory.

        Args:
            cache_dir (str): Path to the directory where cache files will be stored.
                           Directory will be created if it doesn't exist.

        Example:
            >>> cache_mgr = DescriptorCacheManager("/tmp/sift_cache")
        """
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_file_path(self, image_key: str) -> str:
        """
        Generate the file path for a specific image's cache file.

        Args:
            image_key (str): Base filename of the image (used as cache key)

        Returns:
            str: Full path to the cache file for this image

        Example:
            >>> cache_mgr._get_cache_file_path("patch_123.jpg")
            "/tmp/sift_cache/patch_123.jpg.pkl"
        """
        return os.path.join(self.cache_dir, f"{image_key}.pkl")

    def _is_cached(self, image_key: str) -> bool:
        """
        Check if cached data exists for a specific image.

        Args:
            image_key (str): Base filename of the image

        Returns:
            bool: True if cache file exists, False otherwise

        Example:
            >>> cache_mgr._is_cached("patch_123.jpg")
            True
        """
        return os.path.exists(self._get_cache_file_path(image_key))

    def _load_cache(self, image_key: str) -> Dict:
        """
        Load cached SIFT data for an image.

        Args:
            image_key (str): Base filename of the image

        Returns:
            Dict or None: Dictionary containing keypoints and descriptors,
                         or None if cache file doesn't exist or is corrupted

        Dictionary Structure:
            {
                "keypoints": List of serialized keypoint tuples,
                "descriptors": numpy.ndarray of SIFT descriptors
            }

        Example:
            >>> data = cache_mgr._load_cache("patch_123.jpg")
            >>> data["descriptors"].shape
            (156, 128)  # 156 keypoints, 128-dim descriptors
        """
        cache_file = self._get_cache_file_path(image_key)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, image_key: str, data: Dict):
        """
        Save computed SIFT data to a cache file.

        Args:
            image_key (str): Base filename of the image
            data (Dict): Dictionary containing keypoints and descriptors to cache

        Side Effects:
            Creates or overwrites the cache file for this image

        Example:
            >>> data = {"keypoints": serialized_kp, "descriptors": desc_array}
            >>> cache_mgr._save_cache("patch_123.jpg", data)
        """
        cache_file = self._get_cache_file_path(image_key)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    def _serialize_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[Tuple]:
        """
        Convert OpenCV KeyPoint objects to serializable tuples.

        OpenCV KeyPoint objects cannot be directly pickled, so this method
        extracts their essential properties into tuples that can be saved.

        Args:
            keypoints (List[cv2.KeyPoint]): List of OpenCV keypoint objects

        Returns:
            List[Tuple]: List of tuples containing keypoint properties:
                        (point_coords, size, angle, response, octave, class_id)

        Example:
            >>> kp_list = [cv2.KeyPoint(x=10, y=20, size=5, ...)]
            >>> serialized = cache_mgr._serialize_keypoints(kp_list)
            >>> serialized[0]
            ((10.0, 20.0), 5.0, -1.0, 0.1, 0, -1)
        """
        return [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints
        ]

    def _deserialize_keypoints(self, keypoints_data: List[Tuple]) -> List[cv2.KeyPoint]:
        """
        Convert serialized keypoint tuples back to OpenCV KeyPoint objects.

        Args:
            keypoints_data (List[Tuple]): List of serialized keypoint tuples

        Returns:
            List[cv2.KeyPoint]: List of reconstructed OpenCV KeyPoint objects

        Example:
            >>> tuples = [((10.0, 20.0), 5.0, -1.0, 0.1, 0, -1)]
            >>> keypoints = cache_mgr._deserialize_keypoints(tuples)
            >>> keypoints[0].pt
            (10.0, 20.0)
        """
        return [
            cv2.KeyPoint(
                x=pt[0][0],
                y=pt[0][1],
                size=pt[1],
                angle=pt[2],
                response=pt[3],
                octave=pt[4],
                class_id=pt[5],
            )
            for pt in keypoints_data
        ]

    def process_image(self, file_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Process an image to extract SIFT features, using cache when available.

        This method first checks if SIFT features for the image are already cached.
        If cached data exists, it loads and returns it. Otherwise, it computes
        SIFT features from scratch and caches the results for future use.

        Args:
            file_path (str): Full path to the image file to process

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: Tuple containing:
                - List of detected keypoints
                - Array of SIFT descriptors (shape: n_keypoints x 128)

        Raises:
            ValueError: If the image file cannot be loaded

        Example:
            >>> keypoints, descriptors = cache_mgr.process_image("patch.jpg")
            >>> len(keypoints)
            156
            >>> descriptors.shape
            (156, 128)
        """
        image_key = os.path.basename(file_path)

        if self._is_cached(image_key):
            # Load cached data if available
            cached_data = self._load_cache(image_key)
            if cached_data:
                keypoints = self._deserialize_keypoints(
                    cached_data["keypoints"]
                )
                descriptors = cached_data["descriptors"]
                return keypoints, descriptors

        # If not cached, compute SIFT features and cache the result
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Save the computed data to the cache
        self._save_cache(
            image_key,
            {
                "keypoints": self._serialize_keypoints(keypoints),
                "descriptors": descriptors,
            },
        )

        return keypoints, descriptors


class NaiveImageMatcher:
    """
    Performs feature matching between image pairs using SIFT features and brute-force matching.

    This class implements a straightforward approach to image matching by:
    1. Extracting SIFT descriptors from both images
    2. Using brute-force matcher to find nearest neighbors
    3. Applying Lowe's ratio test to filter high-quality matches

    The matcher uses a ratio threshold of 0.75, which is a standard value
    that provides a good balance between match precision and recall.

    Attributes:
        descriptor_cache (DescriptorCacheManager): Cache manager for SIFT features
    """

    def __init__(self, descriptor_cache: DescriptorCacheManager):
        """
        Initialize the image matcher with a descriptor cache.

        Args:
            descriptor_cache (DescriptorCacheManager): Cache manager for storing/loading
                                                     SIFT features

        Example:
            >>> cache = DescriptorCacheManager("/tmp/cache")
            >>> matcher = NaiveImageMatcher(cache)
        """
        self.descriptor_cache = descriptor_cache

    def calc_matches(self, file1: str, file2: str) -> List[cv2.DMatch]:
        """
        Calculate SIFT feature matches between two images using Lowe's ratio test.

        This method performs the following steps:
        1. Extract SIFT descriptors from both images (via cache)
        2. Use brute-force matcher to find 2 nearest neighbors for each descriptor
        3. Apply Lowe's ratio test: accept match if distance_1 < 0.75 * distance_2
        4. Return list of good matches

        The ratio test helps filter out ambiguous matches where the closest and
        second-closest matches are very similar, indicating low discriminability.

        Args:
            file1 (str): Path to the first image file
            file2 (str): Path to the second image file

        Returns:
            List[cv2.DMatch]: List of good matches that passed the ratio test.
                             Each DMatch contains queryIdx, trainIdx, and distance.

        Note:
            Returns empty list if matching fails due to insufficient features
            or other errors.

        Example:
            >>> matches = matcher.calc_matches("patch1.jpg", "patch2.jpg")
            >>> len(matches)
            23
            >>> matches[0].distance
            45.7  # Euclidean distance between descriptors
        """
        # Get descriptors and keypoints for both images
        kp1, des1 = self.descriptor_cache.process_image(file1)
        kp2, des2 = self.descriptor_cache.process_image(file2)

        try:
            good_matches = []
            bf = cv2.BFMatcher()
            # BFMatcher stands for Brute-Force Matcher. It compares each descriptor
            # from des1 with all the descriptors from des2.
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test to filter out good matches
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        except Exception as e:
            print(f"Error occurred while filtering matches: {e}")
            return []

        return good_matches


class FragmentMatcher:
    """
    Orchestrates the complete fragment matching pipeline for image patch datasets.

    This class manages the entire process of matching image fragments:
    1. Discovers all image files in the dataset directory
    2. Generates all possible image pairs (excluding same-directory pairs)
    3. Performs SIFT-based matching for each pair
    4. Saves results to CSV with resume capability
    5. Tracks progress and handles large datasets efficiently

    The matcher assumes that patches from the same original image are stored
    in the same subdirectory, and only compares patches across different
    subdirectories to find potential matches between different source images.

    Attributes:
        image_base_path (str): Root directory containing image subdirectories
        matcher (NaiveImageMatcher): Matcher instance for performing comparisons
    """

    def __init__(self, image_base_path: str, cache_dir: str):
        """
        Initialize the fragment matcher with dataset and cache paths.

        Args:
            image_base_path (str): Root directory containing image files/subdirectories
            cache_dir (str): Directory for caching SIFT features

        Example:
            >>> fm = FragmentMatcher("/data/patches", "/tmp/sift_cache")
        """
        self.image_base_path = image_base_path
        self.matcher = NaiveImageMatcher(DescriptorCacheManager(cache_dir))

    def get_image_files(self) -> List[str]:
        """
        Recursively discover all JPEG image files in the base directory.

        Walks through all subdirectories of the base path and collects
        full paths to all .jpg files found.

        Returns:
            List[str]: List of full paths to all discovered image files

        Example:
            >>> fm.get_image_files()
            ['/data/patches/img1/patch_1.jpg', '/data/patches/img1/patch_2.jpg',
             '/data/patches/img2/patch_1.jpg', ...]
        """
        image_files = []
        for root, _, files in os.walk(self.image_base_path):
            for file in files:
                if file.endswith(".jpg"):  # Assuming patches are in .jpg format
                    image_files.append(os.path.join(root, file))
        return image_files

    def _get_processed_pairs(self, success_csv: str) -> set:
        """
        Read existing CSV results to determine which image pairs have been processed.

        This enables resume capability by tracking which comparisons have already
        been completed and stored in the results file.

        Args:
            success_csv (str): Path to the CSV file containing previous results

        Returns:
            set: Set of tuples (file1, file2) representing processed pairs

        Example:
            >>> processed = fm._get_processed_pairs("results.csv")
            >>> ("patch1.jpg", "patch2.jpg") in processed
            True
        """
        processed_pairs = set()
        if os.path.exists(success_csv):
            with open(success_csv, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    processed_pairs.add((row["file1"], row["file2"]))
        return processed_pairs

    def calculate_distances(
            self, image_files: List[str], success_csv: str, debug: bool = False
    ) -> None:
        """
        Calculate and save matching distances for all valid image pairs.

        This method performs the core matching computation:
        1. Generates all possible pairs from the image list
        2. Filters out pairs from the same directory
        3. Skips already processed pairs (resume capability)
        4. Computes SIFT matches for each remaining pair
        5. Saves results with match count and detailed match data

        Only pairs with at least one good match are saved to the CSV file.
        Progress is tracked with a progress bar for long-running operations.

        Args:
            image_files (List[str]): List of all image file paths to compare
            success_csv (str): Path to output CSV file for results
            debug (bool): Enable debug output (currently unused)

        Side Effects:
            - Creates or appends to the CSV results file
            - Updates progress bar during processing
            - Flushes results to disk after each successful match

        CSV Output Format:
            file1,file2,distance,matches
            patch1.jpg,patch2.jpg,23,"[(0,5,45.7), (1,12,38.2), ...]"

        Example:
            >>> fm.calculate_distances(image_list, "results.csv")
            Processing Patches: 100%|██████████| 1000/1000 [05:23<00:00, 3.09it/s]
        """
        total_iterations = sum(
            range(1, len(image_files))
        )  # Total number of comparisons
        processed_pairs = self._get_processed_pairs(success_csv)

        with open(success_csv, mode="a", newline="") as file:
            fieldnames = ["file1", "file2", "distance", "matches"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header only if the file is newly created
            if not processed_pairs:
                writer.writeheader()

            with tqdm(
                    total=total_iterations,
                    desc="Processing Patches",
                    disable=False,
            ) as pbar:
                for i, j in itertools.combinations(range(len(image_files)), 2):
                    image_path1 = image_files[i]
                    image_path2 = image_files[j]

                    dirname1 = os.path.dirname(image_path1)
                    dirname2 = os.path.dirname(image_path2)

                    base_name1 = os.path.basename(image_path1)
                    base_name2 = os.path.basename(image_path2)

                    # Different patches can't be from the same directory
                    if dirname1 == dirname2:
                        pbar.update(1)  # Update progress bar
                        continue

                    # Check if the pair has already been processed
                    if (image_path1, image_path2) in processed_pairs or (
                            image_path2,
                            image_path1,
                    ) in processed_pairs:
                        pbar.update(1)  # Update progress bar
                        continue  # Skip this pair

                    # Calculate matches for this pair
                    pbar.update(1)
                    good_matches = self.matcher.calc_matches(
                        image_path1, image_path2
                    )

                    if len(good_matches) <= 0:
                        continue

                    # Write match details to the CSV
                    writer.writerow(
                        {
                            "file1": base_name1,
                            "file2": base_name2,
                            "distance": len(good_matches),
                            "matches": [
                                (m.queryIdx, m.trainIdx, m.distance)
                                for m in good_matches
                            ],
                        }
                    )

                    # Flush to ensure data is written to the file immediately
                    file.flush()

    def run(self, success_csv, debug=False):
        """
        Execute the complete fragment matching pipeline.

        This is the main entry point that orchestrates the entire matching process:
        1. Discovers all image files in the dataset
        2. Performs pairwise matching with progress tracking
        3. Saves results to the specified CSV file

        Args:
            success_csv (str): Path to output CSV file for storing match results
            debug (bool): Enable debug mode (passed to calculate_distances)

        Side Effects:
            - Prints completion message with output file path
            - Creates the output CSV file with match results

        Example:
            >>> fm = FragmentMatcher("/data/patches", "/tmp/cache")
            >>> fm.run("fragment_matches.csv")
            Results written to fragment_matches.csv
        """
        image_files = self.get_image_files()
        self.calculate_distances(image_files, success_csv, debug=debug)
        print(f"Results written to {success_csv}")


class PamProcessor:
    """
    Processes and validates SIFT matches against PAM (Patch Annotation Metadata) ground truth data.

    This class takes the raw SIFT matching results and cross-references them with metadata
    about scroll/fragment relationships to identify true positive matches. It focuses on
    finding matches between different boxes of the same scroll/fragment combination, which
    represent genuine fragment matches.

    The validation process involves:
    1. Loading PAM metadata containing scroll, fragment, and box information
    2. Identifying potential true positive pairs (same scroll/fragment, different boxes)
    3. Cross-referencing with SIFT match results to mark validated matches
    4. Generating a final sorted output with match confidence scores

    Attributes:
        pam_csv_file (str): Path to the PAM metadata CSV file
        image_base_path (str): Root directory containing patch images
        sift_matches_file (str): Path to the SIFT matching results CSV
    """

    def __init__(self, image_base_path: str, pam_csv_file: str, sift_matches_file: str):
        """
        Initialize the PAM processor with required file paths.

        Args:
            image_base_path (str): Root directory containing patch image files
            pam_csv_file (str): Path to PAM metadata CSV file
            sift_matches_file (str): Path to SIFT matching results CSV file

        Raises:
            AssertionError: If any of the required files/directories don't exist

        Example:
            >>> processor = PamProcessor("/data/patches", "/data/pam.csv", "/data/matches.csv")
        """
        self.pam_csv_file = pam_csv_file
        assert os.path.exists(self.pam_csv_file), (
            f"PAM CSV file not found: {self.pam_csv_file}"
        )

        self.image_base_path = image_base_path
        assert os.path.exists(self.image_base_path), (
            f"Image base path not found: {self.image_base_path}"
        )

        self.sift_matches_file = sift_matches_file
        assert os.path.exists(self.sift_matches_file), (
            f"SIFT matches file not found: {self.sift_matches_file}"
        )

    def read_pam_csv(self) -> pd.DataFrame:
        """
        Read the PAM metadata CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing PAM metadata with columns for
                         Scroll, Fragment (Frg), File, Box, and other metadata

        Example:
            >>> df = processor.read_pam_csv()
            >>> df.columns
            Index(['Scroll', 'Frg', 'File', 'Box', ...])
        """
        return pd.read_csv(self.pam_csv_file)

    def read_sift_csv_chunks(self, chunk_size: int = 100000) -> pd.io.parsers.readers.TextFileReader:
        """
        Read the SIFT matches CSV file in chunks to manage memory usage.

        Args:
            chunk_size (int): Number of rows to read per chunk (default: 100000)

        Returns:
            pd.io.parsers.readers.TextFileReader: Chunked CSV reader for processing
                                                 large SIFT match files

        Example:
            >>> for chunk in processor.read_sift_csv_chunks():
            ...     # Process each chunk of SIFT matches
            ...     process_chunk(chunk)
        """
        return pd.read_csv(self.sift_matches_file, chunksize=chunk_size)

    def get_matched_pam_files(self) -> pd.DataFrame:
        """
        Identify potential true positive matches from PAM metadata.

        This method finds pairs of patches that come from the same scroll and fragment
        but different boxes, which represent genuine fragment matches that should be
        validated against SIFT results.

        The process:
        1. Remove rows with NaN Box values
        2. Self-join PAM data on Scroll and Fragment columns
        3. Filter for pairs with different Box values
        4. Return potential match pairs for validation

        Returns:
            pd.DataFrame: DataFrame with columns [Scroll, Frg, File_x, Box_x, File_y, Box_y]
                         representing potential true positive match pairs

        Example:
            >>> matches = processor.get_matched_pam_files()
            >>> matches.head()
               Scroll  Frg    File_x  Box_x    File_y  Box_y
            0      1    2  scroll1_2      1  scroll1_2      3
            1      1    2  scroll1_2      1  scroll1_2      5
        """
        df = self.read_pam_csv()

        # Remove rows where Box is NaN
        df = df.dropna(subset=["Box"])

        # Perform an inner join on the Scroll and Frg columns
        merged_df = pd.merge(df, df, on=["Scroll", "Frg"])

        # Filter the results where the Box values are different
        filtered_df = merged_df[merged_df["Box_x"] != merged_df["Box_y"]]

        # Select only the necessary columns: file1, box1, file2, box2
        result_df = filtered_df[
            ["Scroll", "Frg", "File_x", "Box_x", "File_y", "Box_y"]
        ]
        return result_df

    def get_image_path(self, file_name: str, box: int) -> str:
        """
        Generate the standardized image filename based on file and box information.

        Args:
            file_name (str): Base filename from PAM metadata
            box (int): Box number for the specific patch

        Returns:
            str: Standardized image filename in format "{file_name}_{box}.jpg"

        Example:
            >>> processor.get_image_path("scroll1_frg2", 3)
            "scroll1_frg2_3.jpg"
        """
        return f"{file_name}_{int(box)}.jpg"

    def process_and_save_matches(self, output_filename: str, top_n: int = -1) -> None:
        """
        Process SIFT matches against PAM metadata and save validated results.

        This method performs the core validation process:
        1. Identifies potential true positive pairs from PAM metadata
        2. Processes SIFT matches in chunks to manage memory
        3. Marks matches that correspond to true positive pairs
        4. Sorts results by match confidence (distance/score)
        5. Saves final validated results to output file

        The process uses chunked reading for the SIFT matches to handle large datasets
        efficiently, and includes a temporary file mechanism to manage intermediate results.

        Args:
            output_filename (str): Path for the final validated results CSV file
            top_n (int): Number of top matches to include in final output.
                        If -1, includes all matches (default: -1)

        Side Effects:
            - Creates a temporary file during processing
            - Writes the final sorted results to output_filename
            - Prints progress updates and completion message

        Output CSV Format:
            The output file contains all original SIFT match columns plus a "Match" column
            indicating validation status (1 for validated true positives, 0 otherwise).

        Example:
            >>> processor.process_and_save_matches("validated_matches.csv", top_n=1000)
            Processing Matches: 100%|██████████| 500/500 [02:15<00:00, 3.70it/s]
            Match found: scroll1_2_1.jpg - scroll1_2_3.jpg
            Processing final sorting...
            Processed and sorted results saved to: validated_matches.csv
        """
        matches_df = self.get_matched_pam_files()

        # Create a temporary file for the intermediate results
        temp_output = output_filename + ".temp"

        # Process and save matches with validation
        with open(temp_output, "w") as f:
            header_written = False
            for chunk in self.read_sift_csv_chunks():
                # Add Match column to track validation status
                chunk["Match"] = 0

                # Check each potential true positive pair against current chunk
                for index, row in tqdm(
                        matches_df.iterrows(),
                        total=matches_df.shape[0],
                        desc="Processing Matches",
                ):
                    image_path1 = self.get_image_path(
                        row["File_x"], row["Box_x"]
                    )
                    image_path2 = self.get_image_path(
                        row["File_y"], row["Box_y"]
                    )

                    # Check if both image paths are present in the current chunk
                    # (in either order since matching is bidirectional)
                    matching_rows = chunk[
                        (
                                (chunk["file1"] == image_path1)
                                & (chunk["file2"] == image_path2)
                        )
                        | (
                                (chunk["file2"] == image_path1)
                                & (chunk["file1"] == image_path2)
                        )
                        ].index

                    if len(matching_rows) > 0:
                        chunk.loc[matching_rows, "Match"] = 1
                        print(f"Match found: {image_path1} - {image_path2}")

                # Save the chunk to the temporary output file
                if not header_written:
                    chunk.to_csv(f, index=False)
                    header_written = True
                else:
                    chunk.to_csv(f, index=False, header=False, mode="a")

        # Read temporary file, sort results, and save final output
        print("Processing final sorting...")
        df = pd.read_csv(temp_output)

        # Sort by the third column (distance/score) in descending order
        df_sorted = df.sort_values(by=df.columns[2], ascending=False)

        # Select the top N rows if specified
        if top_n > 0:
            df_sorted_top = df_sorted.head(top_n)
        else:
            df_sorted_top = df_sorted

        # Save the final sorted results
        df_sorted_top.to_csv(output_filename, index=False)

        # Clean up the temporary file
        os.remove(temp_output)

        print(f"Processed and sorted results saved to: {output_filename}")


class FragmentMatchingPipeline:
    """
    Main pipeline controller that orchestrates both feature matching and validation stages.

    This class provides a unified interface for running the complete fragment matching
    workflow, from initial SIFT-based feature matching through final validation against
    ground truth metadata.

    The pipeline can be run in different modes:
    - Feature matching only: Generates initial SIFT matches
    - Validation only: Processes existing SIFT matches against PAM data
    - Complete pipeline: Runs both stages sequentially

    Attributes:
        args: Configuration arguments loaded from environment
        patches_dir (str): Directory containing patch images
        patch_cache_dir (str): Directory for SIFT feature cache
        pam_csv_path (str): Path to PAM metadata file
        sift_matches_path (str): Path to SIFT matches output file
        validated_matches_path (str): Path to final validated matches file
    """

    def __init__(self, args):
        """
        Initialize the pipeline with configuration arguments.

        Args:
            args: Configuration object containing paths and settings

        Example:
            >>> args = load_env_arguments()
            >>> pipeline = FragmentMatchingPipeline(args)
        """
        self.args = args

        # Set up all required paths
        self.patches_dir = os.path.join(args.base_path, "OUTPUT_" + args.model_type, args.patches_dir)
        self.patch_cache_dir = os.path.join(args.base_path, "OUTPUT_" + args.model_type,  args.patches_cache)
        self.pam_csv_path = os.path.join(args.base_path, "OUTPUT_" + args.model_type, args.csv_in)
        self.sift_matches_path = os.path.join(args.base_path, "OUTPUT_" + args.model_type, args.sift_matches)
        self.validated_matches_path = os.path.join(args.base_path,"OUTPUT_" + args.model_type,  args.sift_matches_w_tp)

    def run_feature_matching(self) -> None:
        """
        Execute the SIFT-based feature matching stage of the pipeline.

        This stage:
        1. Discovers all patch images in the dataset
        2. Performs pairwise SIFT feature matching
        3. Saves raw match results to CSV
        4. Uses caching for efficiency and supports resume capability

        Side Effects:
            - Creates SIFT feature cache files
            - Generates raw match results CSV file
            - Prints progress and completion status

        Example:
            >>> pipeline.run_feature_matching()
            Processing Patches: 100%|██████████| 5000/5000 [15:23<00:00, 5.41it/s]
            Results written to /data/sift_matches.csv
        """
        print("Starting SIFT-based feature matching stage...")


        # Initialize and run the fragment matcher
        matcher = FragmentMatcher(self.patches_dir, self.patch_cache_dir)
        matcher.run(success_csv=self.sift_matches_path, debug=self.args.debug)

        print("Feature matching stage completed.")

    def run_validation(self) -> None:
        """
        Execute the PAM validation stage of the pipeline.

        This stage:
        1. Loads PAM metadata and SIFT match results
        2. Identifies potential true positive pairs from metadata
        3. Cross-validates SIFT matches against ground truth
        4. Generates final sorted and validated results

        Side Effects:
            - Creates validated match results CSV file
            - Prints validation progress and match discoveries
            - Removes temporary processing files

        Example:
            >>> pipeline.run_validation()
            Processing Matches: 100%|██████████| 1500/1500 [03:45<00:00, 6.67it/s]
            Match found: scroll1_2_1.jpg - scroll1_2_3.jpg
            Processing final sorting...
            Processed and sorted results saved to: /data/validated_matches.csv
        """
        print("Starting PAM validation stage...")

        # Initialize and run the PAM processor
        processor = PamProcessor(
            self.patches_dir,
            self.pam_csv_path,
            self.sift_matches_path
        )
        processor.process_and_save_matches(self.validated_matches_path, top_n=-1)

        print("Validation stage completed.")

    def run_complete_pipeline(self) -> None:
        """
        Execute the complete fragment matching pipeline.

        Runs both feature matching and validation stages sequentially,
        providing a complete end-to-end solution for fragment matching
        with ground truth validation.

        Side Effects:
            - All effects from both run_feature_matching() and run_validation()
            - Prints overall pipeline status messages

        Example:
            >>> pipeline.run_complete_pipeline()
            Starting complete fragment matching pipeline...
            Starting SIFT-based feature matching stage...
            [... feature matching progress ...]
            Feature matching stage completed.
            Starting PAM validation stage...
            [... validation progress ...]
            Validation stage completed.
            Complete pipeline finished successfully.
        """
        print("Starting complete fragment matching pipeline...")

        # Run feature matching stage
        self.run_feature_matching()

        # Run validation stage
        self.run_validation()

        print("Complete pipeline finished successfully.")


def main():
    """
    Main execution function with support for different pipeline stages.

    Loads configuration from environment variables and runs the appropriate
    pipeline stage based on command line arguments or default behavior.

    Supported execution modes:
    - Feature matching only: python fragment_matching_pipeline.py --stage=matching
    - Validation only: python fragment_matching_pipeline.py --stage=validation
    - Complete pipeline: python fragment_matching_pipeline.py --stage=complete
    - Default (complete): python fragment_matching_pipeline.py

    The function handles argument parsing, pipeline initialization, and
    stage execution based on the specified mode.
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fragment Matching Pipeline")
    parser.add_argument(
        "--stage",
        choices=["matching", "validation", "complete"],
        default="complete",
        help="Pipeline stage to execute (default: complete)"
    )
    cmd_args = parser.parse_args()

    # Load environment configuration
    env_args = load_env_arguments()

    # Initialize pipeline
    pipeline = FragmentMatchingPipeline(env_args)

    # Execute appropriate stage
    if cmd_args.stage == "matching":
        pipeline.run_feature_matching()
    elif cmd_args.stage == "validation":
        pipeline.run_validation()
    else:  # complete
        pipeline.run_complete_pipeline()


if __name__ == "__main__":
    """
    Main execution block for the fragment matching pipeline.

    This block handles the primary execution flow when the script is run directly.
    It supports various execution modes through command line arguments and provides
    a complete workflow for archaeological or document fragment matching tasks.

    The pipeline is designed for scenarios where:
    - Image fragments need to be matched based on visual similarity
    - Ground truth metadata is available for validation
    - Large datasets require efficient processing with caching and resume capability
    - Results need to be ranked and filtered for manual review

    Configuration is loaded from environment variables, allowing for flexible
    deployment across different systems and datasets without code modification.
    """
    main()