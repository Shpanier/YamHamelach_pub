"""
Fragment Match Visualization Script - Homography Error Based

This script displays the top N most similar fragment matches side by side,
sorted by minimum homography error (best geometric consistency).
Reads from the SQLite database created by the fragment matching pipeline
with homography error computation.

Dependencies:
    - cv2 (OpenCV): For image loading and processing
    - matplotlib: For visualization
    - sqlite3: For database access
    - numpy: For image array operations
    - dotenv: For loading configuration
"""

import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import pickle


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
    }

    return config


def get_top_matches_by_homography(db_path: str, limit: int = 10, max_error: float = 10.0) -> List[Tuple]:
    """
    Retrieve top matches from database sorted by homography error (lowest first).

    Args:
        db_path: Path to SQLite database
        limit: Number of top matches to retrieve
        max_error: Maximum homography error threshold

    Returns:
        List of tuples: (file1, file2, match_count, mean_homo_err, std_homo_err, is_validated)
    """
    matches = []

    with sqlite3.connect(db_path) as conn:
        # Check if homography_errors table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='homography_errors'
        """)

        if not cursor.fetchone():
            print("Warning: homography_errors table not found. Run homography computation first.")
            print("Falling back to match count ordering...")

            # Fallback to match count
            cursor = conn.execute("""
                SELECT file1, file2, match_count, -1, -1, is_validated
                FROM matches 
                WHERE match_count > 0
                ORDER BY match_count DESC 
                LIMIT ?
            """, (limit,))
        else:
            # Query with homography errors
            cursor = conn.execute("""
                SELECT 
                    m.file1, 
                    m.file2, 
                    m.match_count,
                    h.mean_homo_err,
                    h.std_homo_err,
                    m.is_validated
                FROM matches m
                INNER JOIN homography_errors h ON m.id = h.match_id
                WHERE h.mean_homo_err > 0 
                    AND h.mean_homo_err <= ?
                ORDER BY h.mean_homo_err ASC
                LIMIT ?
            """, (max_error, limit))

        for row in cursor:
            matches.append(row)

    return matches


def find_image_path(base_path: str, filename: str) -> Optional[str]:
    """
    Find the full path to an image file in the patches directory.

    Args:
        base_path: Base directory containing patches
        filename: Image filename to find

    Returns:
        Full path to image file or None if not found
    """
    # Try to construct path based on filename pattern (e.g., PHerc-1691-Fr-3_45.jpg)
    parts = filename.split('-')
    if len(parts) >= 3:
        # Expected pattern: PHerc-XXXX-Fr-Y_Z.jpg
        subdir = '-'.join(parts[:3]).split('_')[0]
        potential_path = os.path.join(base_path, subdir, filename)
        if os.path.exists(potential_path):
            return potential_path

    # Fallback: search recursively
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)

    return None


def load_and_resize_image(image_path: str, max_size: int = 500) -> np.ndarray:
    """
    Load and resize image for display.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension for display

    Returns:
        Resized image array (RGB)
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create placeholder image
        img = np.ones((max_size, max_size, 3), dtype=np.uint8) * 200
        cv2.putText(img, "Image not found", (10, max_size//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return img

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if needed
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img


def get_quality_color(homo_err: float) -> str:
    """
    Get color based on homography error quality.

    Args:
        homo_err: Mean homography error

    Returns:
        Color string for matplotlib
    """
    if homo_err < 0:
        return 'gray'  # No homography data
    elif homo_err <= 2.0:
        return 'darkgreen'  # Excellent
    elif homo_err <= 5.0:
        return 'green'  # Good
    elif homo_err <= 10.0:
        return 'orange'  # Fair
    else:
        return 'red'  # Poor


def plot_top_matches_by_homography(config: dict, num_matches: int = 10,
                                   max_error: float = 10.0, save_path: str = None):
    """
    Create visualization of top fragment matches sorted by homography error.

    Args:
        config: Configuration dictionary
        num_matches: Number of top matches to display
        max_error: Maximum acceptable homography error
        save_path: Optional path to save the figure
    """
    print(f"Loading top {num_matches} matches with best homography errors...")
    print(f"Maximum error threshold: {max_error} pixels")

    # Get top matches from database
    matches = get_top_matches_by_homography(config['db_path'], limit=num_matches, max_error=max_error)

    if not matches:
        print("No matches found in database with homography errors!")
        print("Please run the homography error computation first.")
        return

    print(f"Found {len(matches)} matches. Creating visualization...")

    # Calculate grid dimensions
    n_rows = min(num_matches, len(matches))

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 3.5 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.3, wspace=0.1)

    # Set main title
    fig.suptitle(f'Top {n_rows} Fragment Matches by Homography Error (Best Geometric Consistency)',
                 fontsize=16, fontweight='bold', y=1.02)

    # Process each match
    for idx, (file1, file2, match_count, mean_homo_err, std_homo_err, is_validated) in enumerate(matches[:num_matches]):
        print(f"Processing match {idx+1}/{n_rows}: {file1} <-> {file2}")
        print(f"  Homography error: {mean_homo_err:.2f} ± {std_homo_err:.2f} pixels ({match_count} matches)")

        # Find full paths to images
        path1 = find_image_path(config['image_base_path'], file1)
        path2 = find_image_path(config['image_base_path'], file2)

        if not path1:
            print(f"  Warning: Could not find {file1}")
        if not path2:
            print(f"  Warning: Could not find {file2}")

        # Load and resize images
        img1 = load_and_resize_image(path1) if path1 else load_and_resize_image("")
        img2 = load_and_resize_image(path2) if path2 else load_and_resize_image("")

        # Create subplot for this match pair
        ax1 = fig.add_subplot(gs[idx, 0])
        ax2 = fig.add_subplot(gs[idx, 1])

        # Display images
        ax1.imshow(img1)
        ax2.imshow(img2)

        # Get quality color based on error
        quality_color = get_quality_color(mean_homo_err)

        # Set titles with homography error info
        if mean_homo_err > 0:
            ax1.set_title(f"{file1}\nError: {mean_homo_err:.2f}±{std_homo_err:.2f}px | {match_count} matches",
                         fontsize=9, color=quality_color)

            # Quality label
            if mean_homo_err <= 2.0:
                quality_label = "Excellent Match"
            elif mean_homo_err <= 5.0:
                quality_label = "Good Match"
            elif mean_homo_err <= 10.0:
                quality_label = "Fair Match"
            else:
                quality_label = "Poor Match"
        else:
            ax1.set_title(f"{file1}\n{match_count} matches (no homography)",
                         fontsize=9, color='gray')
            quality_label = "No Homography"

        ax2.set_title(f"{file2}\n{quality_label} {'✓ Validated' if is_validated else ''}",
                     fontsize=9, color='green' if is_validated else quality_color)

        ax1.axis('off')
        ax2.axis('off')

        # Add match rank with color coding
        rank_color = 'white' if mean_homo_err <= 5.0 else 'yellow'
        ax1.text(0.02, 0.98, f"#{idx+1}", transform=ax1.transAxes,
                fontsize=14, fontweight='bold', color='black',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=rank_color, alpha=0.9))

        # Add homography error badge
        if mean_homo_err > 0:
            ax2.text(0.98, 0.98, f"{mean_homo_err:.1f}px", transform=ax2.transAxes,
                    fontsize=12, fontweight='bold', color='white',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.9))

        # Add border based on quality
        border_width = 3 if mean_homo_err <= 5.0 else 2
        rect1 = patches.Rectangle((0, 0), img1.shape[1]-1, img1.shape[0]-1,
                                 linewidth=border_width, edgecolor=quality_color, facecolor='none')
        rect2 = patches.Rectangle((0, 0), img2.shape[1]-1, img2.shape[0]-1,
                                 linewidth=border_width, edgecolor=quality_color, facecolor='none')
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)

        # Special marker for validated matches
        if is_validated:
            ax1.text(0.98, 0.02, "✓", transform=ax1.transAxes,
                    fontsize=20, fontweight='bold', color='green',
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Add legend
    legend_elements = [
        patches.Patch(color='darkgreen', label='Excellent (≤2px)'),
        patches.Patch(color='green', label='Good (2-5px)'),
        patches.Patch(color='orange', label='Fair (5-10px)'),
        patches.Patch(color='red', label='Poor (>10px)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("HOMOGRAPHY-BASED MATCH SUMMARY")
    print("="*60)

    valid_matches = [m for m in matches[:num_matches] if m[3] > 0]

    if valid_matches:
        validated_count = sum(1 for m in valid_matches if m[5])
        avg_error = np.mean([m[3] for m in valid_matches])
        min_error = min(m[3] for m in valid_matches)
        max_error_actual = max(m[3] for m in valid_matches)

        print(f"Matches with homography: {len(valid_matches)}/{min(num_matches, len(matches))}")
        print(f"Validated matches: {validated_count}/{len(valid_matches)}")
        print(f"Average homography error: {avg_error:.2f} pixels")
        print(f"Best homography error: {min_error:.2f} pixels")
        print(f"Worst displayed error: {max_error_actual:.2f} pixels")

        # Quality distribution
        excellent = sum(1 for m in valid_matches if m[3] <= 2.0)
        good = sum(1 for m in valid_matches if 2.0 < m[3] <= 5.0)
        fair = sum(1 for m in valid_matches if 5.0 < m[3] <= 10.0)

        print(f"\nQuality Distribution:")
        print(f"  Excellent (≤2px): {excellent}")
        print(f"  Good (2-5px): {good}")
        print(f"  Fair (5-10px): {fair}")
    else:
        print("No matches with homography data found!")

    print("="*60)


def plot_comparison_view(config: dict, num_matches: int = 5, max_error: float = 10.0, save_path: str = None):
    """
    Create a comparison view showing matches sorted by both match count and homography error.

    Args:
        config: Configuration dictionary
        num_matches: Number of top matches to display for each metric
        max_error: Maximum acceptable homography error
        save_path: Optional path to save the figure
    """
    print("Creating comparison view: Match Count vs Homography Error...")

    # Get top matches by homography error
    matches_by_homo = get_top_matches_by_homography(config['db_path'], limit=num_matches, max_error=max_error)

    # Get top matches by match count (fallback query)
    with sqlite3.connect(config['db_path']) as conn:
        cursor = conn.execute("""
            SELECT file1, file2, match_count, -1, -1, is_validated
            FROM matches 
            WHERE match_count > 0
            ORDER BY match_count DESC 
            LIMIT ?
        """, (num_matches,))
        matches_by_count = list(cursor.fetchall())

    # Create figure with two columns
    fig = plt.figure(figsize=(16, 3 * num_matches))

    fig.suptitle('Fragment Match Comparison: Match Count vs Homography Error',
                fontsize=16, fontweight='bold')

    # Plot matches by count (left column)
    for idx, (file1, file2, match_count, _, _, is_validated) in enumerate(matches_by_count):
        ax = plt.subplot(num_matches, 4, idx * 4 + 1)
        path1 = find_image_path(config['image_base_path'], file1)
        if path1:
            img1 = load_and_resize_image(path1)
            ax.imshow(img1)
        ax.set_title(f"By Count #{idx+1}\n{file1}\n{match_count} matches", fontsize=8)
        ax.axis('off')

        ax = plt.subplot(num_matches, 4, idx * 4 + 2)
        path2 = find_image_path(config['image_base_path'], file2)
        if path2:
            img2 = load_and_resize_image(path2)
            ax.imshow(img2)
        ax.set_title(f"{file2}\n{'✓' if is_validated else ''}", fontsize=8)
        ax.axis('off')

    # Plot matches by homography (right column)
    for idx, match in enumerate(matches_by_homo):
        if idx < len(matches_by_homo):
            file1, file2, match_count, mean_homo_err, std_homo_err, is_validated = match

            ax = plt.subplot(num_matches, 4, idx * 4 + 3)
            path1 = find_image_path(config['image_base_path'], file1)
            if path1:
                img1 = load_and_resize_image(path1)
                ax.imshow(img1)

            color = get_quality_color(mean_homo_err)
            ax.set_title(f"By Homo #{idx+1}\n{file1}\n{mean_homo_err:.2f}±{std_homo_err:.2f}px",
                        fontsize=8, color=color)
            ax.axis('off')

            ax = plt.subplot(num_matches, 4, idx * 4 + 4)
            path2 = find_image_path(config['image_base_path'], file2)
            if path2:
                img2 = load_and_resize_image(path2)
                ax.imshow(img2)
            ax.set_title(f"{file2}\n{match_count} matches {'✓' if is_validated else ''}",
                        fontsize=8, color=color)
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison view saved to: {save_path}")

    plt.show()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize top fragment matches sorted by homography error (geometric consistency)"
    )

    parser.add_argument(
        "--num-matches",
        type=int,
        default=10,
        help="Number of top matches to display (default: 10)"
    )

    parser.add_argument(
        "--max-error",
        type=float,
        default=10.0,
        help="Maximum homography error in pixels (default: 10.0)"
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Path to save the visualization (optional)"
    )

    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Show comparison view (match count vs homography error)"
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

        print("="*60)
        print("FRAGMENT MATCH VISUALIZATION (HOMOGRAPHY-BASED)")
        print("="*60)
        print(f"Database: {config['db_path']}")
        print(f"Image path: {config['image_base_path']}")
        print(f"Displaying top {args.num_matches} matches")
        print(f"Maximum error threshold: {args.max_error} pixels")
        print("="*60)

        if args.comparison:
            plot_comparison_view(
                config,
                num_matches=min(args.num_matches, 5),
                max_error=args.max_error,
                save_path=args.save
            )
        else:
            plot_top_matches_by_homography(
                config,
                num_matches=args.num_matches,
                max_error=args.max_error,
                save_path=args.save
            )

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease ensure your .env file contains:")
        print("  BASE_PATH=/path/to/data")
        print("  MODEL_TYPE=experiment_name")
        print("\nAlso ensure you have run the homography error computation")
        print("on your matches database before visualization.")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()