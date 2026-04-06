"""
Interactive Fragment Match Visualization App - With Rotation & Zoom

This Streamlit app provides comprehensive tools to explore fragment matches
with emphasis on homography error analysis, including image rotation and zoom capabilities.

Requirements:
pip install streamlit opencv-python numpy pandas plotly sqlite3 pillow

Run with: streamlit run fragment_viewer_with_rotation_zoom.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
import pickle
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict, Optional
import base64
from io import BytesIO

def get_environment():
    """Detect if running locally or on Streamlit Cloud"""
    # Check if running on Streamlit Cloud
    if 'STREAMLIT_SHARING_MODE' in os.environ or '/home/appuser' in os.path.expanduser('~'):
        return 'cloud'
    return 'local'


def get_available_profiles():
    """
    Load database profiles from db_profile module (reads .env).
    Returns dict of {name: label} for local environments, or
    a single 'cloud' profile for Streamlit Cloud.
    """
    env = get_environment()
    if env == 'cloud':
        return {'cloud': 'Cloud (Pruned)'}

    try:
        import sys as _sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in _sys.path:
            _sys.path.insert(0, script_dir)
        from db_profile import list_profiles
        profiles = list_profiles()
        return {k: v['label'] for k, v in profiles.items()}
    except Exception:
        return {'default': 'Default'}


def get_profile_paths(profile_name: str = None):
    """Get database/image paths for a given profile name."""
    env = get_environment()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if env == 'cloud':
        return {
            'db_path': os.path.join(script_dir, 'data', 'matches_pruned.db'),
            'image_base_path': os.path.join(script_dir, 'data', 'patches'),
            'complete_image_path': os.path.join(script_dir, 'data', 'bbox', 'output_bbox'),
            'pruned_db_path': os.path.join(script_dir, 'data', 'matches_pruned.db'),
            'pruned_image_base_path': os.path.join(script_dir, 'data', 'patches'),
            'pruned_complete_image_path': os.path.join(script_dir, 'data', 'bbox', 'output_bbox'),
        }

    # Local: resolve profile from .env
    try:
        import sys as _sys
        if script_dir not in _sys.path:
            _sys.path.insert(0, script_dir)
        from db_profile import resolve_profile
        profile = resolve_profile(profile_name)
        output_base = profile.output_base
    except Exception:
        output_base = '/Users/assafspanier/Dropbox/YamHamelach_data_n_model/OUTPUT_faster_rcnn'

    return {
        'db_path': os.path.join(output_base, 'matches.db'),
        'image_base_path': os.path.join(output_base, 'output_patches'),
        'complete_image_path': os.path.join(output_base, 'output_bbox'),
        'pruned_db_path': os.path.join(output_base, 'matches_pruned.db'),
        'pruned_image_base_path': os.path.join(output_base, 'output_patches_pruned'),
        'pruned_complete_image_path': os.path.join(output_base, 'output_bbox_pruned'),
    }


def get_default_paths():
    """Get appropriate default paths based on environment (legacy wrapper)."""
    return get_profile_paths()

# Page configuration
st.set_page_config(
    page_title="Fragment Match Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
    }
    .match-info {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    .good-match {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .bad-match {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_database_connection(db_path):
    """Create a cached database connection."""
    return sqlite3.connect(db_path, check_same_thread=False)


class FragmentMatchViewer:
    """Interactive viewer for fragment matching results with homography focus."""

    def __init__(self, db_path: str, image_base_path: str):
        """Initialize the viewer with database and image paths."""
        self.db_path = db_path
        self.image_base_path = image_base_path
        self.conn = None
        self.current_matches = None

    def connect_db(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return True
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            return False

    def check_database_schema(self):
        """Debug function to check database structure"""
        try:
            # Get all tables
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            st.write("Tables in database:", tables)

            # Check matches table structure
            cursor = self.conn.execute("PRAGMA table_info(matches);")
            columns = cursor.fetchall()
            st.write("Matches table columns:", columns)

            # Check if homography_errors table exists
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='homography_errors';")
            homo_table = cursor.fetchall()

            if homo_table:
                cursor = self.conn.execute("PRAGMA table_info(homography_errors);")
                homo_columns = cursor.fetchall()
                st.write("Homography_errors table columns:", homo_columns)
            else:
                st.warning("homography_errors table doesn't exist!")

        except Exception as e:
            st.error(f"Schema check error: {e}")

    def get_statistics(self) -> Dict:
        """Get overall statistics from the database."""
        if not self.conn:
            return {}

        try:
            # Use simpler conditional aggregation that works with SQLite
            query = """
            SELECT 
                COUNT(m.id) as total_matches,
                SUM(CASE WHEN h.is_valid = 1 THEN 1 ELSE 0 END) as valid_homography,
                SUM(CASE WHEN h.mean_homo_err < 5 THEN 1 ELSE 0 END) as excellent_matches,
                SUM(CASE WHEN h.mean_homo_err >= 5 AND h.mean_homo_err <= 10 THEN 1 ELSE 0 END) as good_matches,
                SUM(CASE WHEN h.mean_homo_err > 10 THEN 1 ELSE 0 END) as poor_matches,
                AVG(m.match_count) as avg_match_count,
                AVG(CASE WHEN h.is_valid = 1 THEN h.mean_homo_err ELSE NULL END) as avg_homo_error,
                MIN(CASE WHEN h.is_valid = 1 THEN h.mean_homo_err ELSE NULL END) as min_homo_error,
                MAX(CASE WHEN h.is_valid = 1 THEN h.mean_homo_err ELSE NULL END) as max_homo_error
            FROM matches m
            LEFT JOIN homography_errors h ON m.id = h.match_id
            """

            df = pd.read_sql_query(query, self.conn)
            return df.iloc[0].to_dict()
        except Exception as e:
            st.error(f"Error fetching statistics: {e}")
            return {}

    def get_matches(self, sort_by: str = 'homo_error_asc', min_matches: int = 10,
                    max_error: float = 100.0, min_error: float = 0.0,
                    limit: int = 100, validated_only: bool = False,
                    valid_homo_only: bool = False) -> pd.DataFrame:
        """Retrieve matches from database with advanced filtering and sorting."""
        if not self.conn:
            return pd.DataFrame()

        try:
            # Build query with proper syntax
            query = """
            SELECT 
                m.id,
                m.file1,
                m.file2,
                m.match_count,
                m.is_validated,
                h.mean_homo_err,
                h.std_homo_err,
                h.max_homo_err,
                h.min_homo_err,
                h.median_homo_err,
                h.is_valid as homo_valid,
                h.len_homo_err as num_inliers
            FROM matches m
            LEFT JOIN homography_errors h ON m.id = h.match_id
            WHERE m.match_count >= ?
            """

            params = [min_matches]

            if validated_only:
                query += " AND m.is_validated = 1"

            if valid_homo_only:
                query += " AND h.is_valid = 1"

            # Add homography error range filter with proper NULL handling
            query += " AND (h.mean_homo_err IS NULL OR (h.mean_homo_err >= ? AND h.mean_homo_err <= ?))"
            params.extend([min_error, max_error])

            # Add sorting
            if sort_by == 'homo_error_asc':
                query += " ORDER BY CASE WHEN h.mean_homo_err IS NULL THEN 999999 ELSE h.mean_homo_err END ASC"
            elif sort_by == 'homo_error_desc':
                query += " ORDER BY h.mean_homo_err DESC"
            elif sort_by == 'match_count_desc':
                query += " ORDER BY m.match_count DESC"
            elif sort_by == 'match_count_asc':
                query += " ORDER BY m.match_count ASC"
            elif sort_by == 'std_error_asc':
                query += " ORDER BY CASE WHEN h.std_homo_err IS NULL THEN 999999 ELSE h.std_homo_err END ASC"
            elif sort_by == 'validated_first':
                query += " ORDER BY m.is_validated DESC, CASE WHEN h.mean_homo_err IS NULL THEN 999999 ELSE h.mean_homo_err END ASC"

            query += " LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, self.conn, params=params)

            # Add quality category column
            def categorize_quality(row):
                if pd.isna(row['mean_homo_err']):
                    return 'No Homography'
                elif row['mean_homo_err'] < 5:
                    return 'Excellent'
                elif row['mean_homo_err'] < 10:
                    return 'Good'
                elif row['mean_homo_err'] < 20:
                    return 'Fair'
                else:
                    return 'Poor'

            df['quality'] = df.apply(categorize_quality, axis=1)

            return df
        except Exception as e:
            st.error(f"Error fetching matches: {e}")
            st.error(f"Query: {query if 'query' in locals() else 'Not built'}")
            st.error(f"Params: {params if 'params' in locals() else 'Not set'}")
            return pd.DataFrame()

    def get_match_details(self, match_id: int) -> Optional[bytes]:
        """Get detailed match data for a specific match."""
        if not self.conn:
            return None

        try:
            cursor = self.conn.execute(
                "SELECT matches_data FROM matches WHERE id = ?",
                (match_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            st.error(f"Error fetching match details: {e}")
            return None

    def get_all_matches_for_fragment(self, fragment_name: str, min_matches: int = 10,
                                     min_error: float = 0.0, max_error: float = 100.0,
                                     exclude_fragment: str = None) -> pd.DataFrame:
        """Get all matches for a specific fragment sorted by homography error with filters."""
        if not self.conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT 
                m.id,
                m.file1,
                m.file2,
                m.match_count,
                m.is_validated,
                h.mean_homo_err,
                h.std_homo_err,
                h.is_valid as homo_valid,
                CASE 
                    WHEN m.file1 = ? THEN m.file2
                    ELSE m.file1
                END as matched_fragment
            FROM matches m
            LEFT JOIN homography_errors h ON m.id = h.match_id
            WHERE (m.file1 = ? OR m.file2 = ?)
                AND m.match_count >= ?
                AND h.mean_homo_err IS NOT NULL
                AND h.mean_homo_err >= ?
                AND h.mean_homo_err <= ?
                AND h.mean_homo_err != -1
            """

            params = [fragment_name, fragment_name, fragment_name, min_matches, min_error, max_error]

            # Exclude the other fragment from the current pair if specified
            if exclude_fragment:
                query += """
                AND CASE 
                    WHEN m.file1 = ? THEN m.file2
                    ELSE m.file1
                END != ?
                """
                params.extend([fragment_name, exclude_fragment])

            query += " ORDER BY h.mean_homo_err ASC"

            df = pd.read_sql_query(query, self.conn, params=params)

            # Add quality category
            def categorize_quality(row):
                if pd.isna(row['mean_homo_err']):
                    return 'No Homography'
                elif row['mean_homo_err'] < 5:
                    return 'Excellent'
                elif row['mean_homo_err'] < 10:
                    return 'Good'
                elif row['mean_homo_err'] < 20:
                    return 'Fair'
                else:
                    return 'Poor'

            df['quality'] = df.apply(categorize_quality, axis=1)
            return df
        except Exception as e:
            st.error(f"Error fetching matches for fragment: {e}")
            return pd.DataFrame()

    def construct_image_path(self, filename: str) -> str:
        """Construct full image path from filename."""
        # Extract folder structure from filename pattern
        # Assuming format: XXX-YYY-ZZZ_N.jpg where folder is XXX-YYY-ZZZ
        parts = filename.split('-')
        if len(parts) >= 3:
            folder = '-'.join(parts[:3]).split('_')[0]
            return os.path.join(self.image_base_path, folder, filename)
        return os.path.join(self.image_base_path, filename)

    def load_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load and return image."""
        if not os.path.exists(filepath):
            # Try without folder structure
            filepath = os.path.join(self.image_base_path, os.path.basename(filepath))

        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle in degrees."""
        if angle == 0:
            return image

        # Get image dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions to prevent cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Rotate the image with white background
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                borderValue=(255, 255, 255))

        return rotated

    def zoom_image(self, image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom into/out of image center (zoom_factor: <1 shows more, >1 shows less but magnified)."""
        if zoom_factor == 1.0:
            return image

        height, width = image.shape[:2]

        # Calculate the size of the region to extract
        # zoom > 1 means zoom in (show less area but magnified)
        # zoom < 1 means zoom out (show more area, may need padding)
        if zoom_factor > 1.0:
            # Zoom in - extract a smaller region and resize it to original size
            new_height = int(height / zoom_factor)
            new_width = int(width / zoom_factor)

            # Calculate center crop coordinates
            y_center = height // 2
            x_center = width // 2

            y1 = max(0, y_center - new_height // 2)
            y2 = min(height, y_center + new_height // 2)
            x1 = max(0, x_center - new_width // 2)
            x2 = min(width, x_center + new_width // 2)

            # Extract the region
            cropped = image[y1:y2, x1:x2]

            # Resize back to original dimensions (magnifying the cropped area)
            zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

        else:
            # Zoom out - shrink the image and pad with white borders
            shrink_height = int(height * zoom_factor)
            shrink_width = int(width * zoom_factor)

            # Shrink the image
            shrunk = cv2.resize(image, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)

            # Create a white canvas of original size
            zoomed = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Calculate position to center the shrunk image
            y_offset = (height - shrink_height) // 2
            x_offset = (width - shrink_width) // 2

            # Place the shrunk image in the center
            zoomed[y_offset:y_offset+shrink_height, x_offset:x_offset+shrink_width] = shrunk

        return zoomed

    def zoom_image_with_center(self, image: np.ndarray, zoom_factor: float,
                               center_x: float = 0.5, center_y: float = 0.5) -> np.ndarray:
        """
        Zoom into/out of image with specified center point.
        zoom_factor: >1 for zoom in, <1 for zoom out
        center_x, center_y: normalized coordinates (0-1) for zoom center
        """
        if zoom_factor == 1.0:
            return image

        height, width = image.shape[:2]

        if zoom_factor > 1.0:
            # Zoom in - extract a smaller region centered at (center_x, center_y)
            new_height = int(height / zoom_factor)
            new_width = int(width / zoom_factor)

            # Convert normalized coordinates to pixel coordinates
            y_center = int(height * center_y)
            x_center = int(width * center_x)

            # Calculate crop boundaries
            y1 = max(0, y_center - new_height // 2)
            y2 = min(height, y1 + new_height)
            if y2 == height:
                y1 = max(0, height - new_height)

            x1 = max(0, x_center - new_width // 2)
            x2 = min(width, x1 + new_width)
            if x2 == width:
                x1 = max(0, width - new_width)

            # Extract and magnify the region
            cropped = image[y1:y2, x1:x2]
            zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

        else:
            # Zoom out - same as before
            shrink_height = int(height * zoom_factor)
            shrink_width = int(width * zoom_factor)

            shrunk = cv2.resize(image, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)

            zoomed = np.ones((height, width, 3), dtype=np.uint8) * 255

            y_offset = (height - shrink_height) // 2
            x_offset = (width - shrink_width) // 2

            zoomed[y_offset:y_offset+shrink_height, x_offset:x_offset+shrink_width] = shrunk

        return zoomed

    def create_zoom_preview(self, image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Create a small preview showing what part of the image is being zoomed."""
        if zoom_factor <= 1.0:
            return None

        height, width = image.shape[:2]

        # Create a small thumbnail of the full image
        thumb_height = 100
        thumb_width = int(width * (thumb_height / height))
        thumbnail = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)

        # Calculate the visible region in the thumbnail
        visible_height = int(thumb_height / zoom_factor)
        visible_width = int(thumb_width / zoom_factor)

        # Center coordinates
        y_center = thumb_height // 2
        x_center = thumb_width // 2

        y1 = y_center - visible_height // 2
        y2 = y_center + visible_height // 2
        x1 = x_center - visible_width // 2
        x2 = x_center + visible_width // 2

        # Draw a red rectangle on the thumbnail
        preview = thumbnail.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return preview

    def process_image(self, image: np.ndarray, rotation: float = 0, zoom: float = 1.0) -> np.ndarray:
        """Apply rotation and zoom to an image."""
        processed = image.copy()

        # Apply zoom first
        if zoom != 1.0:
            processed = self.zoom_image(processed, zoom)

        # Then apply rotation
        if rotation != 0:
            processed = self.rotate_image(processed, rotation)

        return processed

    def visualize_matches_with_lines(self, img1: np.ndarray, img2: np.ndarray,
                                     matches_data: bytes, angle1: float = 0, angle2: float = 0,
                                     zoom1: float = 1.0, zoom2: float = 1.0) -> np.ndarray:
        """Create visualization with match lines between images."""
        try:
            # Process images with rotation and zoom
            processed_img1 = self.process_image(img1, angle1, zoom1)
            processed_img2 = self.process_image(img2, angle2, zoom2)

            # Deserialize match data
            matches = pickle.loads(matches_data)

            if not matches:
                return np.hstack([processed_img1, processed_img2])

            # Create side-by-side visualization
            h1, w1 = processed_img1.shape[:2]
            h2, w2 = processed_img2.shape[:2]

            # Create output image
            height = max(h1, h2)
            width = w1 + w2
            output = np.zeros((height, width, 3), dtype=np.uint8)

            # Place images
            output[:h1, :w1] = processed_img1
            output[:h2, w1:] = processed_img2

            # Draw sample connection lines (in production, use actual keypoint data)
            num_lines_to_draw = min(50, len(matches))  # Limit lines for clarity

            # Sample random matches to visualize
            import random
            sample_matches = random.sample(matches, num_lines_to_draw) if len(matches) > num_lines_to_draw else matches

            # Draw lines (simplified - would need actual keypoint positions)
            for i, match in enumerate(sample_matches):
                # Generate pseudo-random positions for demo
                # In production, use actual keypoint positions
                color = (0, 255, 0) if i < num_lines_to_draw // 2 else (0, 255, 255)
                thickness = 1

                # This is placeholder - replace with actual keypoint coordinates
                y_offset = int((i / num_lines_to_draw) * min(h1, h2))
                cv2.line(output,
                         (w1 - 1, y_offset),
                         (w1 + 1, y_offset),
                         color, thickness)

            # Add text overlay
            cv2.putText(output, f"Total: {len(matches)} matches",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return output

        except Exception as e:
            st.error(f"Error visualizing matches: {e}")
            return np.hstack([processed_img1, processed_img2])


def main():
    """Main application function."""

    env = get_environment()

    # Title and description
    st.title("🔬 Fragment Match Explorer")
    st.markdown("### Advanced visualization and analysis of fragment matches")

    # Show environment indicator
    if env == 'cloud':
        st.info("☁️ Running on Streamlit Cloud (using pruned database)")
    else:
        st.success("💻 Running locally")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ── Database Profile Selector ──
        if env == 'local':
            profiles = get_available_profiles()
            profile_names = list(profiles.keys())
            profile_labels = list(profiles.values())

            selected_idx = st.selectbox(
                "Image Database",
                range(len(profile_names)),
                format_func=lambda i: profile_labels[i],
                help="Select which image database to explore. Matching is strictly within the selected database."
            )
            selected_profile = profile_names[selected_idx]
            default_paths = get_profile_paths(selected_profile)

            use_pruned = st.checkbox(
                "Use pruned database",
                value=False,
                help="Toggle between full database and pruned version"
            )

            if use_pruned:
                default_db = default_paths['pruned_db_path']
                default_img = default_paths['pruned_image_base_path']
                default_complete = default_paths['pruned_complete_image_path']
                st.info(f"📦 {profile_labels[selected_idx]} — pruned (smaller, faster)")
            else:
                default_db = default_paths['db_path']
                default_img = default_paths['image_base_path']
                default_complete = default_paths['complete_image_path']
                st.info(f"📚 {profile_labels[selected_idx]} — full database")
        else:
            # Cloud always uses pruned
            default_paths = get_profile_paths()
            default_db = default_paths['db_path']
            default_img = default_paths['image_base_path']
            default_complete = default_paths['complete_image_path']

        # Database and image path inputs with environment-specific defaults
        db_path = st.text_input(
            "Database Path",
            value=default_db,
            help="Path to the SQLite database with match results"
        )

        image_base_path = st.text_input(
            "Image Base Path",
            value=default_img,
            help="Base directory containing fragment images"
        )

        complete_image_path = st.text_input(
            "Complete Image Path",
            value=default_complete,
            help="Base directory containing complete images"
        )

        # Check if files exist (helpful for debugging)
        if env == 'cloud':
            if not os.path.exists(db_path):
                st.warning(f"⚠️ Database not found at: {db_path}")
                st.info("Make sure to include the data folder in your GitHub repository")
            else:
                st.success(f"✅ Database found")
        else:
            # For local, always check
            if os.path.exists(db_path):
                # Get file size
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                st.success(f"✅ Database found ({size_mb:.1f} MB)")
            else:
                st.warning(f"⚠️ Database not found at: {db_path}")

        if st.button("🔌 Connect", type="primary"):
            conn = get_database_connection(db_path)
            st.session_state.viewer = FragmentMatchViewer(db_path, image_base_path)
            st.session_state.viewer.conn = conn
            st.session_state.complete_image_path = complete_image_path
            st.session_state.connected = True
            if conn:
                st.success("✅ Connected successfully!")
                st.session_state.connected = True
            else:
                st.session_state.connected = False

        # Add debug button if connected
        if st.session_state.get('connected', False):
            if st.button("🔍 Debug Database Schema", type="secondary"):
                st.session_state.viewer.check_database_schema()

    # Check if viewer is initialized
    if 'viewer' not in st.session_state or not st.session_state.get('connected', False):
        st.info("👈 Please configure database and image paths in the sidebar")
        return

    viewer = st.session_state.viewer

    # Display statistics
    stats = viewer.get_statistics()
    if stats:
        st.header("📊 Homography Quality Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Matches",
                f"{int(stats.get('total_matches', 0)):,}"
            )

        with col2:
            excellent = int(stats.get('excellent_matches', 0))
            total = int(stats.get('total_matches', 1))
            if total > 0:
                st.metric(
                    "Excellent (<5px)",
                    f"{excellent:,}",
                    delta=f"{excellent / total * 100:.1f}%"
                )
            else:
                st.metric("Excellent (<5px)", "0")

        with col3:
            good = int(stats.get('good_matches', 0))
            if total > 0:
                st.metric(
                    "Good (5-10px)",
                    f"{good:,}",
                    delta=f"{good / total * 100:.1f}%"
                )
            else:
                st.metric("Good (5-10px)", "0")

        with col4:
            avg_error = stats.get('avg_homo_error', 0)
            if avg_error and avg_error > 0:
                st.metric(
                    "Avg Error",
                    f"{avg_error:.2f} px"
                )
            else:
                st.metric("Avg Error", "N/A")

        with col5:
            min_error = stats.get('min_homo_error', 0)
            if min_error and min_error > 0:
                st.metric(
                    "Min Error",
                    f"{min_error:.2f} px"
                )
            else:
                st.metric("Min Error", "N/A")

    st.divider()

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📊 Table View & Images",
        "📈 Analytics",
        "📉 Error Distribution"
    ])

    with tab1:
        # Filter controls
        st.header("Filter & Sort Matches")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            sort_by = st.selectbox(
                "Sort By",
                options=[
                    ('homo_error_asc', '📈 Homography Error (Best First)'),
                    ('homo_error_desc', '📉 Homography Error (Worst First)'),
                    ('std_error_asc', '📊 Std Deviation (Most Stable)'),
                    ('match_count_desc', '🔢 Match Count (High to Low)'),
                    ('match_count_asc', '🔢 Match Count (Low to High)'),
                    ('validated_first', '✅ Validated First')
                ],
                format_func=lambda x: x[1],
                index=0  # Default to homography error ascending
            )[0]

        with col2:
            # Homography error range slider
            error_range = st.slider(
                "Homography Error Range (px)",
                min_value=0.0,
                max_value=300.0,
                value=(0.0, 200.0),
                step=0.5,
                help="Filter matches by homography error range"
            )
            min_error, max_error = error_range

        with col3:
            min_matches = st.number_input(
                "Min Match Count",
                min_value=1,
                max_value=500,
                value=10,
                step=5,
                help="Minimum number of SIFT matches"
            )

        with col4:
            limit = st.number_input(
                "Result Limit",
                min_value=10,
                max_value=2000,
                value=1000,
                step=10
            )

        # Additional filters
        col1, col2 = st.columns(2)
        with col1:
            validated_only = st.checkbox("Show validated matches only")
        with col2:
            valid_homo_only = st.checkbox("Show valid homography only")

        # Load matches
        if st.button("🔄 Apply Filters", type="primary"):
            with st.spinner("Loading matches..."):
                matches_df = viewer.get_matches(
                    sort_by=sort_by,
                    min_matches=min_matches,
                    min_error=min_error,
                    max_error=max_error,
                    limit=limit,
                    validated_only=validated_only,
                    valid_homo_only=valid_homo_only
                )
                st.session_state.matches_df = matches_df

        # Display matches if loaded
        if 'matches_df' in st.session_state and not st.session_state.matches_df.empty:
            matches_df = st.session_state.matches_df

            st.header(f"📋 Found {len(matches_df)} Matches")

            # Display quality distribution
            if 'quality' in matches_df.columns:
                quality_counts = matches_df['quality'].value_counts()
                st.markdown("**Quality Distribution:**")
                quality_cols = st.columns(len(quality_counts))
                for i, (quality, count) in enumerate(quality_counts.items()):
                    with quality_cols[i]:
                        color = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Poor': '🔴', 'No Homography': '⚫'}.get(
                            quality, '⚪')
                        st.metric(f"{color} {quality}", count)

            # Interactive table with color coding
            def highlight_quality(row):
                if pd.isna(row['mean_homo_err']):
                    return ['background-color: #gray'] * len(row)
                elif row['mean_homo_err'] < 5:
                    return ['background-color: #d4f4dd'] * len(row)
                elif row['mean_homo_err'] < 10:
                    return ['background-color: #fff3cd'] * len(row)
                elif row['mean_homo_err'] < 20:
                    return ['background-color: #ffe5cc'] * len(row)
                else:
                    return ['background-color: #ffcccc'] * len(row)

            display_df = matches_df[['file1', 'file2', 'match_count', 'mean_homo_err',
                                     'std_homo_err', 'num_inliers', 'quality', 'is_validated']]

            styled_df = display_df.style.apply(highlight_quality, axis=1)
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # Select a match to visualize
            st.subheader("Select a Match to Visualize")
            selected_idx = st.selectbox(
                "Choose match",
                options=matches_df.index,
                format_func=lambda
                    x: f"Rank {x + 1}: {matches_df.loc[x, 'file1'][:20]}... ↔ {matches_df.loc[x, 'file2'][:20]}... (Error: {matches_df.loc[x, 'mean_homo_err']:.2f}px)" if pd.notna(
                    matches_df.loc[x, 'mean_homo_err']) else f"Rank {x + 1}: No homography"
            )

            if selected_idx is not None:
                selected_match = matches_df.loc[selected_idx]

                # Display detailed match metrics
                st.markdown("### 📐 Homography Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if pd.notna(selected_match['mean_homo_err']):
                        st.metric(
                            "Mean Error",
                            f"{selected_match['mean_homo_err']:.2f} px",
                            delta="Good" if selected_match['mean_homo_err'] < 10 else "Poor"
                        )
                    else:
                        st.metric("Mean Error", "N/A")

                with col2:
                    if pd.notna(selected_match['std_homo_err']):
                        st.metric(
                            "Std Deviation",
                            f"{selected_match['std_homo_err']:.2f} px"
                        )
                    else:
                        st.metric("Std Deviation", "N/A")

                with col3:
                    if pd.notna(selected_match['min_homo_err']) and pd.notna(selected_match['max_homo_err']):
                        st.metric(
                            "Error Range",
                            f"{selected_match['min_homo_err']:.1f} - {selected_match['max_homo_err']:.1f} px"
                        )
                    else:
                        st.metric("Error Range", "N/A")

                with col4:
                    st.metric(
                        "Match Count",
                        f"{selected_match['match_count']}",
                        delta=f"{selected_match.get('num_inliers', 'N/A')} inliers"
                    )

                # Image comparison directly below selection
                st.markdown("### 🖼️ Fragment Comparison")

                if selected_match['is_validated']:
                    st.success("✅ This is a validated match")

                # Quality indicator
                quality = selected_match.get('quality', 'Unknown')
                quality_color = {'Excellent': '#10b981', 'Good': '#fbbf24', 'Fair': '#fb923c', 'Poor': '#ef4444'}.get(
                    quality, '#gray')
                st.markdown(
                    f"<div style='background-color: {quality_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 20px;'>Quality: {quality}</div>",
                    unsafe_allow_html=True)

                # Load and display images with rotation controls
                path1 = viewer.construct_image_path(selected_match['file1'])
                path2 = viewer.construct_image_path(selected_match['file2'])

                img1 = viewer.load_image(path1)
                img2 = viewer.load_image(path2)

                if img1 is not None and img2 is not None:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Display Fragment 1
                        st.markdown("**Fragment 1 Controls:**")

                        # Create two columns for rotation and zoom
                        ctrl_col1, ctrl_col2 = st.columns(2)

                        with ctrl_col1:
                            rotation1 = st.slider(
                                "🔄 Rotate",
                                min_value=0,
                                max_value=360,
                                value=0,
                                step=1,
                                key=f"rotation1_{selected_idx}",
                                help="Rotate the image by degrees (0-360)"
                            )

                        with ctrl_col2:
                            zoom1 = st.slider(
                                "🔍 Zoom",
                                min_value=0.25,
                                max_value=4.0,
                                value=1.0,
                                step=0.25,
                                key=f"zoom1_{selected_idx}",
                                help="Zoom: <1 zooms out (shows smaller), >1 zooms in (magnifies center)"
                            )

                        # Apply transformations
                        display_img1 = viewer.process_image(img1, rotation1, zoom1)

                        # Show zoom preview if zoomed in
                        if zoom1 > 1.0:
                            preview1 = viewer.create_zoom_preview(img1, zoom1)
                            if preview1 is not None:
                                st.caption("🔍 Zoom preview (red box shows visible area):")
                                st.image(preview1, width=150)

                        st.image(display_img1, caption=f"Fragment 1: {selected_match['file1']}", use_container_width=True)

                        # Reset button for both controls
                        if rotation1 != 0 or zoom1 != 1.0:
                            if st.button("↺ Reset All", key=f"reset1_{selected_idx}"):
                                st.rerun()

                    with col2:
                        # Display Fragment 2
                        st.markdown("**Fragment 2 Controls:**")

                        # Create two columns for rotation and zoom
                        ctrl_col1, ctrl_col2 = st.columns(2)

                        with ctrl_col1:
                            rotation2 = st.slider(
                                "🔄 Rotate",
                                min_value=0,
                                max_value=360,
                                value=0,
                                step=1,
                                key=f"rotation2_{selected_idx}",
                                help="Rotate the image by degrees (0-360)"
                            )

                        with ctrl_col2:
                            zoom2 = st.slider(
                                "🔍 Zoom",
                                min_value=0.25,
                                max_value=4.0,
                                value=1.0,
                                step=0.25,
                                key=f"zoom2_{selected_idx}",
                                help="Zoom: <1 zooms out (shows smaller), >1 zooms in (magnifies center)"
                            )

                        # Apply transformations
                        display_img2 = viewer.process_image(img2, rotation2, zoom2)

                        # Show zoom preview if zoomed in
                        if zoom2 > 1.0:
                            preview2 = viewer.create_zoom_preview(img2, zoom2)
                            if preview2 is not None:
                                st.caption("🔍 Zoom preview (red box shows visible area):")
                                st.image(preview2, width=150)

                        st.image(display_img2, caption=f"Fragment 2: {selected_match['file2']}", use_container_width=True)

                        # Reset button for both controls
                        if rotation2 != 0 or zoom2 != 1.0:
                            if st.button("↺ Reset All", key=f"reset2_{selected_idx}"):
                                st.rerun()

                    # Show combined visualization with rotations and zoom applied
                    if st.checkbox("Show match visualization with connection lines", value=True):
                        match_data = viewer.get_match_details(selected_match['id'])
                        if match_data:
                            # Use current rotation and zoom values directly
                            combined = viewer.visualize_matches_with_lines(
                                img1, img2, match_data,
                                rotation1, rotation2,
                                zoom1, zoom2
                            )
                            st.image(combined, caption="Match Visualization (with transformations applied)", use_container_width=True)
                else:
                    st.warning("⚠️ Could not load one or both images. Please check the image paths.")
                    # Debug information
                    with st.expander("Debug Info"):
                        st.write(f"Image 1 path attempted: {path1}")
                        st.write(f"Image 1 exists: {os.path.exists(path1)}")
                        st.write(f"Image 2 path attempted: {path2}")
                        st.write(f"Image 2 exists: {os.path.exists(path2)}")
                        st.write(f"Base image path: {viewer.image_base_path}")
                        # List files in the base path
                        if os.path.exists(viewer.image_base_path):
                            files = os.listdir(viewer.image_base_path)
                            st.write(f"Files/folders in base path (first 10): {files[:10]}")

                # Button to see all matches for the selected fragments
                st.divider()

                # Create columns for better layout
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    # Button to show complete images
                    if st.button("🖼️ Show complete images",
                                 type="secondary",
                                 use_container_width=True):
                        st.session_state.show_complete_images = True
                        st.session_state.complete_file1 = selected_match['file1']
                        st.session_state.complete_file2 = selected_match['file2']

                with col2:
                    # Store selected match info in session state for the button
                    if st.button("🔍 See all matches of these fragments",
                                 type="primary",
                                 use_container_width=True,
                                 disabled=False):
                        st.session_state.show_all_matches = True
                        st.session_state.selected_file1 = selected_match['file1']
                        st.session_state.selected_file2 = selected_match['file2']
                        st.session_state.min_matches_for_all = min_matches
                        st.session_state.min_error_for_all = min_error
                        st.session_state.max_error_for_all = max_error

                # Display complete images if button was clicked (with rotation controls)
                if st.session_state.get('show_complete_images', False):
                    st.divider()
                    st.subheader("📸 Complete Images Containing These Fragments")

                    # Extract base filenames without fragment numbers
                    file1 = st.session_state.get('complete_file1', '')
                    file2 = st.session_state.get('complete_file2', '')

                    # Parse filenames to get base image names
                    # Example: M43166-1-E_530.jpg -> M43166-1-E.jpg
                    base_file1 = file1.rsplit('_', 1)[0] + '.jpg' if '_' in file1 else file1
                    base_file2 = file2.rsplit('_', 1)[0] + '.jpg' if '_' in file2 else file2

                    # Get complete image path from session state
                    complete_path = st.session_state.get('complete_image_path', '')

                    if complete_path:
                        # Construct full paths for complete images
                        complete_path1 = os.path.join(complete_path, base_file1)
                        complete_path2 = os.path.join(complete_path, base_file2)

                        # Load complete images
                        complete_img1 = viewer.load_image(complete_path1)
                        complete_img2 = viewer.load_image(complete_path2)

                        col1, col2 = st.columns(2)

                        with col1:
                            if complete_img1 is not None:
                                st.markdown("**Complete Image 1 Controls:**")

                                # Create two columns for rotation and zoom
                                ctrl_col1, ctrl_col2 = st.columns(2)

                                with ctrl_col1:
                                    complete_rotation1 = st.slider(
                                        "🔄 Rotate",
                                        min_value=0,
                                        max_value=360,
                                        value=0,
                                        step=1,
                                        key="complete_rotation1",
                                        help="Rotate the complete image by degrees (0-360)"
                                    )

                                with ctrl_col2:
                                    complete_zoom1 = st.slider(
                                        "🔍 Zoom",
                                        min_value=0.25,
                                        max_value=4.0,
                                        value=1.0,
                                        step=0.25,
                                        key="complete_zoom1",
                                        help="Zoom level (0.25x to 4x)"
                                    )

                                # Apply transformations
                                display_complete1 = viewer.process_image(complete_img1, complete_rotation1, complete_zoom1)

                                st.image(display_complete1,
                                         caption=f"Complete Image: {base_file1}",
                                         use_container_width=True)
                                # Show which fragment this is
                                fragment_num1 = file1.rsplit('_', 1)[1].replace('.jpg', '') if '_' in file1 else 'N/A'
                                st.caption(f"Contains fragment #{fragment_num1}")
                            else:
                                st.warning(f"⚠️ Could not load complete image: {base_file1}")
                                st.caption(f"Attempted path: {complete_path1}")

                        with col2:
                            if complete_img2 is not None:
                                st.markdown("**Complete Image 2 Controls:**")

                                # Create two columns for rotation and zoom
                                ctrl_col1, ctrl_col2 = st.columns(2)

                                with ctrl_col1:
                                    complete_rotation2 = st.slider(
                                        "🔄 Rotate",
                                        min_value=0,
                                        max_value=360,
                                        value=0,
                                        step=1,
                                        key="complete_rotation2",
                                        help="Rotate the complete image by degrees (0-360)"
                                    )

                                with ctrl_col2:
                                    complete_zoom2 = st.slider(
                                        "🔍 Zoom",
                                        min_value=0.25,
                                        max_value=4.0,
                                        value=1.0,
                                        step=0.25,
                                        key="complete_zoom2",
                                        help="Zoom level (0.25x to 4x)"
                                    )

                                # Apply transformations
                                display_complete2 = viewer.process_image(complete_img2, complete_rotation2, complete_zoom2)

                                st.image(display_complete2,
                                         caption=f"Complete Image: {base_file2}",
                                         use_container_width=True)
                                # Show which fragment this is
                                fragment_num2 = file2.rsplit('_', 1)[1].replace('.jpg', '') if '_' in file2 else 'N/A'
                                st.caption(f"Contains fragment #{fragment_num2}")
                            else:
                                st.warning(f"⚠️ Could not load complete image: {base_file2}")
                                st.caption(f"Attempted path: {complete_path2}")

                        # Add clear button for complete images
                        if st.button("❌ Hide complete images", use_container_width=False):
                            st.session_state.show_complete_images = False
                            st.rerun()
                    else:
                        st.warning("⚠️ Complete image path not configured. Please set it in the sidebar.")

                # Display all matches if button was clicked (with rotation for thumbnails)
                if st.session_state.get('show_all_matches', False):
                    st.divider()
                    st.subheader("📋 Other Matches for Selected Fragments")

                    # Get the stored parameters
                    file1 = st.session_state.get('selected_file1', '')
                    file2 = st.session_state.get('selected_file2', '')
                    min_match_count = st.session_state.get('min_matches_for_all', 10)
                    min_err = st.session_state.get('min_error_for_all', 0.0)
                    max_err = st.session_state.get('max_error_for_all', 100.0)

                    # Display filter criteria being used
                    st.info(
                        f"Showing other matches with: **{min_match_count}+ matches** | **Error range: {min_err:.1f} - {max_err:.1f} px** | Excluding current pair")

                    # Create two columns for the two fragments
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### Other matches for: {file1}")

                        # Get all matches for first fragment, excluding file2
                        matches_file1 = viewer.get_all_matches_for_fragment(
                            file1, min_match_count, min_err, max_err, exclude_fragment=file2
                        )

                        if not matches_file1.empty:
                            st.markdown(f"Found **{len(matches_file1)}** other matches within criteria")

                            # Display quality distribution
                            quality_counts = matches_file1['quality'].value_counts()
                            quality_text = " | ".join([f"{q}: {c}" for q, c in quality_counts.items()])
                            st.caption(f"Quality: {quality_text}")

                            # Display matches with thumbnails
                            for idx, row in matches_file1.iterrows():
                                error_text = f"{row['mean_homo_err']:.2f}px"
                                quality_color = {
                                    'Excellent': '🟢',
                                    'Good': '🟡',
                                    'Fair': '🟠',
                                    'Poor': '🔴'
                                }.get(row['quality'], '⚪')

                                validated_badge = "✅" if row['is_validated'] else ""

                                # Create container for each match
                                with st.container():
                                    # Match info
                                    st.markdown(f"""
                                    <div style='background-color: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                                        <strong>{quality_color} {row['matched_fragment']}</strong><br>
                                        Error: {error_text} | Matches: {row['match_count']} {validated_badge}
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Load and display thumbnail with rotation
                                    matched_path = viewer.construct_image_path(row['matched_fragment'])
                                    matched_img = viewer.load_image(matched_path)

                                    if matched_img is not None:
                                        # Create expandable image with rotation and zoom controls
                                        with st.expander(f"View image", expanded=False):
                                            # Create two columns for controls
                                            ctrl_col1, ctrl_col2 = st.columns(2)

                                            with ctrl_col1:
                                                thumb_rotation = st.slider(
                                                    f"🔄 Rotate",
                                                    min_value=0,
                                                    max_value=360,
                                                    value=0,
                                                    step=15,
                                                    key=f"thumb1_rot_{idx}",
                                                )

                                            with ctrl_col2:
                                                thumb_zoom = st.slider(
                                                    f"🔍 Zoom",
                                                    min_value=0.5,
                                                    max_value=3.0,
                                                    value=1.0,
                                                    step=0.25,
                                                    key=f"thumb1_zoom_{idx}",
                                                )

                                            # Apply transformations
                                            display_thumb = viewer.process_image(matched_img, thumb_rotation, thumb_zoom)
                                            st.image(display_thumb, caption=row['matched_fragment'],
                                                     use_container_width=True)

                                    st.markdown("---")
                        else:
                            st.info(
                                f"No other matches found within the specified criteria (Error: {min_err:.1f}-{max_err:.1f}px, Min matches: {min_match_count})")

                    with col2:
                        st.markdown(f"### Other matches for: {file2}")

                        # Get all matches for second fragment, excluding file1
                        matches_file2 = viewer.get_all_matches_for_fragment(
                            file2, min_match_count, min_err, max_err, exclude_fragment=file1
                        )

                        if not matches_file2.empty:
                            st.markdown(f"Found **{len(matches_file2)}** other matches within criteria")

                            # Display quality distribution
                            quality_counts = matches_file2['quality'].value_counts()
                            quality_text = " | ".join([f"{q}: {c}" for q, c in quality_counts.items()])
                            st.caption(f"Quality: {quality_text}")

                            # Display matches with thumbnails
                            for idx, row in matches_file2.iterrows():
                                error_text = f"{row['mean_homo_err']:.2f}px"
                                quality_color = {
                                    'Excellent': '🟢',
                                    'Good': '🟡',
                                    'Fair': '🟠',
                                    'Poor': '🔴'
                                }.get(row['quality'], '⚪')

                                validated_badge = "✅" if row['is_validated'] else ""

                                # Create container for each match
                                with st.container():
                                    # Match info
                                    st.markdown(f"""
                                    <div style='background-color: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                                        <strong>{quality_color} {row['matched_fragment']}</strong><br>
                                        Error: {error_text} | Matches: {row['match_count']} {validated_badge}
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Load and display thumbnail with rotation
                                    matched_path = viewer.construct_image_path(row['matched_fragment'])
                                    matched_img = viewer.load_image(matched_path)

                                    if matched_img is not None:
                                        # Create expandable image with rotation and zoom controls
                                        with st.expander(f"View image", expanded=False):
                                            # Create two columns for controls
                                            ctrl_col1, ctrl_col2 = st.columns(2)

                                            with ctrl_col1:
                                                thumb_rotation = st.slider(
                                                    f"🔄 Rotate",
                                                    min_value=0,
                                                    max_value=360,
                                                    value=0,
                                                    step=15,
                                                    key=f"thumb2_rot_{idx}",
                                                )

                                            with ctrl_col2:
                                                thumb_zoom = st.slider(
                                                    f"🔍 Zoom",
                                                    min_value=0.5,
                                                    max_value=3.0,
                                                    value=1.0,
                                                    step=0.25,
                                                    key=f"thumb2_zoom_{idx}",
                                                )

                                            # Apply transformations
                                            display_thumb = viewer.process_image(matched_img, thumb_rotation, thumb_zoom)
                                            st.image(display_thumb, caption=row['matched_fragment'],
                                                     use_container_width=True)

                                    st.markdown("---")
                        else:
                            st.info(
                                f"No other matches found within the specified criteria (Error: {min_err:.1f}-{max_err:.1f}px, Min matches: {min_match_count})")

                    # Add a clear button to hide the results
                    st.divider()
                    if st.button("❌ Clear all matches view", use_container_width=False):
                        st.session_state.show_all_matches = False
                        st.rerun()

    with tab2:
        # Analytics view
        if 'matches_df' in st.session_state and not st.session_state.matches_df.empty:
            matches_df = st.session_state.matches_df

            st.subheader("Match Quality Analysis")

            # Scatter plot: Match count vs Homography error
            valid_errors = matches_df[matches_df['mean_homo_err'].notna()].copy()

            if not valid_errors.empty:
                # Main scatter plot
                fig = px.scatter(
                    valid_errors,
                    x='match_count',
                    y='mean_homo_err',
                    color='quality',
                    size='num_inliers',
                    title="Match Count vs Homography Error",
                    labels={
                        'match_count': 'Number of SIFT Matches',
                        'mean_homo_err': 'Mean Homography Error (pixels)',
                        'quality': 'Quality Category',
                        'num_inliers': 'Number of Inliers'
                    },
                    hover_data=['file1', 'file2', 'std_homo_err'],
                    color_discrete_map={
                        'Excellent': '#10b981',
                        'Good': '#fbbf24',
                        'Fair': '#fb923c',
                        'Poor': '#ef4444'
                    }
                )

                # Add quality threshold lines
                fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="Excellent Threshold")
                fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Good Threshold")
                fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Fair Threshold")

                st.plotly_chart(fig, use_container_width=True)

                # Correlation analysis
                col1, col2 = st.columns(2)

                with col1:
                    # Box plot by quality category
                    fig2 = px.box(
                        valid_errors,
                        x='quality',
                        y='mean_homo_err',
                        title="Error Distribution by Quality",
                        category_orders={'quality': ['Excellent', 'Good', 'Fair', 'Poor']}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with col2:
                    # Correlation heatmap
                    corr_cols = ['match_count', 'mean_homo_err', 'std_homo_err', 'num_inliers']
                    corr_data = valid_errors[corr_cols].dropna()
                    if not corr_data.empty:
                        correlation = corr_data.corr()
                        fig3 = px.imshow(
                            correlation,
                            text_auto=True,
                            aspect="auto",
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Load matches in the Table View first to see analytics")

    with tab3:
        # Error distribution analysis
        if 'matches_df' in st.session_state and not st.session_state.matches_df.empty:
            matches_df = st.session_state.matches_df

            st.subheader("Homography Error Distribution")

            valid_errors = matches_df[matches_df['mean_homo_err'].notna()]

            if not valid_errors.empty:
                # Histogram
                fig1 = px.histogram(
                    valid_errors,
                    x='mean_homo_err',
                    nbins=50,
                    title="Distribution of Mean Homography Errors",
                    labels={'mean_homo_err': 'Mean Error (pixels)', 'count': 'Frequency'},
                    color_discrete_sequence=['#7c3aed']
                )
                fig1.add_vline(x=5, line_dash="dash", line_color="green", annotation_text="Excellent")
                fig1.add_vline(x=10, line_dash="dash", line_color="orange", annotation_text="Good")
                fig1.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="Fair")
                st.plotly_chart(fig1, use_container_width=True)

                # Summary statistics
                st.subheader("Summary Statistics")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Homography Error Statistics:**")
                    error_stats = valid_errors['mean_homo_err'].describe()
                    stats_df = pd.DataFrame({
                        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max'],
                        'Value (px)': [
                            f"{error_stats['count']:.0f}",
                            f"{error_stats['mean']:.2f}",
                            f"{error_stats['std']:.2f}",
                            f"{error_stats['min']:.2f}",
                            f"{error_stats['25%']:.2f}",
                            f"{error_stats['50%']:.2f}",
                            f"{error_stats['75%']:.2f}",
                            f"{error_stats['max']:.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)

                with col2:
                    st.markdown("**Match Count vs Error Correlation:**")
                    correlation = valid_errors[['match_count', 'mean_homo_err']].corr().iloc[0, 1]
                    st.metric("Correlation Coefficient", f"{correlation:.3f}")

                    if correlation < -0.3:
                        st.info("Strong negative correlation: More matches tend to have lower errors")
                    elif correlation < -0.1:
                        st.info("Weak negative correlation: Slight tendency for more matches to have lower errors")
                    elif correlation < 0.1:
                        st.info("No significant correlation between match count and error")
                    elif correlation < 0.3:
                        st.info("Weak positive correlation: Slight tendency for more matches to have higher errors")
                    else:
                        st.info("Strong positive correlation: More matches tend to have higher errors")
        else:
            st.info("Load matches in the Table View first to see error distribution")


if __name__ == "__main__":
    main()