"""
Database Profile Resolver
=========================
Shared utility used by all pipeline scripts (f01–f06) and the Streamlit viewer
to resolve the active database profile from .env.

Each profile defines an isolated image set with its own output directory.
Matching is strictly within each profile — no cross-DB matching.

.env format:
    DEFAULT_DB_PROFILE = "180"
    DB_PROFILE_180_IMAGES_IN = "all_180_images/"
    DB_PROFILE_180_OUTPUT_DIR = "OUTPUT_faster_rcnn"
    DB_PROFILE_180_LABEL = "180 Images (Original)"
"""

import os
from pathlib import Path
from types import SimpleNamespace

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None


def _load_env(env_path: str = None) -> dict:
    """Load .env file into a lowercase dict."""
    if env_path is None:
        env_path = str(Path(__file__).parent / ".env")

    env = {}
    if dotenv_values and os.path.exists(env_path):
        raw = dotenv_values(env_path)
        env = {k.lower(): v for k, v in raw.items()}
    elif os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip().lower()] = v.strip().strip('"').strip("'")
    return env


def list_profiles(env_path: str = None) -> dict:
    """
    Return all database profiles defined in .env.

    Returns:
        dict: {profile_name: {'images_in': ..., 'output_dir': ..., 'label': ...}}
    """
    env = _load_env(env_path)
    profiles = {}

    # Scan for DB_PROFILE_<NAME>_IMAGES_IN keys
    prefix = "db_profile_"
    suffix_images = "_images_in"
    for key, value in env.items():
        if key.startswith(prefix) and key.endswith(suffix_images):
            name = key[len(prefix):-len(suffix_images)]
            if not name:
                continue
            output_dir = env.get(f"{prefix}{name}_output_dir", f"OUTPUT_{name}")
            label = env.get(f"{prefix}{name}_label", name)
            profiles[name] = {
                "images_in": value,
                "output_dir": output_dir,
                "label": label,
            }

    # Fallback: if no profiles defined, raise an error
    if not profiles:
        raise ValueError(
            "No database profiles found in .env. "
            "Define at least one profile using DB_PROFILE_<NAME>_IMAGES_IN "
            "and DB_PROFILE_<NAME>_OUTPUT_DIR."
        )

    return profiles


def resolve_profile(db_name: str = None, env_path: str = None) -> SimpleNamespace:
    """
    Resolve a database profile by name.

    Args:
        db_name: Profile name (e.g. "180", "354"). If None, uses DEFAULT_DB_PROFILE
                 from .env, or the first available profile.
        env_path: Optional path to .env file.

    Returns:
        SimpleNamespace with attributes:
            name, images_in, output_dir, label, base_path,
            output_base (= base_path / output_dir)
    """
    env = _load_env(env_path)
    profiles = list_profiles(env_path)

    if db_name is None:
        db_name = env.get("default_db_profile", "")
    db_name = db_name.strip().lower()

    # Try exact match
    if db_name and db_name in profiles:
        prof = profiles[db_name]
    elif profiles:
        # Fallback to first profile
        db_name = next(iter(profiles))
        prof = profiles[db_name]
    else:
        raise ValueError("No database profiles found in .env")

    base_path = env.get("base_path", "")
    output_base = os.path.join(base_path, prof["output_dir"])

    return SimpleNamespace(
        name=db_name,
        images_in=prof["images_in"],
        output_dir=prof["output_dir"],
        label=prof["label"],
        base_path=base_path,
        output_base=output_base,
    )


def add_db_argument(parser):
    """Add --db argument to an argparse.ArgumentParser."""
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database profile name (e.g. '180', '354'). "
             "Defaults to DEFAULT_DB_PROFILE in .env.",
    )
    return parser
