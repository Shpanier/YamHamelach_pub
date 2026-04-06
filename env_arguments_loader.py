import os
from types import SimpleNamespace

from dotenv import dotenv_values

from db_profile import resolve_profile


def load_env_arguments(db_name: str = None):
    """
    Load environment arguments with optional database profile override.

    Args:
        db_name: Database profile name (e.g. "180", "354").
                 If None, uses DEFAULT_DB_PROFILE from .env.
    """
    args = dotenv_values()

    keys = list(args.keys())
    for key in keys:
        new_key = key.lower()
        args[new_key] = args[key]
        del args[key]

    # Resolve database profile to get images_in and output_dir
    profile = resolve_profile(db_name)
    args["images_in"] = profile.images_in
    args["output_dir"] = profile.output_dir
    args["db_profile"] = profile.name
    args["db_label"] = profile.label

    args["image_path"] = os.path.join(args["base_path"], args["images_in"])
    args["patches_path"] = os.path.join(args["base_path"], args["patches_dir"])


    args = SimpleNamespace(**args)
    return args
