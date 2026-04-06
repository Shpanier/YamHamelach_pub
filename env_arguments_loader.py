import os
from types import SimpleNamespace

from dotenv import dotenv_values

from db_profile import resolve_profile


def load_env_arguments(use_clean_csv: bool = True, db_name: str = None):
    """
    Load environment arguments with optional database profile override.

    Args:
        use_clean_csv: Whether to use the cleaned CSV path (legacy).
        db_name: Database profile name (e.g. "180", "354").
                 If None, uses DEFAULT_DB_PROFILE from .env.
    """
    args = dotenv_values()

    keys = list(args.keys())
    for key in keys:
        new_key = key.lower()
        args[new_key] = args[key]
        del args[key]

    # Resolve database profile and override IMAGES_IN / OUTPUT_DIR
    profile = resolve_profile(db_name)
    args["images_in"] = profile.images_in
    args["output_dir"] = profile.output_dir
    args["db_profile"] = profile.name
    args["db_label"] = profile.label

    if args.get("pam_files_to_process") is not None:
        args["pam_files_to_process"] = args["pam_files_to_process"].split(",")
        args["pam_files_to_process"].sort()

    args["image_path"] = os.path.join(args["base_path"], args["images_in"])
    args["patches_path"] = os.path.join(args["base_path"], args["patches_dir"])

    csv = args.get("clean_sift_matches_w_tp_w_homo")
    if not use_clean_csv:
        csv = args.get("sift_matches_w_tp_w_homo", csv)

    if csv:
        args["csv_file"] = os.path.join(args["base_path"], csv)

    args = SimpleNamespace(**args)
    return args
