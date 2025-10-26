import os
from types import SimpleNamespace

from dotenv import dotenv_values


def load_env_arguments(use_clean_csv: bool = True):
    args = dotenv_values()

    keys = list(args.keys())
    for key in keys:
        new_key = key.lower()
        args[new_key] = args[key]
        del args[key]

    if args["pam_files_to_process"] is not None:
        args["pam_files_to_process"] = args["pam_files_to_process"].split(",")
        args["pam_files_to_process"].sort()

    args["image_path"] = os.path.join(args["base_path"], args["images_in"])
    args["patches_path"] = os.path.join(args["base_path"], args["patches_dir"])

    csv = (
        args["clean_sift_matches_w_tp_w_homo"]
        if use_clean_csv
        else args["sift_matches_w_tp_w_homo"]
    )

    args["csv_file"] = os.path.join(args["base_path"], csv)

    args = SimpleNamespace(**args)
    return args
