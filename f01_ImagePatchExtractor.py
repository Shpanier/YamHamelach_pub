"""
ImagePatchExtractor Module

This module provides functionality for detecting and extracting patches from images using
a YOLO (You Only Look Once) object detection model. It can process images to identify
patch-like objects, extract them as separate images, and save both the detection results
and individual patch crops.

The main class PatchFinder uses a trained YOLO model to:
1. Detect bounding boxes around patch objects in images
2. Extract individual patches from the detected regions
3. Generate visualization images with bounding boxes and tags
4. Save extracted patches as separate image files
5. Store patch metadata including coordinates and filenames

Dependencies:
    - cv2 (OpenCV): Image processing operations
    - numpy: Numerical computations and array operations
    - matplotlib: Image visualization and saving
    - ultralytics: YOLO model implementation
    - tqdm: Progress bar for batch processing

Usage:
    python patch_finder.py --im_path /path/to/images --patches_dir /path/to/output
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

from env_arguments_loader import load_env_arguments


def center(box, dtype=None):
    """
    Calculate the center point of a bounding box.

    Args:
        box (list or tuple): Bounding box coordinates in format [left, top, right, bottom]
        dtype (numpy.dtype, optional): Data type for the returned center coordinates.
                                     If None, uses default float type.

    Returns:
        numpy.ndarray: Center coordinates as [center_x, center_y]

    Example:
        >>> box = [10, 20, 50, 60]
        >>> center(box)
        array([30., 40.])
        >>> center(box, dtype=np.int32)
        array([30, 40], dtype=int32)
    """
    left, top, right, bottom = box
    center = np.array([(right + left) / 2, (bottom + top) / 2])
    if dtype is not None:
        center = center.astype(dtype)
    return center


def center_radius(box):
    """
    Calculate the distance from origin (0,0) to the center of a bounding box.

    This function is typically used as a sorting key to order boxes by their
    distance from the top-left corner of the image.

    Args:
        box (list or tuple): Bounding box coordinates in format [left, top, right, bottom]

    Returns:
        float: Euclidean distance from origin to box center

    Example:
        >>> box = [3, 4, 7, 8]  # center at (5, 6)
        >>> center_radius(box)
        7.810249675906654  # sqrt(5^2 + 6^2)
    """
    return np.linalg.norm(center(box))


class ImagePatchExtractor:
    """
    A class for detecting and extracting patches from images using YOLO object detection.

    This class provides comprehensive functionality for:
    - Loading and processing images
    - Detecting patch objects using a trained YOLO model
    - Extracting individual patch regions
    - Generating tagged visualization images
    - Saving patches and metadata

    Attributes:
        image (numpy.ndarray): Currently loaded image
        _model (YOLO): Loaded YOLO model for object detection
        _img_filename (str): Path to the currently loaded image file
        _extracted_boxes (list): List of detected bounding boxes
        _id_map (numpy.ndarray): Grid-based ID mapping for patch tagging
        _tags (list): List of unique tags for detected patches
        _patch_info (dict): Metadata dictionary for extracted patches
    """

    def __init__(self, cp):
        """
        Initialize PatchFinder with a YOLO model checkpoint.

        Args:
            cp (str): Path to the YOLO model checkpoint file (.pt format)

        Example:
            >>> pf = PatchFinder("models/yolov8_patches.pt")
        """
        self.image = None
        self._model = YOLO(cp)
        self._img_filename = None
        self._extracted_boxes = None
        self._id_map = None
        self._tags = []
        self._patch_info = {}

    @property
    def id_map(self):
        """
        Generate or return a spatial ID mapping grid for the current image.

        Creates a 32x32 grid of unique IDs that is resized to match the current
        image dimensions. This grid is used to assign unique tags to detected
        patches based on their center coordinates.

        Returns:
            numpy.ndarray: 2D array of IDs matching image dimensions, where each
                          pixel contains a unique identifier (0-1023 for 32x32 grid)

        Note:
            The ID map is regenerated if the image dimensions change or if it
            doesn't exist yet.
        """
        if (
            self._id_map is None
            or self._id_map.shape[:2] != self.image.shape[:2]
        ):
            cols, rows = (32, 32)
            ids = np.arange(rows * cols)
            self._id_map = ids.reshape((cols, rows))
            self._id_map = cv2.resize(
                self._id_map,
                self.image.shape[:2][::-1],
                interpolation=cv2.INTER_NEAREST,
            )
        return self._id_map

    def predict_bounding_box(
        self,
        model_input_shape: tuple[int, ...] = (640, 640),
        patch_cls_name: str = "patch",
    ):
        """
        Detect patch objects in the current image using YOLO model.

        Resizes the image to the model's expected input size, runs inference,
        and extracts bounding boxes for objects classified as patches. The
        detected boxes are scaled back to original image coordinates and
        sorted by distance from the origin.

        Args:
            model_input_shape (tuple): Target size for model input as (height, width).
                                     Default is (640, 640) for YOLOv8.
            patch_cls_name (str): Class name to filter detections. Only objects
                                with this class name will be extracted.

        Side Effects:
            - Updates self._extracted_boxes with detected bounding boxes
            - Updates self._images with cropped patch images
            - Boxes are sorted by distance from origin (top-left corner)

        Example:
            >>> pf.load_image("sample.jpg")
            >>> pf.predict_bounding_box()
            >>> len(pf._extracted_boxes)  # Number of detected patches
            5
        """
        img = cv2.resize(self.image, model_input_shape)
        results = self._model.predict([img], verbose=False)
        names = results[0].names
        boxes = results[0].boxes
        cls = boxes.cls.numpy()

        self._images = []
        extracted_boxes = []
        for i, box in enumerate(boxes):
            if names[cls[i]] != patch_cls_name:
                continue

            left, top, right, bottom = box.xyxy.numpy()[0]
            im_shape = self.image.shape
            left = int(left / model_input_shape[1] * im_shape[1])
            right = int(right / model_input_shape[1] * im_shape[1])
            top = int(top / model_input_shape[0] * im_shape[0])
            bottom = int(bottom / model_input_shape[0] * im_shape[0])
            extracted_boxes.append([left, top, right, bottom])

        self._extracted_boxes = sorted(extracted_boxes, key=center_radius)

        for left, top, right, bottom in self._extracted_boxes:
            self._images.append(self.image[top:bottom, left:right])

    def load_image(self, img_filename):
        """
        Load an image file and reset patch-related data.

        Args:
            img_filename (str): Path to the image file to load

        Side Effects:
            - Sets self.image to the loaded image array
            - Resets internal state (tags, patch_info)
            - Stores the filename for later reference

        Raises:
            cv2.error: If the image file cannot be read

        Example:
            >>> pf.load_image("/path/to/image.jpg")
            >>> pf.image.shape
            (1080, 1920, 3)
        """
        self._img_filename = img_filename
        self.image = cv2.imread(img_filename)
        self._tags = []
        self._patch_info = {}

    @property
    def tags(self):
        """
        Generate unique tags for detected patches based on their spatial location.

        Uses the ID map to assign each patch a unique identifier based on its
        center coordinates. If multiple patches have the same base tag, decimal
        suffixes are added to maintain uniqueness.

        Returns:
            list: List of unique numeric tags for each detected patch

        Note:
            Tags are generated lazily and cached. The list corresponds to
            patches in self._extracted_boxes order.

        Example:
            >>> pf.tags
            [123, 456, 789.1, 789.2]  # Two patches at similar locations
        """
        if self._tags == []:
            for i, box in enumerate(self._extracted_boxes):
                center_x, center_y = center(box, dtype=np.uint16)
                tag = self.id_map[center_y][center_x]
                if tag in self._tags:
                    count = len([t for t in self._tags if int(t) == tag])
                    tag = tag + count / 10
                self._tags.append(tag)
        return self._tags

    def _generate_image_with_detection(self):
        """
        Create a visualization image with bounding boxes and tags overlaid.

        Generates a copy of the original image with colored rectangles around
        detected patches and numeric tags at their centers. Each patch gets
        a random color for visual distinction.

        Returns:
            numpy.ndarray: Image array with detection visualizations overlaid

        Note:
            This is an internal method used by show_image() and save_image().
            Colors are randomly generated for each patch.
        """
        im = np.array(self.image)
        for i, box in enumerate(self._extracted_boxes):
            (left, top, right, bottom) = box
            c = np.random.randint(0, 125, 3)
            im = cv2.rectangle(
                im, (left, top), (right, bottom), c.tolist(), 10
            )
            tag = self.tags[i]
            im = cv2.putText(
                im,
                str(tag),
                (int((right + left) / 2), int((bottom + top) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                c.tolist(),
                5,
            )
        return im

    def show_image(self):
        """
        Display the current image with detection results using matplotlib.

        Shows the image with bounding boxes drawn around detected patches
        and numeric tags displayed at patch centers. Useful for interactive
        visualization and verification of detection results.

        Example:
            >>> pf.load_image("test.jpg")
            >>> pf.predict_bounding_box()
            >>> pf.show_image()  # Opens matplotlib window
        """
        img = self._generate_image_with_detection()
        plt.imshow(img)
        plt.show()

    def save_image(self, fn):
        """
        Save the detection visualization image to a file.

        Args:
            fn (str or Path): Output filename/path for the visualization image.
                            Parent directories will be created if they don't exist.

        Example:
            >>> pf.save_image("output/detections.jpg")
        """
        os.makedirs(fn.parent, exist_ok=True)
        img = self._generate_image_with_detection()
        plt.imsave(fn, img)

    def save_patches(self, path):
        """
        Extract and save individual patch images to separate files.

        Crops each detected patch from the original image and saves it as a
        separate JPEG file. Filenames include the original image stem and
        the patch's unique tag. Also stores patch metadata for later use.

        Args:
            path (str or Path): Directory path where patch images will be saved.
                              Directory will be created if it doesn't exist.

        Side Effects:
            - Creates individual patch image files
            - Updates self._patch_info with patch metadata
            - Files are named as "{original_stem}_{tag}.jpg"

        Example:
            >>> pf.save_patches("output/patches/")
            # Creates: image1_123.jpg, image1_456.jpg, etc.
        """
        os.makedirs(path, exist_ok=True)
        img = np.array(self.image)
        for i, box in enumerate(self._extracted_boxes):
            (left, top, right, bottom) = box
            patch = img[top:bottom, left:right]
            tag = self.tags[i]
            filename = Path(
                path, f"{Path(self._img_filename).stem}_{tag}"
            ).with_suffix(".jpg")
            plt.imsave(filename, patch)

            # Store patch information
            self._patch_info[str(tag)] = {
                "filename": filename.name,
                "coordinates": [left, top, right, bottom],
            }

    def save_patch_info(self, path):
        """
        Save patch metadata to a JSON file.

        Creates a JSON file containing information about all extracted patches,
        including their filenames and bounding box coordinates. This metadata
        can be used for further processing or analysis.

        Args:
            path (str or Path): Directory where the JSON file will be saved.
                              File is named "{original_stem}_patch_info.json"

        JSON Structure:
            {
                "tag1": {
                    "filename": "image_tag1.jpg",
                    "coordinates": [left, top, right, bottom]
                },
                "tag2": { ... }
            }

        Example:
            >>> pf.save_patch_info("output/patches/")
            # Creates: image1_patch_info.json
        """
        info_file = Path(
            path, f"{Path(self._img_filename).stem}_patch_info.json"
        )
        with open(info_file, "w") as file:
            json.dump(self._patch_info, file, indent=2)


def load_args():
    """
    Load and parse command line arguments combined with environment settings.

    Loads default values from environment configuration and allows command line
    arguments to override them. Combines both sources into a single namespace.

    Returns:
        SimpleNamespace: Object containing all configuration parameters with
                        attributes for im_path, patches_dir, bbox_dir, and cp

    Command Line Arguments:
        --im_path: Path to input images directory
        --patches_dir: Path to save extracted patch images
        --bbox_dir: Path to save detection visualization images
        --cp: Path to YOLO model checkpoint file

    Example:
        >>> args = load_args()
        >>> args.im_path
        '/path/to/input/images'
        >>> args.cp
        '/path/to/model.pt'
    """
    args = load_env_arguments()

    parser = ArgumentParser()
    parser.add_argument(
        "--im_path",
        help="paths to input images containing multiple patches",
        default=args["images_in"],
    )
    parser.add_argument(
        "--patches_dir",
        help="path to save images with bounding boxes and patches crops",
        default=args["patches_dir"],
    )
    parser.add_argument(
        "--bbox_dir",
        help="path to save bounding box images",
        default=args["bbox_dir"],
    )

    parser.add_argument(
        "--cp", help="yolov8 cp path", default=args["model_nn_weights"]
    )

    parsed_args = parser.parse_args()
    return SimpleNamespace(**args, **parsed_args.__dict__)


if __name__ == "__main__":
    """
    Main execution block for batch processing images.
    
    Processes all JPEG images in the input directory, detecting patches,
    saving visualization images, extracting individual patches, and
    storing metadata files.
    
    Processing Steps:
    1. Load configuration arguments
    2. Create output directories
    3. Initialize PatchFinder with model
    4. Process each image:
       - Load image
       - Detect patches
       - Save detection visualization
       - Extract and save individual patches
       - Save patch metadata
    
    Progress is displayed using tqdm progress bar.
    """
    args = load_args()

    os.makedirs(args.im_path, exist_ok=True)
    os.makedirs(args.patches_dir, exist_ok=True)

    patch_finder = ImagePatchExtractor(args.cp)
    paths = list(Path(args.im_path).glob("*.jpg"))
    pbar = tqdm(paths, desc="Processing images")

    for i, img_filename in enumerate(pbar):
        pbar.set_description(f"{img_filename} {i + 1}/{len(paths)}")
        patch_finder.load_image(str(img_filename))
        patch_finder.predict_bounding_box()

        filename = Path(args.bbox_dir, img_filename.name)
        patch_finder.save_image(filename)

        filepath = Path(args.patches_dir, img_filename.stem)
        patch_finder.save_patches(filepath)

        # Save patch information
        patch_finder.save_patch_info(filepath)