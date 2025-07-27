"""
ImagePatchExtractor Module

This module provides functionality for detecting and extracting patches from images using
either a YOLO (You Only Look Once) or Faster R-CNN object detection model. It can process
images to identify patch-like objects, extract them as separate images, and save both the
detection results and individual patch crops.

The main class ImagePatchExtractor supports both:
1. YOLO models (ultralytics implementation)
2. Faster R-CNN models (PyTorch implementation)

Dependencies:
    - cv2 (OpenCV): Image processing operations
    - numpy: Numerical computations and array operations
    - matplotlib: Image visualization and saving
    - ultralytics: YOLO model implementation
    - torch: PyTorch for Faster R-CNN
    - torchvision: PyTorch vision utilities
    - PIL: Image processing
    - tqdm: Progress bar for batch processing
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from enum import Enum

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torchvision
from PIL import Image, ImageOps

# Try to import YOLO, make it optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO models will not work.")

from env_arguments_loader import load_env_arguments


class ModelType(Enum):
    """Enumeration for supported model types."""
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"


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
    A class for detecting and extracting patches from images using either YOLO or Faster R-CNN.

    This class provides comprehensive functionality for:
    - Loading and processing images
    - Detecting patch objects using either YOLO or Faster R-CNN models
    - Extracting individual patch regions
    - Generating tagged visualization images
    - Saving patches and metadata

    Attributes:
        image (numpy.ndarray): Currently loaded image
        _model: Loaded model for object detection (YOLO or Faster R-CNN)
        _model_type (ModelType): Type of the loaded model
        _device (torch.device): Device for PyTorch models
        _img_filename (str): Path to the currently loaded image file
        _extracted_boxes (list): List of detected bounding boxes
        _scores (list): Confidence scores for detected boxes
        _id_map (numpy.ndarray): Grid-based ID mapping for patch tagging
        _tags (list): List of unique tags for detected patches
        _patch_info (dict): Metadata dictionary for extracted patches
    """

    def __init__(self, model_path, model_type="auto", confidence_threshold=0.5):
        """
        Initialize ImagePatchExtractor with a model checkpoint.

        Args:
            model_path (str): Path to the model checkpoint file
            model_type (str): Type of model - "yolo", "faster_rcnn", or "auto" for auto-detection
            confidence_threshold (float): Minimum confidence score for detections

        Example:
            >>> # YOLO model
            >>> pf = ImagePatchExtractor("models/yolov8_patches.pt", "yolo")
            >>> # Faster R-CNN model
            >>> pf = ImagePatchExtractor("models/frcnn_r50fpn_epoch_2299.pth", "faster_rcnn")
            >>> # Auto-detect model type
            >>> pf = ImagePatchExtractor("models/model.pt", "auto")
        """
        self.image = None
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._img_filename = None
        self._extracted_boxes = None
        self._scores = None
        self._id_map = None
        self._tags = []
        self._patch_info = {}

        # Determine model type
        if model_type == "auto":
            self._model_type = self._detect_model_type(model_path)
        else:
            self._model_type = ModelType(model_type)

        # Load the appropriate model
        self._model = self._load_model()

    def _detect_model_type(self, model_path):
        """
        Auto-detect model type based on file extension and structure.

        Args:
            model_path (str): Path to the model file

        Returns:
            ModelType: Detected model type
        """
        if model_path.endswith('.pt') and 'yolo' in model_path.lower():
            return ModelType.YOLO
        elif model_path.endswith('.pth') and ('frcnn' in model_path.lower() or 'faster' in model_path.lower()):
            return ModelType.FASTER_RCNN
        elif model_path.endswith('.pt'):
            # Default to YOLO for .pt files
            return ModelType.YOLO
        elif model_path.endswith('.pth'):
            # Default to Faster R-CNN for .pth files
            return ModelType.FASTER_RCNN
        else:
            raise ValueError(f"Cannot auto-detect model type for {model_path}. Please specify model_type explicitly.")

    def _load_model(self):
        """
        Load the appropriate model based on model type.

        Returns:
            Model object (YOLO or torch.nn.Module)
        """
        if self._model_type == ModelType.YOLO:
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics package is required for YOLO models")
            return YOLO(self._model_path)

        elif self._model_type == ModelType.FASTER_RCNN:
            # Import the load_model function from tools.predict
            try:
                from tools.predict import load_model
            except ImportError:
                raise ImportError("tools.predict module is required for Faster R-CNN models")

            model = load_model(self._model_path)
            model.to(self._device)
            model.eval()
            return model

        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

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

    def _predict_yolo(self, model_input_shape=(640, 640), patch_cls_name="patch"):
        """
        Detect patches using YOLO model.

        Args:
            model_input_shape (tuple): Target size for model input
            patch_cls_name (str): Class name to filter detections
        """
        img = cv2.resize(self.image, model_input_shape)
        results = self._model.predict([img], verbose=False)
        names = results[0].names
        boxes = results[0].boxes
        cls = boxes.cls.numpy()

        self._images = []
        extracted_boxes = []
        scores = []

        for i, box in enumerate(boxes):
            if names[cls[i]] != patch_cls_name:
                continue

            left, top, right, bottom = box.xyxy.numpy()[0]
            confidence = box.conf.numpy()[0]

            if confidence < self._confidence_threshold:
                continue

            im_shape = self.image.shape
            left = int(left / model_input_shape[1] * im_shape[1])
            right = int(right / model_input_shape[1] * im_shape[1])
            top = int(top / model_input_shape[0] * im_shape[0])
            bottom = int(bottom / model_input_shape[0] * im_shape[0])

            extracted_boxes.append([left, top, right, bottom])
            scores.append(confidence)

        # Sort by distance from origin
        sorted_indices = sorted(range(len(extracted_boxes)),
                              key=lambda i: center_radius(extracted_boxes[i]))

        self._extracted_boxes = [extracted_boxes[i] for i in sorted_indices]
        self._scores = [scores[i] for i in sorted_indices]

        for left, top, right, bottom in self._extracted_boxes:
            self._images.append(self.image[top:bottom, left:right])

    def _predict_faster_rcnn(self):
        """
        Detect patches using Faster R-CNN model.
        """
        # Convert image to PIL and then to tensor
        pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert to tensor
        im_tensor = torchvision.transforms.ToTensor()(pil_image)
        im_tensor = im_tensor.unsqueeze(0).float().to(self._device)

        # Get predictions
        with torch.no_grad():
            frcnn_output = self._model(im_tensor, None)[0]

        boxes = frcnn_output['boxes'].detach().cpu().numpy()
        labels = frcnn_output['labels'].detach().cpu().numpy()
        scores = frcnn_output['scores'].detach().cpu().numpy()

        # Filter by confidence threshold
        valid_indices = scores >= self._confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]

        # Convert boxes to integer coordinates
        extracted_boxes = []
        valid_scores = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
            extracted_boxes.append([left, top, right, bottom])
            valid_scores.append(scores[i])

        # Sort by distance from origin
        sorted_indices = sorted(range(len(extracted_boxes)),
                              key=lambda i: center_radius(extracted_boxes[i]))

        self._extracted_boxes = [extracted_boxes[i] for i in sorted_indices]
        self._scores = [valid_scores[i] for i in sorted_indices]

        # Extract patch images
        self._images = []
        for left, top, right, bottom in self._extracted_boxes:
            self._images.append(self.image[top:bottom, left:right])

    def predict_bounding_box(
        self,
        model_input_shape: tuple[int, ...] = (640, 640),
        patch_cls_name: str = "patch",
    ):
        """
        Detect patch objects in the current image using the loaded model.

        Args:
            model_input_shape (tuple): Target size for model input (YOLO only)
            patch_cls_name (str): Class name to filter detections (YOLO only)

        Side Effects:
            - Updates self._extracted_boxes with detected bounding boxes
            - Updates self._scores with confidence scores
            - Updates self._images with cropped patch images
            - Boxes are sorted by distance from origin (top-left corner)
        """
        if self._model_type == ModelType.YOLO:
            self._predict_yolo(model_input_shape, patch_cls_name)
        elif self._model_type == ModelType.FASTER_RCNN:
            self._predict_faster_rcnn()
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

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
        a random color for visual distinction. Also displays confidence scores.

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
            confidence = self._scores[i] if self._scores else 1.0
            label = f"{tag} ({confidence:.2f})"

            im = cv2.putText(
                im,
                label,
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
        os.makedirs(Path(fn).parent, exist_ok=True)
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
            confidence = self._scores[i] if self._scores else 1.0

            filename = Path(
                path, f"{Path(self._img_filename).stem}_{tag}"
            ).with_suffix(".jpg")
            plt.imsave(filename, patch)

            # Store patch information
            self._patch_info[str(tag)] = {
                "filename": filename.name,
                "coordinates": [left, top, right, bottom],
                "confidence": float(confidence),
                "model_type": self._model_type.value
            }

    def save_patch_info(self, path):
        """
        Save patch metadata to a JSON file.

        Creates a JSON file containing information about all extracted patches,
        including their filenames, bounding box coordinates, confidence scores,
        and model type used. This metadata can be used for further processing
        or analysis.

        Args:
            path (str or Path): Directory where the JSON file will be saved.
                              File is named "{original_stem}_patch_info.json"

        JSON Structure:
            {
                "tag1": {
                    "filename": "image_tag1.jpg",
                    "coordinates": [left, top, right, bottom],
                    "confidence": 0.95,
                    "model_type": "yolo"
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

    def save_to_csv(self, csv_path):
        """
        Save detection results to a CSV file (compatible with Faster R-CNN format).

        Args:
            csv_path (str): Path to the output CSV file
        """
        import csv

        os.makedirs(Path(csv_path).parent, exist_ok=True)

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'x1', 'y1', 'x2', 'y2', 'confidence', 'tag', 'model_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, box in enumerate(self._extracted_boxes):
                left, top, right, bottom = box
                confidence = self._scores[i] if self._scores else 1.0
                tag = self.tags[i]

                writer.writerow({
                    'image_path': self._img_filename,
                    'x1': left,
                    'y1': top,
                    'x2': right,
                    'y2': bottom,
                    'confidence': confidence,
                    'tag': tag,
                    'model_type': self._model_type.value
                })


def load_args():
    """
    Load and parse command line arguments combined with environment settings.

    Loads default values from environment configuration and allows command line
    arguments to override them. Combines both sources into a single namespace.

    Returns:
        SimpleNamespace: Object containing all configuration parameters with
                        attributes for im_path, patches_dir, bbox_dir, cp, and model_type

    Command Line Arguments:
        --im_path: Path to input images directory
        --patches_dir: Path to save extracted patch images
        --bbox_dir: Path to save detection visualization images
        --cp: Path to model checkpoint file
        --model_type: Type of model ("yolo", "faster_rcnn", or "auto")
        --confidence_threshold: Minimum confidence score for detections

    Example:
        >>> args = load_args()
        >>> args.im_path
        '/path/to/input/images'
        >>> args.cp
        '/path/to/model.pt'
        >>> args.model_type
        'auto'
    """
    args = load_env_arguments()

    parser = ArgumentParser()
    parser.add_argument(
        "--im_path",
        help="paths to input images containing multiple patches",
        default=args.images_in,
    )
    parser.add_argument(
        "--patches_dir",
        help="path to save images with bounding boxes and patches crops",
        default=args.patches_dir,
    )
    parser.add_argument(
        "--bbox_dir",
        help="path to save bounding box images",
        default=args.bbox_dir,
    )
    parser.add_argument(
        "--cp",
        help="model checkpoint path",
        default=args.model_nn_weights
    )
    parser.add_argument(
        "--model_type",
        help="type of model: 'yolo', 'faster_rcnn', or 'auto'",
        default=args.model_type,
        choices=["yolo", "faster_rcnn", "auto"]
    )
    parser.add_argument(
        "--confidence_threshold",
        help="minimum confidence score for detections",
        type=float,
        default=args.confidence_threshold
    )

    parsed_args = parser.parse_args()
    # return SimpleNamespace(**args, **parsed_args.__dict__)
    merged_dict = {**args.__dict__, **parsed_args.__dict__}
    combined_args = SimpleNamespace(**merged_dict)
    return combined_args

if __name__ == "__main__":
    """
    Main execution block for batch processing images.
    
    Processes all JPEG images in the input directory, detecting patches using
    either YOLO or Faster R-CNN models, saving visualization images, extracting 
    individual patches, and storing metadata files.
    
    Processing Steps:
    1. Load configuration arguments
    2. Create output directories
    3. Initialize ImagePatchExtractor with model
    4. Process each image:
       - Load image
       - Detect patches
       - Save detection visualization
       - Extract and save individual patches
       - Save patch metadata
       - Save CSV results (for compatibility)
    
    Progress is displayed using tqdm progress bar.
    """
    args = load_args()

    image_path = args.base_path + "/" + args.images_in
    patches_path = args.base_path +  "/OUTPUT_" + args.model_type + "/" + args.patches_dir
    bbox_path = args.base_path + "/OUTPUT_" + args.model_type + "/" + args.bbox_dir
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input directory '{image_path}' does not exist.")

    os.makedirs(patches_path, exist_ok=True)
    os.makedirs(bbox_path, exist_ok=True)

    print(f"Using model: {args.cp}")
    print(f"Model type: {args.model_type}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    patch_finder = ImagePatchExtractor(
        args.cp,
        model_type=args.model_type,
        confidence_threshold=args.confidence_threshold
    )

    paths = list(Path(image_path).glob("*.jpg"))
    pbar = tqdm(paths, desc="Processing images")

    for i, img_filename in enumerate(pbar):
        pbar.set_description(f"{img_filename} {i + 1}/{len(paths)}")

        try:
            patch_finder.load_image(str(img_filename))
            patch_finder.predict_bounding_box()

            # Save detection visualization
            bbox_filename = Path(bbox_path, img_filename.name)
            patch_finder.save_image(bbox_filename)

            # Save individual patches
            patches_filepath = Path(patches_path, img_filename.stem)
            patch_finder.save_patches(patches_filepath)

            # Save patch information as JSON
            patch_finder.save_patch_info(patches_filepath)

            # Save results as CSV (for compatibility with existing workflows)
            csv_filename = Path(patches_filepath, f"{img_filename.stem}.csv")
            patch_finder.save_to_csv(csv_filename)

        except Exception as e:
            print(f"Error processing {img_filename}: {str(e)}")
            continue

    print("Processing completed!")