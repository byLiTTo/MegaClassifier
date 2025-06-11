"""YoloV5 base detector class."""

import numpy as np
from PIL import Image
import supervision as sv

import torch
from torch.hub import load_state_dict_from_url

from yolov5.utils.general import non_max_suppression, scale_boxes

from ..base_detector import BaseDetector
from ....data import transforms as pw_trans


class YOLOV5Base(BaseDetector):
    """
    Base detector class for YOLO V5. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """

    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the YOLO V5 detector.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
            transform (callable, optional):
                Optional transform to be applied on the image. Defaults to None.
        """
        self.transform = transform
        super(YOLOV5Base, self).__init__(weights=weights, device=device, url=url)
        self._load_model(weights, device, url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLO V5 model weights.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        if weights:
            checkpoint = torch.load(weights, map_location=torch.device(device))
        elif url:
            checkpoint = load_state_dict_from_url(url, map_location=torch.device(self.device))
        else:
            raise Exception("Need weights for inference.")
        self.model = checkpoint["model"].float().fuse().eval().to(self.device)

        if not self.transform:
            self.transform = pw_trans.MegaDetector_v5_Transform(
                target_size=self.IMAGE_SIZE, stride=self.STRIDE
            )

    def results_generation(self, preds, img_id, id_strip=None) -> dict:
        """
        Generate results for detection based on model predictions.

        Args:
            preds (numpy.ndarray):
                Model predictions.
            img_id (str):
                Image identifier.
            id_strip (str, optional):
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        results = {"img_id": str(img_id).strip(id_strip)}
        results["detections"] = sv.Detections(
            xyxy=preds[:, :4], confidence=preds[:, 4], class_id=preds[:, 5].astype(int)
        )
        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(
                results["detections"].confidence, results["detections"].class_id
            )
        ]
        return results

    def single_image_detection(
        self, img, img_path=None, det_conf_thres=0.2, id_strip=None
    ) -> dict:
        """
        Perform detection on a single image.

        Args:
            img (str or ndarray):
                Image path or ndarray of images.
            img_path (str, optional):
                Image path or identifier.
            det_conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        if type(img) == str:
            if img_path is None:
                img_path = img
            img = np.array(Image.open(img_path).convert("RGB"))
        img_size = img.shape
        img = self.transform(img)

        if img_size is None:
            img_size = img.permute((1, 2, 0)).shape  # We need hwc instead of chw for coord scaling
        preds = self.model(img.unsqueeze(0).to(self.device))[0]
        preds = (
            torch.cat(non_max_suppression(prediction=preds, conf_thres=det_conf_thres), axis=0)
            .cpu()
            .numpy()
        )
        # preds[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, preds[:, :4], img_size).round()
        preds[:, :4] = scale_boxes([self.IMAGE_SIZE] * 2, preds[:, :4], img_size).round()
        res = self.results_generation(preds, img_path, id_strip)

        normalized_coords = [
            [x1 / img_size[1], y1 / img_size[0], x2 / img_size[1], y2 / img_size[0]]
            for x1, y1, x2, y2 in preds[:, :4]
        ]
        res["normalized_coords"] = normalized_coords

        return res
