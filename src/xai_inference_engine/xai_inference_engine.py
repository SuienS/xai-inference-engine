import threading
import torch
from PIL import Image

import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from .services import FMGCam
from .utils import ImageUtils

class XAIInferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        last_conv_layer: torch.nn.Module,
        device: torch.device = None,
        num_workers: int = 1,
    ):
        self.model = model
        self.num_workers = num_workers
        self.last_conv_layer = last_conv_layer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        self.fm_g_cam_generator = FMGCam(self.model, self.last_conv_layer, self.device)

    def predict(
        self,
        img: Image.Image,
        img_tensor: torch.Tensor,
        class_count: int = 3,
        class_rank_index: int = None,
        act_mode: str = "relu",
        enhance: bool = True,
        alpha: float = 0.8,
        image_width: int = 224,
        image_height: int = 224,
    ):
        # Increase the stack size to prevent stack overflow  # TODO: Try to remove this
        threading.stack_size(1000000)
        fm_g_cam_generator_thread = threading.Thread(
            target=self.fm_g_cam_generator,
            kwargs={
                "img_tensor": img_tensor,
                "class_count": class_count,
                "enhance": enhance,
                "class_rank_index": class_rank_index,
                "act_mode": act_mode,
            },
        )
        fm_g_cam_generator_thread.start()
        preds, sorted_pred_indices, saliency_maps = self.fm_g_cam_generator.my_queue.get()

        saliency_maps = ImageUtils.colourise_heatmaps(saliency_maps)

        super_imp_img = ImageUtils.super_imposed_image(
            saliency_maps,
            img,
            alpha=alpha,
            image_width=image_width,
            image_height=image_height,
        )

        return preds, sorted_pred_indices, super_imp_img, saliency_maps
