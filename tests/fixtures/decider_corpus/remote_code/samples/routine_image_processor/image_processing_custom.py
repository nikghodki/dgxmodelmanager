from typing import List, Union

import numpy as np
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


class CustomImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, size=448, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__(**kwargs)
        self.size, self.mean, self.std = size, np.array(mean), np.array(std)

    def preprocess(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        images = images if isinstance(images, list) else [images]
        arrs = [(np.asarray(im.convert("RGB").resize((self.size, self.size))) / 255.0 - self.mean) / self.std
                for im in images]
        return BatchFeature({"pixel_values": np.stack(arrs).transpose(0, 3, 1, 2)})
