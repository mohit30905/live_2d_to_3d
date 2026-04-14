import torch 
import cv2 
import numpy as np


class MiDaSDepth:
    def __init__(self, model_type='MiDaS_small', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        print(f"[MiDaS] using device: {self.device}, model: {self.model_type}")
        # load model (downloads on first run)
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.model.to(self.device).eval()
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.transforms.small_transform if model_type == 'MiDaS_small' else self.transforms.default_transform

    def predict(self, bgr_img, downscale=1.0):
        """
        bgr_img: HxWx3 BGR numpy
        downscale: if >1.0, we resize input to speed up MiDaS and then resize output back
        returns HxW float32 normalized depth in range 0..1
        """
        if downscale != 1.0:
            h, w = bgr_img.shape[:2]
            small = cv2.resize(bgr_img, (int(w/downscale), int(h/downscale)), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
            depth = prediction.cpu().numpy()
            # resize back to original frame size
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False
                ).squeeze()
            depth = prediction.cpu().numpy()

        # normalize to 0..1 (relative depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype('float32')
