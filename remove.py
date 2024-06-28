import os
import sys
import tqdm
import wget
import gdown
import torch
import shutil
import warnings
import importlib

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from packaging import version
from easydict import EasyDict

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, "../../models/transparent-background")
home_dir = os.path.abspath(relative_path)
if not os.path.exists(home_dir):
    os.makedirs(home_dir)

filepath = os.path.abspath(__file__)
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from transparent_background.InSPyReNet import InSPyReNet_SwinB
from transparent_background.utils import *

class Remover:
    def __init__(self, mode="base", jit=False, device=None, ckpt=None, fast=None):
        """
        Args:
            mode   (str): Choose among below options
                                   base -> slow & large gpu memory required, high quality results
                                   fast -> resize input into small size for fast computation
                                   base-nightly -> nightly release for base mode
            jit    (bool): use TorchScript for fast computation
            device (str, optional): specifying device for computation. find available GPU resource if not specified.
            ckpt   (str, optional): specifying model checkpoint. find downloaded checkpoint or try download if not specified.
            fast   (bool, optional, DEPRECATED): replaced by mode argument. use fast mode if True.
        """
        # home_dir = os.path.expanduser(os.path.join("~", ".transparent-background"))
        # os.makedirs(home_dir, exist_ok=True)

        if not os.path.isfile(os.path.join(home_dir, "config.yaml")):
            shutil.copy(os.path.join(repopath, "config.yaml"), os.path.join(home_dir, "config.yaml"))
        self.meta = load_config(os.path.join(home_dir, "config.yaml"))[mode]

        if fast is not None:
            warnings.warn("fast argument is deprecated. use mode argument instead.")
            if fast:
                mode = "fast"

        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif (
                version.parse(torch.__version__) >= version.parse("1.13")
                and torch.backends.mps.is_available()
            ):
                self.device = "mps:0"

        download = False
        if ckpt is None:
            ckpt_dir = home_dir
            ckpt_name = self.meta.ckpt_name

            if not os.path.isfile(os.path.join(ckpt_dir, ckpt_name)):
                download = True
            elif (
                self.meta.md5
                != hashlib.md5(
                    open(os.path.join(ckpt_dir, ckpt_name), "rb").read()
                ).hexdigest()
            ):
                if self.meta.md5 is not None:
                    download = True

            if download:
                if 'drive.google.com' in self.meta.url:
                    gdown.download(self.meta.url, os.path.join(ckpt_dir, ckpt_name), fuzzy=True, proxy=self.meta.http_proxy)
                elif 'github.com' in self.meta.url:
                    wget.download(self.meta.url, os.path.join(ckpt_dir, ckpt_name))
                else:
                    raise NotImplementedError('Please use valid URL')
        else:
            ckpt_dir, ckpt_name = os.path.split(os.path.abspath(ckpt))

        self.model = InSPyReNet_SwinB(depth=64, pretrained=False, threshold=None, **self.meta)
        self.model.eval()
        self.model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, ckpt_name), map_location="cpu"),
            strict=True,
        )
        self.model = self.model.to(self.device)

        if jit:
            ckpt_name = self.meta.ckpt_name.replace(
                ".pth", "_{}.pt".format(self.device)
            )
            try:
                traced_model = torch.jit.load(
                    os.path.join(ckpt_dir, ckpt_name), map_location=self.device
                )
                del self.model
                self.model = traced_model
            except:
                traced_model = torch.jit.trace(
                    self.model,
                    torch.rand(1, 3, *self.meta.base_size).to(self.device),
                    strict=True,
                )
                del self.model
                self.model = traced_model
                torch.jit.save(self.model, os.path.join(ckpt_dir, ckpt_name))

        self.transform = transforms.Compose(
            [
                static_resize(self.meta.base_size)
                if jit
                else static_resize(size=[384, 384])
                if 'fast' in mode
                else dynamic_resize(L=1280),
                tonumpy(),
                normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                totensor(),
            ]
        )

        self.background = {'img': None, 'name': None, 'shape': None}
        desc = "Mode={}, Device={}, Torchscript={}".format(
            mode, self.device, "enabled" if jit else "disabled"
        )
        print("Settings -> {}".format(desc))

    def process(self, img, type="rgba", threshold=None):
        """
        Args:
            img (PIL.Image): input image as PIL.Image type
            type (str): output type option as below.
                        'rgba' will generate RGBA output regarding saliency score as an alpha map. 
                        'green' will change the background with green screen.
                        'white' will change the background with white color.
                        '[255, 0, 0]' will change the background with color code [255, 0, 0]. 
                        'blur' will blur the background.
                        'overlay' will cover the salient object with translucent green color, and highlight the edges.
                        Another image file (e.g., 'samples/backgroud.png') will be used as a background, and the object will be overlapped on it.
            threshold (float or str, optional): produce hard prediction w.r.t specified threshold value (0.0 ~ 1.0)
        Returns:
            PIL.Image: output image

        """
        shape = img.size[::-1]
        x = self.transform(img)
        x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            pred = self.model(x)

        pred = F.interpolate(pred, shape, mode="bilinear", align_corners=True)
        pred = pred.data.cpu()
        pred = pred.numpy().squeeze()

        if threshold is not None:
            pred = (pred > float(threshold)).astype(np.float64)

        img = np.array(img)

        if type.startswith("["):
            type = [int(i) for i in type[1:-1].split(",")]

        if type == "map":
            img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)

        elif type == "rgba":
            r, g, b = cv2.split(img)
            pred = (pred * 255).astype(np.uint8)
            img = cv2.merge([r, g, b, pred])

        elif type == "green":
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155]
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])

        elif type == "white":
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [255, 255, 255]
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])

        elif len(type) == 3:
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * type
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])

        elif type == "blur":
            img = img * pred[..., np.newaxis] + cv2.GaussianBlur(img, (0, 0), 15) * (
                1 - pred[..., np.newaxis]
            )

        elif type == "overlay":
            bg = (
                np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155] + img
            ) // 2
            img = bg * pred[..., np.newaxis] + img * (1 - pred[..., np.newaxis])
            border = cv2.Canny(((pred > 0.5) * 255).astype(np.uint8), 50, 100)
            img[border != 0] = [120, 255, 155]

        elif type.lower().endswith((".jpg", ".jpeg", ".png")):
            if self.background['name'] != type:
                background_img = cv2.cvtColor(cv2.imread(type), cv2.COLOR_BGR2RGB)
                background_img = cv2.resize(background_img, img.shape[:2][::-1])
                
                self.background['img'] = background_img
                self.background['shape'] = img.shape[:2][::-1]
                self.background['name'] = type
            
            elif self.background['shape'] != img.shape[:2][::-1]:
                self.background['img'] = cv2.resize(self.background['img'], img.shape[:2][::-1])
                self.background['shape'] = img.shape[:2][::-1]

            img = img * pred[..., np.newaxis] + self.background['img'] * (
                1 - pred[..., np.newaxis]
            )

        return Image.fromarray(img.astype(np.uint8))


