import os 
import sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from PIL import Image
from remove import Remover


device = "cuda" if torch.cuda.is_available() else "cpu"
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class InSPyResnet_ModelLoader:
    """
    InSPyResnet Model Loader
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["base", "fast", "base-nightly"],),
            }
        }

    RETURN_NAMES = ("INSPYMODEL",)
    FUNCTION = "load_model"
    CATEGORY = "🎨INSPY RMBG"
    RETURN_TYPES = ("InspyMODEL",)

    def load_model(self, mode):
        assert mode in ["base", "fast", "base-nightly"]
        if mode == "base":
            net = Remover()
        elif mode == "fast":
            net = Remover(mode='fast', jit=True)
        elif mode == "base-nightly":
            net = Remover(mode='base-nightly')
        net.model.to(device)
        return [net]


class INSPY_RMBG_Apply:
    pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "INSPYMODEL": ("InspyMODEL",),
                "image": ("IMAGE",),
                "background": (['RGBA', 'BLACK', 'WHITE', 'RED'],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "remove_background"
    CATEGORY = "🎨INSPY RMBG"

    def resize(self, img: Image, target_size_px: int):
        aspect_ratio = img.width / img.height

        if img.width > img.height:
            new_width = target_size_px
            new_height = int(target_size_px / aspect_ratio)
        else:
            new_height = target_size_px
            new_width = int(target_size_px * aspect_ratio)

        img = img.resize((new_width, new_height))

        return img
    def remove_background(self, INSPYMODEL, image, background):
        processed_images = []
        processed_masks = []
        assert background in ['RGBA', 'BLACK', 'WHITE', 'RED']
        for image in image:
            image = tensor2pil(image)
            alpha_prediction = INSPYMODEL.process(image, type="map")
            pil_im = Image.fromarray(np.squeeze(alpha_prediction)).convert('L')
            if background == "RGBA":
                new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
            elif background == "BLACK":
                new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 255))
            elif background == "WHITE":
                new_im = Image.new("RGBA", pil_im.size, (255, 255, 255, 255))
            elif background == "RED":
                new_im = Image.new("RGBA", pil_im.size, (255,0,0,255))
            new_im.paste(image, mask=pil_im)
            if background != "RGBA":
                new_im = new_im.convert("RGB")

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)
        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks
    
NODE_CLASS_MAPPINGS = {
    "INSPY removebg ModelLoader": InSPyResnet_ModelLoader,
    "INSPY removebg Apply": INSPY_RMBG_Apply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSPY_RMBG_ModelLoader": "INSPY removebg ModelLoader",
    "INSPY_RMBG_Apply": "INSPY RMBG",
}