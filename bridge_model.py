# -*- coding: utf-8 -*-
"""Verified architecture: SMP DeepLabV3+ (resnet50), ASPP hook -> official VMamba SS2D.
Checkpoint expects base.* and vmamba.*. No ZoeDepth, no YOLO.
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ['background', 'crack', 'spalling', 'corrosion', 'exposed_rebar']
CLASS_CN = ['背景', '裂缝', '混凝土剥落', '可见锈蚀迹象', '钢筋外露']
COLORS = [(0,0,0), (230,35,35), (255,140,0), (230,215,0), (0,200,230)]


def crop_center(image: Image.Image, width=640, height=480):
    """Same center crop as HRCDS evaluation. Unobserved outside must remain 255."""
    right = max(0, width - image.width)
    bottom = max(0, height - image.height)
    if right or bottom:
        image = ImageOps.expand(image, border=(0, 0, right, bottom), fill=0)
    x0 = (image.width - width) // 2
    y0 = (image.height - height) // 2
    return image.crop((x0, y0, x0 + width, y0 + height)), (x0, y0), (right, bottom)


def colorize(pred):
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for i, c in enumerate(COLORS):
        rgb[pred == i] = c
    return rgb


def build_model(vmamba_path):
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    p = Path(vmamba_path)
    if not (p / 'vmamba.py').is_file():
        raise FileNotFoundError('未找到官方 VMamba vmamba.py: ' + str(p))
    sys.path.insert(0, str(p))
    from vmamba import SS2D

    class DeepLabV3PlusVMamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights=None,
                                           in_channels=3, classes=5, activation=None)
            self.vmamba = SS2D(d_model=256, d_state=16, ssm_ratio=1.0,
                                dt_rank='auto', d_conv=3, forward_type='v03',
                                channel_first=True, dropout=0.0)
            self._aspp_hook = self.base.decoder.aspp.register_forward_hook(
                lambda module, inputs, output: self.vmamba(output))

        def forward(self, x):
            return self.base(x)
    return DeepLabV3PlusVMamba()


def infer(image_path, checkpoint_path, vmamba_path, device=None):
    import torch
    from torchvision.transforms import functional as TF
    p = Path(image_path)
    source = Image.open(p).convert('RGB')
    crop, (x0, y0), (pad_r, pad_b) = crop_center(source)
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(vmamba_path)
    # Only load the checkpoint when it is from your own trusted training server.
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get('model_state_dict'), dict):
        raise ValueError('无效 checkpoint：没有 model_state_dict，必须使用 DeepLabV3+ / VMamba 正式权重。')
    loaded = model.load_state_dict(ckpt['model_state_dict'], strict=True)
    assert len(loaded.missing_keys) == len(loaded.unexpected_keys) == 0
    model.to(device).eval()
    tensor = TF.normalize(TF.to_tensor(crop),
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.amp.autocast(device_type='cuda', enabled=(str(device).startswith('cuda'))):
            logits = model(tensor)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    if pred.shape != (480,640):
        raise RuntimeError(f'Unexpected mask size: {pred.shape}')
    # Keep model's native 640x480 coordinate system to prevent false full-image coverage.
    return pred, np.asarray(crop), {
        'image_path': str(p.resolve()), 'input_wh': list(source.size),
        'prediction_wh': [640,480], 'crop_xy_in_padded_image':[x0,y0],
        'padding_right_bottom':[pad_r,pad_b], 'checkpoint_epoch':ckpt.get('epoch'),
        'class_map_authority':'Verified HRCDS integer GT IDs, NOT checkpoint config class_names',
        'experiment':ckpt.get('experiment'), 'best_val_miou_recorded':ckpt.get('best_val_miou'),
        'device':str(device), 'coverage':'center_crop_only'
    }
