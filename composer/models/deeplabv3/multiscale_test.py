import os
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF

from composer.models import composer_deeplabv3
from composer.trainer import Trainer
from composer.metrics import CrossEntropy, MIoU
import timeit

miou_metric = MIoU(num_classes=150)

MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)
normalize = torchvision.transforms.Normalize(MEAN, STD)
resize_512_img = torchvision.transforms.Resize(size=(512, 512), interpolation=TF.InterpolationMode.BILINEAR)
resize_512_ann = torchvision.transforms.Resize(size=(512, 512), interpolation=TF.InterpolationMode.NEAREST)


model = composer_deeplabv3(num_classes=150,
                           backbone_arch="resnet101",
                           backbone_weights=None,
                           use_plus=True,
                           sync_bn=False,
                           cross_entropy_weight=0.375,
                           dice_weight=1.125)

state_dict = torch.load("/root/data/dice-ep128-ba20096-rank0")
model.load_state_dict(state_dict["state"]["model"])

model = model.cuda()
model = model.eval()

val_dir = '/root/data/val-images/'
ann_dir = '/root/data/val-annotations/'

corr_pixels = 0
total_pixels = 0

for filename in os.listdir(val_dir):
    img_path = os.path.join(val_dir, filename)
    ann_path = os.path.join(ann_dir, filename.replace('.jpg', '.png'))
    # open an image
    image = Image.open(img_path)
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(dim=0).float().cuda()

    # open the annotations
    target = Image.open(ann_path)
    target = np.array(target)
    target = torch.from_numpy(target).cuda()
    target = target.unsqueeze(0) - 1

    # Prep image and target for eval
    image = resize_512_img(image)
    target = resize_512_ann(target)
    image = normalize(image)

    # Run the image through the network
    output = model.forward((image, None))

    # Compute MIoU
    miou_metric.update(output.cpu(), target.cpu())
    print("miou:", miou_metric.compute())

    # Compute pixel accuracy
    predicted = output.argmax(dim=1)
    matched = (predicted == target)
    mask = torch.zeros_like(target)
    mask[target >= 0] = 1

    corr_pixels += torch.sum(matched)
    total_pixels += torch.sum(mask)
    acc = corr_pixels / total_pixels
    print("accuracy:", acc)