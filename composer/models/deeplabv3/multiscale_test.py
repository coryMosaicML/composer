import os
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm

from composer.models import composer_deeplabv3
from composer.trainer import Trainer
from composer.metrics import CrossEntropy, MIoU

ss_miou_metric = MIoU(num_classes=150)
ms_miou_metric = MIoU(num_classes=150)
ms_flips_miou_metric = MIoU(num_classes=150)

class PixelAccuracy:
    def __init__(self):
        self.corr_pixels = 0
        self.total_pixels = 0

    def update(self, output, target):
        predicted = output.argmax(dim=1)
        matched = (predicted == target)
        mask = torch.zeros_like(target)
        mask[target >= 0] = 1

        self.corr_pixels += torch.sum(matched)
        self.total_pixels += torch.sum(mask)

    def compute(self):
        return 100 * self.corr_pixels / self.total_pixels

ss_pacc_metric = PixelAccuracy()
ms_pacc_metric = PixelAccuracy()
ms_flips_pacc_metric = PixelAccuracy()

MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)
normalize = torchvision.transforms.Normalize(MEAN, STD)
resize_512_img = torchvision.transforms.Resize(size=(512, 512), interpolation=TF.InterpolationMode.BILINEAR)
resize_512_ann = torchvision.transforms.Resize(size=(512, 512), interpolation=TF.InterpolationMode.NEAREST)

resize_256_img = torchvision.transforms.Resize(size=(256, 256), interpolation=TF.InterpolationMode.BILINEAR)
resize_384_img = torchvision.transforms.Resize(size=(384, 384), interpolation=TF.InterpolationMode.BILINEAR)
resize_640_img = torchvision.transforms.Resize(size=(640, 640), interpolation=TF.InterpolationMode.BILINEAR)
resize_768_img = torchvision.transforms.Resize(size=(768, 768), interpolation=TF.InterpolationMode.BILINEAR)
resize_896_img = torchvision.transforms.Resize(size=(896, 896), interpolation=TF.InterpolationMode.BILINEAR)

resize_512_out = torchvision.transforms.Resize(size=(512, 512), interpolation=TF.InterpolationMode.BILINEAR)
multisizes = [resize_256_img, resize_384_img, resize_640_img, resize_768_img, resize_896_img]

# [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
def get_output(model, image):
    output = model((image, None))
    output = torch.nn.functional.softmax(output, dim=1)
    return output

def multiscale_test(model, image):
    outputs = get_output(model, image)
    ss_outputs = torch.clone(outputs)
    ms_outputs = torch.clone(outputs)
    ms_flips_outputs = torch.clone(outputs)

    flipped_image = torchvision.transforms.functional.hflip(image)
    flipped_outputs = get_output(model, flipped_image)
    flipped_outputs = torchvision.transforms.functional.hflip(flipped_outputs)
    ms_flips_outputs += flipped_outputs

    for size in multisizes:
        sized_image = size(image)
        sized_outputs = get_output(model, sized_image)
        sized_outputs = resize_512_out(sized_outputs)
        ms_outputs += sized_outputs
        ms_flips_outputs += sized_outputs

        flipped_sized_image = torchvision.transforms.functional.hflip(sized_image)
        flipped_sized_outputs = get_output(model, flipped_sized_image)
        flipped_sized_outputs = resize_512_out(flipped_sized_outputs)
        flipped_sized_outputs = torchvision.transforms.functional.hflip(flipped_sized_outputs)
        ms_flips_outputs += flipped_sized_outputs
    return ss_outputs, ms_outputs, ms_flips_outputs


model = composer_deeplabv3(num_classes=150,
                           backbone_arch="resnet101",
                           backbone_weights=None,
                           use_plus=True,
                           sync_bn=False,
                           cross_entropy_weight=0.375,
                           dice_weight=1.125)

# Baseline dice checkpoint
#state_dict = torch.load("/root/data/dice-ep128-ba20096-rank0")
# Broken ema+sam+mx ssr 1 checkpoint
state_dict = torch.load("/root/data/broken-ema-mx-sam-ep128-ba20096-rank0")
# xent baseline checkpoint
#state_dict = torch.load("/root/data/xent-ep128-ba20096-rank0")
model.load_state_dict(state_dict["state"]["model"])

model = model.cuda()
model = model.eval()

val_dir = '/root/data/inference_data/val-images/'
ann_dir = '/root/data/inference_data/val-annotations/'

with torch.no_grad():
    for filename in tqdm(os.listdir(val_dir)):
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
        #output = model.forward((image, None))
        # Run the multiscale testing
        ss_output, ms_output, ms_flips_output = multiscale_test(model, image)

        # Update MIoU
        ss_miou_metric.update(ss_output.cpu(), target.cpu())
        ms_miou_metric.update(ms_output.cpu(), target.cpu())
        ms_flips_miou_metric.update(ms_flips_output.cpu(), target.cpu())

        # Update pixel accuracy
        ss_pacc_metric.update(ss_output.cpu(), target.cpu())
        ms_pacc_metric.update(ms_output.cpu(), target.cpu())
        ms_flips_pacc_metric.update(ms_flips_output.cpu(), target.cpu())

        print("ss MIoU: {:.2f}, ms MIoU {:.2f} ms+flips MIoU {:.2f}".format(ss_miou_metric.compute().item(),
                                                                ms_miou_metric.compute().item(),
                                                                ms_flips_miou_metric.compute().item()))

        print("ss PAcc: {:.2f}, ms PAcc {:.2f} ms+flips PAcc {:.2f}".format(ss_pacc_metric.compute().item(),
                                                                ms_pacc_metric.compute().item(),
                                                                ms_flips_pacc_metric.compute().item()))