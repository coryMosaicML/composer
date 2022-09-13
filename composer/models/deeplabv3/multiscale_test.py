import argparse
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

parser = argparse.ArgumentParser(description='Testing script for deeplabV3+')
parser.add_argument('-c','--checkpoint', help='Path to the checkpoint to evaluate', required=True)
parser.add_argument('-i','--images', help='Path to the images to evaluate on', required=True)
parser.add_argument('-a','--annotations', help='Path to the image annotations to evaluate on', required=False, default=None)
parser.add_argument('-s','--save_dir', help='Place to save generated outputs', required=False, default=None)
args = parser.parse_args()


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
def load_images(images_path, annotations_path):
    images = []
    for filename in tqdm(os.listdir(images_path)):
        img_path = os.path.join(images_path, filename)

        # open an image
        pil_image = Image.open(img_path)
        image = np.array(pil_image)
        if image.ndim == 3:
            # Image is rgb, everythings fine
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
        elif image.ndim == 2:
            # Convert greyscale image to rgb
            rgbimg = Image.new("RGB", pil_image.size)
            rgbimg.paste(pil_image)
            image = np.array(rgbimg)
            image = torch.from_numpy(image)
            print(filename, image.shape)
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f'Uh oh! Something wrong with {filename} of shape {image.shape}')

        image = image.unsqueeze(dim=0).float().cuda()
        # Make the resize transform back to original resolution
        resize_original = torchvision.transforms.Resize(size=(image.shape[-2], image.shape[-1]), interpolation=TF.InterpolationMode.BILINEAR)

        # Prep image for eval
        image = resize_512_img(image)
        image = normalize(image)

        # open the annotations if specified
        if annotations_path is not None:
            ann_path = os.path.join(annotations_path, filename.replace('.jpg', '.png'))
            target = Image.open(ann_path)
            target = np.array(target)
            target = torch.from_numpy(target).cuda()
            target = target.unsqueeze(0) - 1
            # Prep for eval
            target = resize_512_ann(target)
        else:
            target = None
        images.append((image, target, resize_original, filename))

    return images

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

# Load a checkpoint
state_dict = torch.load(args.checkpoint)
model.load_state_dict(state_dict["state"]["model"])

model = model.cuda()
model = model.eval()

images = load_images(args.images, args.annotations)

with torch.no_grad():
    for image, target, resize_original, filename in tqdm(images):
        # Run the multiscale testing
        ss_output, ms_output, ms_flips_output = multiscale_test(model, image)

        if target is not None:
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

        if args.save_dir is not None:
            # Resize the multiscale+flips output to original size
            resized_output = resize_original(ms_flips_output)
            # Create the segmentation output
            segmentation_output = torch.argmax(resized_output, dim=1).squeeze()
            segmentation_output = segmentation_output.data.cpu().numpy()
            # Convert to an image and save
            segmentation_image = Image.fromarray(np.uint8(segmentation_output))
            output_name = filename.replace('.jpg', '.png')
            segmentation_image.save(os.path.join(args.save_dir, output_name))