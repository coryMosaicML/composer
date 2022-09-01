from PIL import Image
import numpy as np
import torch

from composer.models import composer_deeplabv3
from composer.trainer import Trainer
from composer.metrics import CrossEntropy, MIoU
import timeit

model = composer_deeplabv3(num_classes=150,
                           backbone_arch="resnet101",
                           backbone_weights=None,
                           use_plus=True,
                           sync_bn=False,
                           cross_entropy_weight=0.375,
                           dice_weight=1.125)

model = model.cuda()
# open an image
image = Image.open('/root/data/inference_data/test-images/ADE_test_00001361.jpg')
image = np.asarray(image)
image = torch.from_numpy(image)
image = image.permute(2, 0, 1)
image = image.unsqueeze(dim=0).float()
print(image.shape)
image = torch.randn((1, 3, 512, 512)).cuda()
start = timeit.timeit()
model = model.eval()
stop = timeit.timeit()
output = model.forward((image, None))
print(output.shape)
print("Time:", stop - start)