import torch
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from torch_cka import CKA
from src.model import Model
from src import utility
from src import data

# Usage: git clone EDSR_pytorch: https://github.com/sanghyun-son/EDSR-PyTorch
# put this script into the EDSR dir
# modify some "import" in EDSR code
# config option1.py + option2.py
# run this code

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

from src.option1 import args as args1
from src.option2 import args as args2

ck1 = utility.checkpoint(args1)
ck2 = utility.checkpoint(args2)
batch_size = 16

loader = data.Data(batch_size=batch_size, args = args1).loader_test[0]

model1 = Model(args1, ck1)
model2 = Model(args2, ck2)

layers = []
for name, layer in model2.named_modules():
    print(name)
    if ("body.2" in name and "body.2.b" not in name) or ("body.1" in name and "body.1.b" not in name):
        layers = layers + [name]


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



dataset = CIFAR10(root='.',
                  train=False,
                  download=True,
                  transform=transform)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)

# cka = CKA(model1, model2,
#         model1_name="ResNet18", model2_name="ResNet34",
#         device='cuda')
#
# cka.compare(dataloader)
#
# cka.plot_results(save_path="../assets/resnet_compare.png")


#===============================================================
"""model1 = resnet50(pretrained=True)
model2 = wide_resnet50_2(pretrained=True)"""


cka = CKA(model1, model2,
        model1_name="x3", model2_name="x4",
        model1_layers=layers, model2_layers=layers,
        device='cuda')

cka.compare(batch_size, loader)

cka.plot_results(save_path="./EDSRx3_EDSRx4_compare.png")