
using PyCall


py"""
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

def init():
    return transforms.Compose(
                    [   transforms.ToPILImage(),
                        transforms.RandomResizedCrop(
                            (32, 32),
                            scale=(0.08, 1.0) 
                        ),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    ]
                )


def transform_x(transform_model, x):
    return transform_model(x)
"""


function transform(x)
    return py"transform_x"(py"init"(), x)[:numpy]()
end
