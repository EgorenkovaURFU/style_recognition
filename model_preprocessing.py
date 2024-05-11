from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
 

# TODO можно оформить как функции
# аугментация тренировочного датасета и нормолизация валидационного и тестового

train_dataset = ImageFolder(root='data/train')
valid_dataset = ImageFolder(root='data/valid')
test_dataset  = ImageFolder(root='data/test')

normalize = transforms.Normalize(mean=[0.49333772, 0.51176786, 0.51791704],
                                 std=[0.26378724, 0.26562205, 0.3115852])

train_dataset.transform = transforms.Compose([
                          transforms.Resize(size=224, interpolation=Image.BILINEAR),
                          transforms.CenterCrop(size=(224, 224)),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomAutocontrast(),
                          transforms.RandomEqualize(),
                          transforms.ToTensor(),
                          normalize
])

valid_dataset.transform = transforms.Compose([
                          transforms.Resize(size=224, interpolation=Image.BILINEAR),
                          transforms.CenterCrop(size=(224, 224)),
                          transforms.ToTensor(),
                          normalize
])

test_dataset.transform = transforms.Compose([
                          transforms.Resize(size=224, interpolation=Image.BILINEAR),
                          transforms.CenterCrop(size=(224, 224)),
                          transforms.ToTensor(),
                          normalize
])
