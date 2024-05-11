from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from src import load_model

BATCH_SIZE = 32
PATH = 'models/model.pth' # путь для сохранения модели

train_dataset = ImageFolder(root='data/train')
valid_dataset = ImageFolder(root='data/valid')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'



def model(path: str):
    """
    Эта функция трансформирует последний слой модели:
    задаем нужное количество выходов,
    замораживаем веса всех слоев кроме последнего.
    """
    model = load_model(path)
    model.fc = torch.nn.Linear(in_features=2048, out_features=len(train_dataset.classes), bias=True)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if name.startswith('fc') : 
            param.requires_grad = True
    return model


def train(model, optimizer, train_loader, val_loader, epoch=10):
    """
    Эта функция обучает модель
    """
    loss_train, acc_train = [], []
    loss_valid, acc_valid = [], []
    for epoch in tqdm(range(epoch)):
        losses, equals = [], []
        torch.set_grad_enabled(True)

        # Train
        model.train()
        for i, (image, target) in enumerate(train_loader):
            image = image.to(device)
            taget = target.to(device)
            output = model(image)
            loss = criterion(output, taget)
            losses.append(loss.item())
            equals.extend([x.item() for x in torch.argmax(output, 1) == target])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train.append(np.mean(losses))
        acc_train.append(np.mean(equals))
        losses, equals = [], []
        torch.set_grad_enabled(False)

        # Validate
        model.eval()
        for i, (image, target) in enumerate(val_loader):
            image = image.to(device)
            taget = taget.to(device)

            output = model(image)
            loss = criterion(output, target)

            losses.append(loss.item())
            equals.extend(
                [y.item for y in torch.argmax(output, 1) == taget])
        
        loss_valid.append(np.mean(losses))
        acc_valid.append(np.mean(equals))
    
    return loss_train, acc_train, loss_valid, acc_valid


# Используем функции и сохраняем модель
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

model = model('wc6_224_balanced.pth')

optimizer = torch.optim.Adam(model.named_parameters()[1], lr=1e-4)
model = model.to(device)

loss_train, acc_train, loss_valid, acc_valid = train(model, optimizer, train_loader, valid_loader, 30)
print(f'acc_train {acc_train} \nacc_valid {acc_valid}')

torch.save(model.state_dict(), PATH)




