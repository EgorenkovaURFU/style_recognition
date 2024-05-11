import torch
import torchvision


def load_model(path: str):
    """
    Эта функция загружает модель на базе которой будет проходить обучение
    """
    model = torchvision.models.resnet50(num_classes=6)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def predict(model, image, labels):
    model.eval()
    out = model(image)
    with open(labels) as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return (labels[index[0]], percentage[index[0]].item())