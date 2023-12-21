import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import requests
import numpy as np
import uvicorn


class Item(BaseModel):
    url: HttpUrl


app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Welcome!'}


def load_model(path: str):
  model = torchvision.models.resnet50(num_classes=6)
  model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  return model


def save_image(item : Item):
    img_data = requests.get(item.url).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)


def picture_prepare(picture_path):
  img = Image.open(picture_path).convert('RGB')

  transform = transforms.Compose([
              transforms.Resize(size=224, interpolation=Image.BILINEAR),
              transforms.CenterCrop(size=(224, 224)),
              transforms.ToTensor(),
              transforms.Normalize(
                  mean=[0.49333772, 0.51176786, 0.51791704],
                  std=[0.26378724, 0.26562205, 0.3115852]
                  )
              ])

  img = transform(img)
  img = torch.unsqueeze(img, 0)
  return img


def predict(model, image, labels):
  model.eval()
  out = model(image)
  with open(labels) as f:
      labels = [line.strip() for line in f.readlines()]

  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
  return(labels[index[0]], percentage[index[0]].item())

# 
# if __name__=="__main__":
#     path = 'wc6_224_balanced.pth'
#     picture_path = "pictures/Q1389_wd0.jpg"
#     labels = 'lab.txt'
#     image = picture_prepare(picture_path)
#     model = load_model(path)
#     print(predict(model, image, labels))



@app.post("/prediction/")
async def get_net_image_prediction(item: Item):
  if item.url == "":
    print(item.url)
    return {"message": "No image link provided"}
    
  save_image(item)

  image = picture_prepare('image_name.jpg')
  model = load_model('wc6_224_balanced.pth')
  print(f'--------------{type(model)}----------------')
  labels = 'lab.txt'
  prediction, score = predict(model, image, labels)

  return {'prediction': prediction, 'score': score}


if __name__ == "__main__":
  uvicorn.run("main:app", host="127.0.0.1", port=5000)

