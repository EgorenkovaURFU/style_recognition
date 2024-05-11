import streamlit as st

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import os


def picture_prepare(img_source):
    #img = Image.open(img_source).convert('RGB')
    img = img_source.convert('RGB')

    transform = transforms.Compose([
                #transforms.Resize(size=224, interpolation=InterpolationMode.BILINEAR),
                transforms.Resize(size=224),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                  mean=[0.49333772, 0.51176786, 0.51791704],
                  std=[0.26378724, 0.26562205, 0.3115852]
                  )
              ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    #return {'img': img, 'img_tensor': img_tensor}
    return img, img_tensor

def save_uploadedfile(uploadedfile, path):
    with open(os.path.join( path, uploadedfile.name), 'wb') as f:
        f.write(uploadedfile.getbuffer())
        return st.success('Файл сохранен'.format(uploadedfile.name))



def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return image

def load_file():
    uploaded_file = st.file_uploader(label='Выберите файл')

    if uploaded_file is not None:
        save_uploadedfile(uploaded_file, path)
        st.session_state.stage = 0
        return uploaded_file
    else:
        return None

def load_model(path: str):
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


#Основное тело программы

if 'stage' not in st.session_state:
    st.session_state.stage = 0

path = 'Pictures/'  # Папка с изображениями

if st.button('Добавить изображение в коллекцию из файла'):
    print(1)
    st.session_state.stage = 1

if st.session_state.stage == 1:
    load_file()
    #file = load_file()


if st.button('Проанализировать изображения'):
    image_list = []
    image_pr_list = []
    image_tensor_list = []
    prediction_list = []
    score_list = []

    model = load_model('models\wc6_224_balanced.pth')
    labels = 'lab.txt'



    for file in os.listdir(path):
        print(path, '   ', file)
        if file.endswith(".jpg"):
            print(type(file.endswith))
            image_loaded = load_image(path + file)
            st.image(image_loaded)
            image_list.append(image_loaded)

            image_pr, image_tensor = picture_prepare(image_loaded)
            #st.image(image_pr)
            image_pr_list.append(image_pr)
            image_tensor_list.append(image_tensor)

            prediction, score = predict(model, image_tensor, labels)
            st.success(prediction)
            st.success(score)
            prediction_list.append(prediction)
            score_list.append(score)

    print(prediction_list)
    print(score_list)