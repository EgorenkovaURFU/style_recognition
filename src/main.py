import streamlit as st
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import base64
import os
import uuid


def picture_prepare(img_source):
    img = img_source.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49333772, 0.51176786, 0.51791704],
            std=[0.26378724, 0.26562205, 0.3115852],
        ),
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img, img_tensor


def save_uploadedfile(uploadedfile, path):
    with open(os.path.join(path, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        return st.success("Файл {0} сохранен".format(uploadedfile.name))


def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return image


def load_file():
    uploaded_file = st.file_uploader(label="Выберите файл",
                                     type=["jpg", "jpeg"])
    if uploaded_file is not None:
        save_uploadedfile(uploaded_file, path)
        st.session_state.stage = 0
        return uploaded_file
    else:
        return None


def load_model(path: str):
    model = torchvision.models.resnet50(num_classes=6)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model


def predict(model, image, labels):
    model.eval()
    out = model(image)
    with open(labels) as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return (labels[index[0]], percentage[index[0]].item())


# Блок для обработки данных для вставки в датафрейм
def get_thumbnail(i):
    i.thumbnail((50, 50), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, "jpeg")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}" alt="" width = "150"height = "100">'


def text_formatter(tm):
    return f"<p>{tm}</p>"


# Основное тело программы
# Create interface
# Настройка боковой панели
st.sidebar.title("Это программа определения архитектурного стиля здания")
st.sidebar.info("В основе программы модель Resnet50 от TorchVision. "
                "Интерфейс реализован на Streamlit")

# Main window
st.image(Image.open("src/Logo2.jpeg"),
         caption="Achitecture style timeline",
         width=600)
st.header("Программа сделана в рамках проектного курса MLops")
st.markdown(
    "Для работы вы можете сформировать набор изображений зданий и "
    "далее определить архитектурный стиль по собранному набору изображений"
)

# Process part
if "stage" not in st.session_state:
    st.session_state.stage = 0

path = "Pictures/"  # Папка с изображениями

if st.sidebar.button("Добавить изображение в коллекцию из файла"):
    print(1)
    st.session_state.stage = 1

if st.session_state.stage == 1:
    load_file()

if st.sidebar.button("Проанализировать изображения"):
    image_list = []
    image_pr_list = []
    image_tensor_list = []
    prediction_list = []
    score_list = []
    df = pd.DataFrame(columns=["Image_building", "Architecture_type", "Score"])

    model = load_model("models/wc6_224_balanced.pth")
    labels = "src/lab.txt"

    for file in os.listdir(path):
        print("Файл (1): ", file)
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            print("Файл (2): ", file)
            image_loaded = load_image(path + file)

            image_list.append(image_loaded)
            image_pr, image_tensor = picture_prepare(image_loaded)
            image_pr_list.append(image_pr)
            image_tensor_list.append(image_tensor)

            prediction, score = predict(model, image_tensor, labels)
            prediction_list.append(prediction)
            score_list.append(score)

            # print('Файл (3): ',file,' Стиль: ',
            # prediction,' Вероятность: ', score)
            df.loc[len(
                df.index)] = [image_pr, prediction,
                              str(int(score)) + "%"]

    df.reset_index(drop=True, inplace=True)

    df = df.to_html(
        formatters={
            "Image_building": image_formatter,
            "Architecture_type": text_formatter,
            "Score": text_formatter,
        },
        escape=False,
        classes="table table-striped",
    )
    st.write(df, unsafe_allow_html=True)

    from itertools import zip_longest


#    def grouper(df, n, fillvalue=None):
       # args = [iter(df)] * n
      #  return zip_longest(*args, fillvalue=fillvalue)

#    for g in grouper(range(1, 51), 10):
      #  a = print(g, 'data{}.txt'.format(g[-1]))

    with open('result/' + str(uuid.uuid4()), "w") as file1:
        file1.write(df)
