import wget
from sklearn.model_selection import train_test_split


#TODO добавить ссылки для скачивания данных, изменить имена переменных   
raw_data_1 = '' # link
raw_data_2 = '' # link

# это мб и не надо
style_list = [
    ['fisrt style', '<path.png>'],
    ['second style', '<path.png>'],
]

# загрузить данные 
link_list = [raw_data_1, raw_data_2]
for link in link_list:
    wget.download(link, 'data/')

#TODO как-то разделить данные и сохранить по папкам
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
