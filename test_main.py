from fastapi.testclient import TestClient
from main import app, load_model

client = TestClient(app)

def test_read_main():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Welcome!'}


def test_predict():
    response = client.post('/prediction/',
                           json={"url": "https://rustytraveltrunk.com/wp-content/uploads/2016/03/paris-churches-header-1280x640.jpg"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['prediction'] == 'Gothic'


def test_load_model():
    model = load_model('wc6_224_balanced.pth')
    assert str(type(model)) == "<class 'torchvision.models.resnet.ResNet'>"

