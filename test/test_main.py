from streamlit.testing.v1 import AppTest


# проверка запуска приложения
def test_start_app():
    at = AppTest.from_file("src/main.py", default_timeout=30).run()
    assert not at.exception


def test_button_app():
    at = AppTest.from_file("src/main.py").run()
    at.button[0].click().run()
    at.button[1].click().run()
    assert at.button[0].label == "Добавить изображение в коллекцию из файла"
    assert at.button[1].label == "Проанализировать изображения"
    assert not at.exception
