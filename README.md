# hackathon

# Запуск локально через терминал

В командной строке выполните следующие команды:

1. Перейти в директорию с проектом
   
cd /путь/к/вашему/проекту

2. Создать виртуальное окружение

python -m venv venv venv\Scripts\activate.bat

3. Установить зависимости
 
pip install -r requirements.txt

5. Запустить сервер

python app/api.py

---
# Структура

|-- app

|-- models

|-- SberAuto_2 (1).ipynb

|-- model.py

|-- requirements.txt


Файл model.py создает тестовую модель для загрузки.

В папке app находится сам сервер(app.py) и model_handler.py который загружает обученную модель.

В файле requirements.txt находятся зависимости проекта
