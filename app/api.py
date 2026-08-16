from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Путь к модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_v1.pkl')

# Загружаем модель при старте приложения
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Ошибка загрузки модели: {e}")

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности сервиса"""
    if MODEL_LOADED:
        return jsonify({'status': 'healthy', 'model_version': 'v1'}), 200
    else:
        return jsonify({'status': 'unhealthy', 'error': 'Model not loaded'}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Эндпоинт для предсказания.
    Ожидает JSON с ключом "features", содержащим список из 24 чисел.
    Пример:
    {
        "features": [20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]
    }
    Возвращает:
    {
        "prediction": 0 или 1,
        "probability": 0.75,
        "model_version": "v1"
    }
    """
    # Проверяем, загружена ли модель
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not available'}), 503

    # Получаем JSON из запроса
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    # Проверяем наличие ключа "features"
    if 'features' not in data:
        return jsonify({'error': 'Missing "features" key'}), 400

    features = data['features']

    # Проверяем, что это список и содержит ровно 24 числа
    if not isinstance(features, list):
        return jsonify({'error': '"features" must be a list'}), 400
    # if len(features) != 24:
    #     return jsonify({'error': f'Expected 24 features, got {len(features)}'}), 400

    # Преобразуем в numpy-массив и делаем reshape для одной выборки
    try:
        X = np.array(features, dtype=float).reshape(1, -1)
    except ValueError:
        return jsonify({'error': 'All features must be numbers'}), 400

    # Выполняем предсказание
    try:
        pred_class = int(model.predict(X)[0])
        pred_proba = float(model.predict_proba(X)[0][1])   # вероятность класса 1 (дефолт)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Возвращаем результат
    return jsonify({
        'prediction': pred_class,
        'probability': pred_proba,
        'model_version': 'v1'
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)




# from flask import Flask, request, jsonify
# import numpy as np
# import os
# from model_handler import load_model

# app = Flask(__name__)

# # Глобальная переменная для хранения модели
# model = None

# # --- ВАЖНЫЙ БЛОК ЗАГРУЗКИ МОДЕЛИ ---
# # Мы загружаем модель сразу при импорте этого модуля.
# # Это заменяет старый @app.before_first_request и работает во всех версиях Flask.
# try:
#     model = load_model()
#     print("Модель успешно загружена при старте приложения.")
# except Exception as e:
#     print(f"Критическая ошибка загрузки модели: {e}")
#     model = None

# # Список колонок в ТОЧНОМ порядке, как при обучении модели.
# # Исключены: 'ID' и целевая переменная 'default.payment.next.month'
# EXPECTED_COLUMNS = [
#     "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
#     "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
#     "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
#     "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
# ]

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Эндпоинт для предсказания дефолта"""
    
#     # Проверка: загружена ли модель вообще
#     if model is None:
#         return jsonify({'error': 'Model is not loaded. Check server logs.'}), 503

#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No JSON data provided in request body.'}), 400

#         # 1. Валидация наличия всех нужных колонок
#         missing_cols = [col for col in EXPECTED_COLUMNS if col not in data]
#         if missing_cols:
#             return jsonify({
#                 'error': f'Missing required features in JSON. Please provide all columns.',
#                 'missing_columns': missing_cols,
#                 'expected_columns': EXPECTED_COLUMNS
#             }), 400

#         # 2. Формирование массива СТРОГО в порядке EXPECTED_COLUMNS
#         # Это критически важно! Даже если JSON пришел в другом порядке, мы пересоберем его правильно.
#         features_list = [data[col] for col in EXPECTED_COLUMNS]
#         features = np.array(features_list).reshape(1, -1)

#         # 3. Предсказание
#         prediction = model.predict(features)
#         probability = model.predict_proba(features)

#         return jsonify({
#             'prediction': int(prediction),
#             'probability': float(probability),
#             'model_version': 'v1',
#             'status': 'success'
#         }), 200

#     except Exception as e:
#         # Ловим любые ошибки внутри логики и возвращаем понятный JSON
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health():
#     """Проверка здоровья сервиса"""
#     status = 'healthy' if model is not None else 'degraded'
#     return jsonify({
#         'status': status,
#         'message': 'Service running' if model is not None else 'Model failed to load'
#     }), 200

# if __name__ == '__main__':
#     # Запуск только для локальной отладки. 
#     # В Docker и на проде этот блок игнорируется (там используется gunicorn)
#     app.run(host='0.0.0.0', port=5000, debug=False)
