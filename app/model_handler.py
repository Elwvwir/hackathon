import pickle
import os

MODEL_PATH = "models/model_v1.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена по пути {MODEL_PATH}. Сначала запустите models/train_model.py")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Модель успешно загружена!")
    return model