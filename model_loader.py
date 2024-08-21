import pickle
from typing import Any

class ModelDecorator:
    def __init__(self, model: Any, predict_method: str):
        self.model = model
        self.predict_method = predict_method
        
    def predict(self, X):
        # Usando getattr para chamar o método de predição especificado
        predict_function = getattr(self.model, self.predict_method)
        return predict_function(X)


class ModelLoader:
    def load_model(self, model_path: str, predict_method: str = 'predict') -> ModelDecorator:
        # Carrega o modelo de um arquivo usando pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Retorna o modelo decorado
        return ModelDecorator(model, predict_method)


if __name__ == '__main__':
    model_path = 'random_forest_model.pkl'
    model_loader = ModelLoader()
    model = model_loader.load_model(model_path=model_path, predict_method='predict')
