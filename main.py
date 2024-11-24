from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели из pickle-файла
if not os.path.exists('best_model.pkl'):
    raise FileNotFoundError("Файл модели 'model.pkl' не найден. Убедитесь, что он находится в текущем каталоге.")
model = joblib.load('best_model.pkl')

medians = {'mileage': 19.3, 'engine': 1248.0, 'max_power': 82.0, 'seats': 5.0}

# Модели данных
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Обрабатывает словарь, выполняет предобработку данных и возвращает предсказание модели.
    """
    data = pd.DataFrame([item.dict()])
    prepared_data = preprocess_data(data)
    return float(model.predict(prepared_data)[0])


@app.post("/predict_items")
def predict_items(file: UploadFile):
    """
    Обрабатывает CSV-файл, выполняет предобработку данных и возвращает предсказания модели.
    """
    # Загрузка файла в DataFrame
    data = pd.read_csv(file.file)

    # Проверка необходимых колонок
    required_columns = Item.schema()["properties"].keys()
    if not set(required_columns).issubset(data.columns):
        raise HTTPException(status_code=400, detail="Некорректный формат данных в файле.")

    # Предобработка данных
    prepared_data = preprocess_data(data)

    # Предсказания
    predictions = model.predict(prepared_data)

    # Добавление предсказаний в DataFrame
    data['predicted_price'] = predictions

    # Возвращаем предсказания в виде списка
    return data.to_dict(orient='records')


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для подготовки данных перед предсказанием.
    Масштабированеи вещественных данных, one-hot encoding категориальных признаков.
    """
    # переводим вещественные данные из строки в число
    data['mileage'] = data['mileage'].str.split(' ').str[0]
    data['mileage'] = data['mileage'].astype(float)

    data['engine'] = data['engine'].str.split(' ').str[0]
    data['engine'] = data['engine'].astype(int)

    data['max_power'] = data['max_power'].str.split(' ').str[0]
    data['max_power'] = data['max_power'].astype(float)

    data = data.drop(columns=['torque', 'selling_price'])

    data['seats'] = data['seats'].astype(int)

    if len(data) == 1:
        # выделяем только вещественные данные
        df_wo_cat = data.drop(data.select_dtypes(include=['object', 'category']).columns, axis=1)

        # заполняем пропуски медианой
        for key in medians:
            df_wo_cat[key].fillna(medians[key], inplace=True)

        df_wo_cat.columns = [0, 1, 2, 3, 4, 5]

        df_cat = data[['fuel', 'seller_type', 'transmission', 'owner', 'seats']] # без name

        # загружаем encoder из pickle-файла
        if not os.path.exists('encoder.pkl'):
            raise FileNotFoundError("Файл 'encoder.pkl' не найден. Убедитесь, что он находится в текущем каталоге.")
        encoder = joblib.load('encoder.pkl')

        X_dum = encoder.transform(df_cat)

        X_dum_dense = X_dum.toarray()
        X_dum_df = pd.DataFrame(X_dum_dense, columns=encoder.get_feature_names_out(df_cat.columns))
        X_dum_df.index = df_cat.index

        X = pd.concat([df_wo_cat, X_dum_df], axis=1)
    else:
        # выделяем только вещественные данные
        df_wo_cat = data.drop(data.select_dtypes(include=['object', 'category']).columns, axis=1)

        # заполняем пропуски медианой
        for key in medians:
            df_wo_cat[key].fillna(medians[key], inplace=True)

        # загружаем scaler из pickle-файла
        if not os.path.exists('scaler.pkl'):
            raise FileNotFoundError("Файл 'scaler.pkl' не найден. Убедитесь, что он находится в текущем каталоге.")
        scaler = joblib.load('scaler.pkl')

        X_scal = scaler.transform(df_wo_cat)
        X_scal = pd.DataFrame(data=X_scal)

        df_cat = data[['fuel', 'seller_type', 'transmission', 'owner', 'seats']] # без name

        # загружаем encoder из pickle-файла
        if not os.path.exists('encoder.pkl'):
            raise FileNotFoundError("Файл 'encoder.pkl' не найден. Убедитесь, что он находится в текущем каталоге.")
        encoder = joblib.load('encoder.pkl')

        X_dum = encoder.transform(df_cat)

        X_dum_dense = X_dum.toarray()
        X_dum_df = pd.DataFrame(X_dum_dense, columns=encoder.get_feature_names_out(df_cat.columns))
        X_dum_df.index = df_cat.index

        X = pd.concat([X_scal, X_dum_df], axis=1)

    # для предупреждения ошибки приводим наименования колонок к строковому типу
    X.columns = X.columns.astype(str)

    return X
