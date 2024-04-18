from fastapi import FastAPI, HTTPException, Query, Depends
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    """
    Представляет элемент ввода для анализа тональности.

    Атрибуты:
        text (str): Текст, который будет анализироваться на тональность.
    """
    text: str

def get_classifier():
    """
    Инициализация конвейера анализа тональности.
    """
    return pipeline("sentiment-analysis")

@app.post("/predict/")
def predict(item: Item, classifier=Depends(get_classifier)):
    """
    Конечная точка для предсказания анализа тональности для заданного текста.

    Аргументы:
        item (Item): Элемент ввода, содержащий текст для анализа.
        classifier (transformers.Pipeline): Конвейер анализа тональности.

    Возвращает:
        dict: Результат анализа тональности.
    """
    if not item.text.strip():
        raise HTTPException(status_code=400, detail="Входной текст не может быть пустым или содержать только пробелы.")
    try:
        result = classifier(item.text)[0]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
