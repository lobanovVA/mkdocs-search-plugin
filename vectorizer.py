import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def create_embeddings(input_json_path, output_vector_path, output_metadata_path):
    # 1. Загружаем обработанные данные из Шага 1
    print("Загрузка обработанных данных...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Инициализируем модель (локально)
    print("Загрузка локальной ИИ-модели (BERT-based)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 3. Извлекаем тексты для векторизации
    # Мы векторизуем именно 'processed_content', где слова в начальной форме
    texts = [doc['processed_content'] for doc in documents]

    print(f"Начинаю векторизацию {len(texts)} фрагментов. Это может занять время...")
    
    # 4. Генерируем эмбеддинги (векторы)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # 5. Сохраняем векторы в бинарный файл (для скорости)
    np.save(output_vector_path, embeddings)

    # 6. Сохраняем метаданные (пути к файлам и оригинал текста) отдельно
    # Это нужно, чтобы ИИ знал, какой текст соответствует какому вектору
    metadata = []
    for i, doc in enumerate(documents):
        metadata.append({
            "id": i,
            "source": doc['source'],
            "original_content": doc['original_content']
        })
    
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"\nГотово! Векторы сохранены в: {output_vector_path}")
    print(f"Метаданные сохранены в: {output_metadata_path}")

if __name__ == "__main__":
    create_embeddings(
        input_json_path="processed_docs.json", # Файл из Шага 1
        output_vector_path="embeddings.npy", 
        output_metadata_path="metadata.json"
    )