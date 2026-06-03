import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

class LocalSearchEngine:
    def __init__(self, vector_path, metadata_path):
        # 1. Загружаем модель (ту же, что и при векторизации)
        print("Загрузка модели поиска...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 2. Загружаем векторы и метаданные
        print("Загрузка базы знаний...")
        self.embeddings = np.load(vector_path).astype('float32')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # Создаем индекс FAISS для быстрого поиска по косинусному сходству
        if self.embeddings.ndim != 2:
            raise ValueError(f"Ожидаются матрица эмбеддингов, получено: shape={self.embeddings.shape}")
        dimension = self.embeddings.shape[1]
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        print(f"Поисковый движок готов. Проиндексировано фрагментов: {len(self.metadata)}")

    def search(self, query, top_k=3):
        """Поиск наиболее похожих фрагментов документации"""
        # Превращаем запрос пользователя в вектор
        query_vector = self.model.encode([query]).astype('float32')
        
        # Ищем ближайшие векторы в индексе
        faiss.normalize_L2(query_vector)
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        # FAISS возвращает матрицу shape=(num_queries, top_k)
        for rank, idx in enumerate(indices[0]):
            if idx != -1:
                similarity = float(similarities[0][rank])
                similarity_percent = max(0.0, min(100.0, (similarity + 1.0) * 50.0))
                results.append({
                    "similarity": similarity_percent,
                    "source": self.metadata[idx]['source'],
                    "content": self.metadata[idx]['original_content']
                })
        return results

# --- Пример использования ---
if __name__ == "__main__":
    # Указываем пути к файлам из Шага 2
    engine = LocalSearchEngine("embeddings.npy", "metadata.json")
    
    while True:
        user_query = input("\nВведите поисковый запрос (или 'exit' для выхода): ")
        if user_query.lower() == 'exit':
            break
            
        matches = engine.search(user_query)
        
        print("\n=== Результаты поиска ===")
        for match in matches:
            print(f"\n[Источник: {match['source']}] (Сходство: {round(match['similarity'], 2)}%)")
            # Выводим первые 300 символов для предпросмотра
            print(f"Текст: {match['content'][:300]}...")