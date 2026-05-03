import os
import re
import json
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy3

# Загрузка необходимых ресурсов NLTK (выполняется один раз)
nltk.download('punkt')
nltk.download('stopwords')

class DocumentationPreprocessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        # Регулярное выражение для очистки Markdown разметки
        self.md_cleaner = re.compile(r'(\*\*|__|~~|`|#|\[|\]|\(|\)|!|>|\||-)')

    def clean_markdown(self, text):
        """Очистка текста от символов Markdown разметки [2]"""
        # Удаляем ссылки [текст](ссылка) -> оставляем только текст
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Удаляем специфические символы разметки
        text = self.md_cleaner.sub('', text)
        # Удаляем лишние пробелы и пустые строки
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_text(self, text):
        """Токенизация и лемматизация текста [1]"""
        # Приводим к нижнему регистру и токенизируем
        tokens = word_tokenize(text.lower())
        
        processed_tokens = []
        for token in tokens:
            # Оставляем только буквы и цифры, убираем стоп-слова
            if token.isalnum() and token not in self.stop_words:
                # Приводим слово к нормальной форме (лемматизация) [1]
                parsed = self.morph.parse(token)
                if parsed:
                    lemma = parsed[0].normal_form
                else:
                    lemma = token
                processed_tokens.append(lemma)
        
        return " ".join(processed_tokens)

    def scan_directory(self, root_path):
        """Рекурсивный обход папок и сбор данных [2]"""
        extracted_data = []
        
        # Используем rglob для поиска всех .md файлов на любой глубине
        for md_file in Path(root_path).rglob('*.md'):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 1. Очистка от разметки
                    clean_text = self.clean_markdown(content)
                    
                    # 2. Лемматизация и токенизация
                    nlp_ready_text = self.process_text(clean_text)
                    
                    extracted_data.append({
                        "source": str(md_file.relative_to(root_path)),
                        "original_content": content,
                        "processed_content": nlp_ready_text
                    })
                    print(f"Обработан файл: {md_file}")
            except Exception as e:
                print(f"Ошибка при обработке {md_file}: {e}")
        
        return extracted_data

# --- Запуск ---
if __name__ == "__main__":
    # Укажите путь к папке с вашей документацией
    input_dir = "./docs" 
    output_file = "processed_docs.json"

    if not os.path.exists(input_dir):
        print(f"Ошибка: Папка {input_dir} не найдена.")
    else:
        preprocessor = DocumentationPreprocessor()
        print("Начинаю сбор и обработку данных...")
        
        data = preprocessor.scan_directory(input_dir)
        
        # Сохраняем результат в JSON для следующего этапа (векторизации)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"\nГотово! Обработано файлов: {len(data)}")
        print(f"Результат сохранен в: {output_file}")