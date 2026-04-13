import os
import sys

def split_file(file_path, chunk_size_mb=90):
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден.")
        return

    chunk_size = chunk_size_mb * 1024 * 1024  # Переводим в байты
    file_name = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        chunk_num = 1
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            
            chunk_name = f"{file_name}.part{chunk_num}"
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(data)
            
            print(f"Создан: {chunk_name}")
            chunk_num += 1
    print("Готово! Теперь можешь пушить файлы .partX на GitHub.")

split_file("adapter_model.safetensors")
split_file("pure_lora_weights.pt")