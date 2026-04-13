import os
import glob
import sys

def merge_file(original_file_name):
    # Определяем абсолютный путь к папке, где лежит этот скрипт
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ищем части файлов именно в этой папке (core/lora/)
    search_pattern = os.path.join(base_dir, f"{original_file_name}.part*")
    parts = sorted(glob.glob(search_pattern), 
                   key=lambda x: int(x.split('.part')[-1]))
    
    if not parts:
        print(f"--- DEBUG ---")
        print(f"Current WD: {os.getcwd()}")
        print(f"Script dir: {base_dir}")
        print(f"Looking for: {search_pattern}")
        print(f"Files in dir: {os.listdir(base_dir)}")
        print(f"-------------")
        # Если файлов нет, не падаем тихо, а ругаемся
        raise FileNotFoundError(f"Части для {original_file_name} не найдены!")

    target_path = os.path.join(base_dir, original_file_name)
    print(f"Merging {len(parts)} parts into {target_path}...")
    
    with open(target_path, 'wb') as output_file:
        for part in parts:
            with open(part, 'rb') as f:
                output_file.write(f.read())
            os.remove(part)
            
    print(f"Done: {original_file_name}")

if __name__ == "__main__":
    try:
        merge_file("pure_lora_weights.pt")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) # Заставляем Docker остановить билд при ошибке