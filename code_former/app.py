import sys
import torchvision.transforms.functional as F
# ТОТ САМЫЙ ФИКС: подменяем отсутствующий модуль в новых версиях torchvision
sys.modules['torchvision.transforms.functional_tensor'] = F

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import shutil
import subprocess
import uuid

app = FastAPI()

INPUT_DIR = "temp_input"
OUTPUT_DIR = "results" # CodeFormer по дефолту пишет сюда

os.makedirs(INPUT_DIR, exist_ok=True)

@app.post("/process")
async def process_face(image: UploadFile = File(...), w: float = 0.6):
    task_id = str(uuid.uuid4())
    input_dir = f"temp/{task_id}/input"
    output_dir = f"temp/{task_id}/output"
    os.makedirs(input_dir, exist_ok=True)

    # 1. Сохраняем входящий файл
    input_path = os.path.join(input_dir, "to_fix.png")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 2. Запускаем команду (ту самую из Kaggle)
    # Важно: используем --face_upsample и -w
    command = [
        "python", "inference_codeformer.py",
        "-w", str(w),
        "--input_path", input_dir,
        "--output_path", output_dir,
        "--face_upsample"
    ]
    
    try:
        subprocess.run(command, check=True)
        
        # 3. Ищем результат (CodeFormer создает вложенные папки)
        # Обычно это: output/final_results/to_fix.png
        result_path = os.path.join(output_dir, "final_results", "to_fix.png")
        
        if os.path.exists(result_path):
            with open(result_path, "rb") as f:
                content = f.read()
            return Response(content=content, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        # Очистка мусора
        shutil.rmtree(f"temp/{task_id}", ignore_errors=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}