# 1. Базовый образ
FROM my-girl-base:latest

# 5. Слой зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка необходимых системных библиотек
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Аргумент для Hugging Face (если модель приватная, для SD 2.1 обычно не нужен, но оставим)
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# 6. СЛОЙ ЗАГРУЗКИ МОДЕЛИ (Запекаем SD 2.1-v в образ)
RUN echo "import os\n\
import torch\n\
from huggingface_hub import login\n\
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler\n\
\n\
token = os.getenv('HF_TOKEN')\n\
if token: login(token=token)\n\
\n\
model_id = 'sd2-community/stable-diffusion-2-1' # Официальный ID\n\
\n\
print('🚀 Скачиваю Stable Diffusion 2.1-v (768px)...')\n\
# Используем нужный model_id и параметры\n\
pipe = StableDiffusionPipeline.from_pretrained(\n\
    model_id,\n\
    torch_dtype=torch.float16,\n\
    variant='fp16',\n\
    use_safetensors=True\n\
)\n\
\n\
# КРИТИЧНО для версии 2.1-v: настраиваем планировщик\n\
pipe.scheduler = DPMSolverMultistepScheduler.from_config(\n\
    pipe.scheduler.config,\n\
    prediction_type='v_prediction'\n\
)\n\
\n\
pipe.save_pretrained('./sd21-768v-model')\n\
print('✅ Модель 2.1-v успешно сохранена!')" > download_model.py

# Запускаем загрузку
RUN python download_model.py && rm download_model.py

# Доустанавливаем специфичные библиотеки для LoRA и ускорения
RUN pip install --no-cache-dir peft==0.7.1 accelerate==0.25.0

# 7. Слой приложения
COPY . .

# Если вы используете LoRA, убедитесь, что merge_file.py 
# адаптирован под структуру SD 2.1 (OpenCLIP ViT-H/14)
RUN python3 lora/merge_file.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]