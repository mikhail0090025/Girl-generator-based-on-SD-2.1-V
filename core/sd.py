import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from peft import LoraConfig, get_peft_model
import os
import requests
import shutil

# Глобальный статус для API
generation_status = {
    "is_running": False,
    "current_step": 0,
    "total_steps": 0,
    "stage": "idle", # idle, base, refining, done
}

# 1. Определяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
# Путь к модели SD 2.1-v внутри Docker-контейнера
model_path = "./sd21-768v-model" 
my_lora_file = "./lora/pure_lora_weights.pt"

def load_pipeline():
    print(f"🖥️ Инициализация на устройстве: {device.upper()}")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Ошибка: Папка {model_path} не найдена в образе!")

    print(f"📦 Загрузка SD 2.1-v из локальной папки: {model_path}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        use_safetensors=True,
        safety_checker=None
    )
    
    # КРИТИЧНО для 2.1-v: Настройка v-prediction и планировщика
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        prediction_type="v_prediction"
    )
    
    pipe.to(device)
    
    if device == "cpu":
        pipe.enable_attention_slicing()
    
    print(f"✅ SD 2.1-v готова! Режим: v_prediction")
    return pipe

# Инициализируем пайплайн
try:
    pipeline = load_pipeline()
except Exception as e:
    print(f"🚨 Критическая ошибка при загрузке SD: {e}")
    pipeline = None

def load_my_lora(model, path, device="cuda"):
    if not os.path.exists(path):
        print("❌ Файл LoRA не найден!")
        return

    saved_state_dict = torch.load(path, map_location=device)
    current_model_dict = model.state_dict()
    cleaned_state_dict = {}
    
    for saved_key, saved_tensor in saved_state_dict.items():
        found = False
        for current_key in current_model_dict.keys():
            if current_key.endswith(saved_key.replace("base_model.model.", "")):
                cleaned_state_dict[current_key] = saved_tensor
                found = True
                break

    model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"🔥 Загружено {len(cleaned_state_dict)} тензоров LoRA")

# Настройка LoRA (аналогично ноутбуку обучения)
lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=[
        "to_q", "to_v", "to_k", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ],
    lora_dropout=0.05,
    bias="none",
)

if pipeline:
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    load_my_lora(pipeline.unet, my_lora_file, device=device)

def progress_callback(pipe, step_index, timestep, callback_kwargs):
    global generation_status
    generation_status["current_step"] += 1
    return callback_kwargs

negative_prompt = (
    "cartoon, illustration, painting, drawing, blurry, "
    "smooth skin, plastic, low quality, distorted face, bad anatomy"
)

additional_prompt = "portrait photography, high resolution, masterpiece, best quality, photo-realistic, 8k uhd"

def modify_prompt(base_prompt, additional=""):
    # Заменяем "Lora" на "sks_girl" (как в обучении SD 2.1)
    base_prompt = base_prompt.replace("Lora", "sks_girl")
    if not additional.strip():
        return base_prompt
    return f"{base_prompt}, {additional}"

def generate_image(user_prompt, negative_prompt=negative_prompt, num_inference_steps=25, guidance_scale=7.5):
    prompt = modify_prompt(user_prompt, additional_prompt)
    with torch.inference_mode():
        image = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=768, # SD 2.1-v требует 768
            width=768
        ).images[0]
    return image

def get_refine_pipeline():
    if not pipeline: return None
    print("🧼 Инициализация Refine Pipeline (Img2Img)...")
    refine_pipe = StableDiffusionImg2ImgPipeline(
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device)
    return refine_pipe

refine_pipeline = get_refine_pipeline()

def refine_image(image, prompt, negative_prompt=negative_prompt, steps=30, strength=0.3, guidance=7.5):
    if not refine_pipeline: return image
    actual_steps = int(steps / strength)
    final_prompt = modify_prompt(prompt, additional_prompt)
    
    with torch.inference_mode():
        refined_img = refine_pipeline(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=image.convert("RGB"),
            strength=strength,
            num_inference_steps=actual_steps,
            guidance_scale=guidance,
        ).images[0]
    return refined_img

def request_face_fix(image_path, fidelity=0.6):
    global generation_status
    generation_status["stage"] = "codeformer"
    print("🧠 Отправка на CodeFormer...")
    url = f"http://code_former:8001/process?w={fidelity}"
    try:
        with open(image_path, "rb") as f:
            files = {"image": ("img.png", f, "image/png")}
            response = requests.post(url, files=files, timeout=120)
        if response.status_code == 200:
            final_path = image_path.replace("_refined.png", "_final.png")
            with open(final_path, "wb") as f:
                f.write(response.content)
            return final_path
        else:
            return image_path
    except Exception as e:
        print(f"🚨 Ошибка связи с CodeFormer: {e}")
        return image_path

def run_dual_pipeline(user_prompt, base_steps=25, refine_steps=30, strength=0.3, guidance=7.5):
    global generation_status
    base_steps, refine_steps = int(base_steps), int(refine_steps)
    total_expected = base_steps + refine_steps + 1
    
    generation_status.update({
        "is_running": True,
        "current_step": 0,
        "total_steps": total_expected,
        "stage": "base"
    })
    
    try:
        # --- ЭТАП 1: ГЕНЕРАЦИЯ ---
        prompt = modify_prompt(user_prompt, additional_prompt)
        with torch.inference_mode():
            raw_image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=base_steps,
                guidance_scale=guidance,
                height=768,
                width=768,
                callback_on_step_end=progress_callback 
            ).images[0]
        
        os.makedirs("static/results", exist_ok=True)
        raw_path = "static/results/last_raw.png"
        raw_image.save(raw_path)
        refined_path = "static/results/last_refined.png"

        # --- ЭТАП 2: ШЛИФОВКА ---
        if refine_steps > 0:
            generation_status["stage"] = "refining"
            safe_strength = max(strength, 0.01)
            actual_refine_steps = int(refine_steps / safe_strength)
            
            with torch.inference_mode():
                refined_image = refine_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=raw_image.convert("RGB"),
                    strength=safe_strength,
                    num_inference_steps=actual_refine_steps,
                    guidance_scale=guidance,
                    callback_on_step_end=progress_callback
                ).images[0]
            refined_image.save(refined_path)
        else:
            shutil.copyfile(raw_path, refined_path)
            generation_status["current_step"] += refine_steps

        # --- ЭТАП 3: FACE FIX ---
        generation_status["stage"] = "face_fixing" 
        generation_status["current_step"] += 1
        final_img_path = request_face_fix(refined_path, fidelity=0.6)

        generation_status["stage"] = "done"
        return {
            "raw": f"/{raw_path}",
            "polished": f"/{refined_path}",
            "final": f"/{final_img_path}"
        }

    except Exception as e:
        print(f"🚨 Ошибка в пайплайне: {e}")
        generation_status["stage"] = "error"
        return None
    finally:
        generation_status["is_running"] = False