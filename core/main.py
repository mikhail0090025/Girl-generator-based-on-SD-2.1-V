from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import sd

app = FastAPI()

# Монтируем статику (стили, картинки, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Указываем путь к папке с шаблонами
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Эндпоинт для отображения главной страницы.
    Jinja2 ищет файл index.html внутри папки templates.
    """
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={} # Если нужны доп. переменные, пиши их сюда
    )

@app.post("/api/generate")
async def start_generation(request: Request, background_tasks: BackgroundTasks):
    # Проверка: если GPU/CPU уже занят, не пускаем вторую задачу
    if sd.generation_status["is_running"]:
        return {"status": "error", "message": "System is busy, please wait..."}, 429

    try:
        data = await request.json()
        
        # Собираем параметры из запроса с дефолтными значениями
        prompt = data.get("prompt", "Lora, masterpiece, photorealistic")
        base_steps = int(data.get("steps", 25))
        refine_steps = int(data.get("refine_steps", 30))
        guidance = float(data.get("guidance", 7.5))
        # Можно даже fidelity для лица прокинуть, если захочешь
        fidelity = float(data.get("fidelity", 0.6)) 

        # Запускаем тяжелую задачу в фоновом потоке FastAPI
        background_tasks.add_task(
            sd.run_dual_pipeline, # Вызываем обновленный метод
            user_prompt=prompt,
            base_steps=base_steps,
            refine_steps=refine_steps,
            guidance=guidance
            # Если добавил fidelity в аргументы run_dual_pipeline, раскомментируй:
            # fidelity=fidelity 
        )

        return {"status": "started", "message": "Generation process initiated"}

    except Exception as e:
        return {"status": "error", "message": str(e)}, 400

@app.get("/api/status")
async def get_status():
    # Фронтенд будет стучаться сюда каждую секунду
    return sd.generation_status

@app.get("/health")
async def health_check():
    return {"status": "healthy"}