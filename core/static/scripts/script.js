document.addEventListener('DOMContentLoaded', () => {
    const genBtn = document.getElementById('generate_button');
    const resultSection = document.getElementById('result_section');
    const mainImg = document.getElementById('main_result');
    const viewBtns = document.querySelectorAll('.view-btn');

    let currentImages = {}; 
    let pollInterval;

    if ("Notification" in window) {
        Notification.requestPermission();
    }

    // Мапа для "человеческих" названий этапов
    const stageNames = {
        'base': 'Creating base...',
        'refining': 'Polishing details...',
        'face_fixing': 'Enhancing face & resolution...',
        'done': 'Masterpiece ready!'
    };

    genBtn.addEventListener('click', async () => {
        // Очищаем старые данные
        currentImages = {};
        
        const payload = {
            prompt: document.getElementById('user_prompt').value,
            steps: document.getElementById('steps').value,
            refine_steps: document.getElementById('refine_steps').value,
            guidance: document.getElementById('guidance_scale').value
        };

        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) return alert("System busy!");

        resultSection.style.display = 'block';
        genBtn.disabled = true;
        mainImg.style.opacity = '0.3';

        pollInterval = setInterval(async () => {
            const statusRes = await fetch('/api/status');
            const status = await statusRes.json();

            if (status.is_running || status.stage !== 'idle') {
                // Ограничиваем проценты максимум сотней на всякий случай
                let percent = Math.round((status.current_step / status.total_steps) * 100);
                if (percent > 100) percent = 100;
                
                const stageText = stageNames[status.stage] || 'Processing...';
                genBtn.innerText = `${percent}% — ${stageText}`;
            } 
            
            if (status.stage === 'done') {
                clearInterval(pollInterval);
                genBtn.disabled = false;
                genBtn.innerText = 'Generate Masterpiece';
                mainImg.style.opacity = '1';
                
                // Таймштамп, чтобы браузер не брал старую картинку из кэша
                const t = new Date().getTime();
                
                // Теперь у нас 3 разных файла
                currentImages = {
                    raw: `/static/results/last_raw.png?t=${t}`,
                    polished: `/static/results/last_refined.png?t=${t}`,
                    final: `/static/results/last_final.png?t=${t}` 
                };

                // По умолчанию показываем финальный результат
                mainImg.src = currentImages.final;
                
                // Активируем кнопку "FaceFix XL" визуально
                viewBtns.forEach(b => b.classList.remove('active'));
                document.querySelector('[data-type="final"]').classList.add('active');

                if (Notification.permission === "granted") {
                    new Notification("Lora Studio", {
                        body: "✨ Your masterpiece is ready!",
                        icon: "/static/images/1.png" // Или любой логотип
                    });
                }
            }
        }, 1000);
    });

    viewBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.getAttribute('data-type');
            if (!currentImages[type]) return; // Если еще не сгенерировано

            viewBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            mainImg.src = currentImages[type];
        });
    });
});

function updateWarnings() {
    const steps = document.getElementById('steps');
    const gs = document.getElementById('guidance_scale');
    const refine = document.getElementById('refine_steps');

    const stepsWarn = document.getElementById('steps_warning');
    const gsWarn = document.getElementById('gs_warning');
    const refineWarn = document.getElementById('refine_warning');

    // 1. Проверка Steps
    if (steps.value < 15) {
        showWarning(stepsWarn, "⚠️ Low quality expected");
    } else if (steps.value > 40) {
        showWarning(stepsWarn, "⏳ Will take longer to generate");
    } else {
        hideWarning(stepsWarn);
    }

    // 2. Проверка Guidance
    if (gs.value < 3) {
        showWarning(gsWarn, "⚠️ AI might ignore your prompt");
    } else if (gs.value > 10) {
        showWarning(gsWarn, "⚠️ High saturation/artifact risk");
    } else {
        hideWarning(gsWarn);
    }

    // 3. Проверка Refine Steps
    if (refine.value == 0) {
        showWarning(refineWarn, "⏩ Refinement will be skipped");
    } else if (refine.value > 0 && refine.value < 6) {
        showWarning(refineWarn, "⚠️ Too few steps for visible change");
    } else if (refine.value > 30) {
        showWarning(refineWarn, "⏳ Slow but high-detail mode");
    } else {
        hideWarning(refineWarn);
    }
}

function showWarning(el, text) {
    el.innerText = text;
    el.classList.add('visible');
}

function hideWarning(el) {
    el.classList.remove('visible');
}

// Вешаем обработчики
['steps', 'guidance_scale', 'refine_steps'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
        // Обновляем цифру рядом с лейблом
        document.getElementById(`${id}_val`).innerText = el.value;
        // Проверяем предупреждения
        updateWarnings();
    });
});

// Запускаем проверку один раз при загрузке
updateWarnings();