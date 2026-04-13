FROM my-girl-base:latest

# 2. Клонируем репозиторий (Кэшировано)
RUN git clone https://github.com/sczhou/CodeFormer.git .

# 3. Ставим зависимости (Кэшировано)
RUN pip install --no-cache-dir numpy==1.24.1
RUN SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p $SITE_PKG/basicsr && \
    echo '__version__ = "0.0.0"\n__gitsha__ = "unknown"' > $SITE_PKG/basicsr/version.py

# 4. Теперь ставим остальные требования из файла
RUN pip install --no-cache-dir "numpy<2.0" -r requirements.txt

# И твоя ручная загрузка из Kaggle
RUN mkdir -p weights/CodeFormer && \
    wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth \
    -O weights/CodeFormer/codeformer.pth

COPY . .

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]