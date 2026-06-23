# 🚀 MLOps Sentinel

**Production-ready MLOps pipeline for text classification with CI/CD, Docker, and monitoring**

[![CI/CD](https://github.com/KadmiIgro/MLOps-Sentinel-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/KadmiIgro/MLOps-Sentinel-v2/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker)](https://www.docker.com/)

---

## 📌 О проекте

MLOps Sentinel — это сквозной MLOps пайплайн для классификации текстов. Проект демонстрирует промышленный подход к разработке и эксплуатации ML-систем:

- ✅ Обучение модели DistilBERT на датасете твитов
- ✅ REST API на FastAPI с автоматической документацией
- ✅ Контейнеризация с Docker
- ✅ CI/CD пайплайн на GitHub Actions
- ✅ Мониторинг (Prometheus + Grafana)
- ✅ Версионирование модели (MLflow)

---

## 🛠 Технологический стек

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| **ML** | DistilBERT, Transformers | Классификация текстов |
| **API** | FastAPI, Uvicorn | REST API с Swagger |
| **Контейнеризация** | Docker, Docker Compose | Упаковка и запуск |
| **CI/CD** | GitHub Actions | Автоматическое тестирование и деплой |
| **Мониторинг** | Prometheus, Grafana | Сбор метрик и визуализация |
| **Версионирование** | MLflow | Логирование экспериментов |

---

## 🚀 Быстрый старт

### 1. Клонировать репозиторий
git clone https://github.com/KadmiIgro/MLOps-Sentinel-v2.git
cd MLOps-Sentinel-v2

### 2. Установить зависимости
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

pip install -r requirements.txt

### 3. Обучить модель (опционально)
python src/train.py

### 4. Запустить API локально
uvicorn src.app:app --reload

### 5. Запустить через Docker
docker-compose up -d --build

---

## 📖 API Документация

После запуска API доступны:

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| /docs | GET | Swagger UI с документацией |
| /health | GET | Проверка статуса сервиса |
| /predict | POST | Классификация текста |

### Пример запроса к /predict

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Fire in the building! Help!"}'

Ответ:
{
  "prediction": 1,
  "confidence": 0.96,
  "label": "disaster"
}

---

## 📊 Мониторинг

После запуска через Docker Compose доступны:

- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (логин: admin, пароль: admin)

---

## 🧪 Тестирование

pytest tests/ -v

---

## 📂 Структура проекта

MLOps-Sentinel/
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD пайплайн
├── src/
│   ├── app.py              # FastAPI приложение
│   ├── data_loader.py      # Загрузка данных
│   └── train.py            # Обучение модели
├── tests/
│   ├── test_api.py         # Тесты API
│   ├── test_data_loader.py # Тесты загрузки данных
│   └── test_model.py       # Тесты модели
├── data/
│   └── train.csv           # Датасет (не в репозитории)
├── models/                 # Сохранённая модель (не в репозитории)
├── config.yaml             # Конфигурация
├── Dockerfile              # Docker образ
├── docker-compose.yml      # Оркестрация
├── requirements.txt        # Зависимости
└── README.md               # Документация

---

## 🎯 Что дальше?

- [ ] Добавить Telegram бота
- [ ] Настроить автоскейлинг в Kubernetes
- [ ] Подключить S3 для хранения моделей
- [ ] Добавить A/B тестирование

---

## 📄 Лицензия

MIT © 2026

## ✉️ Контакты

**Автор:** [Kirill](https://github.com/KadmiIgro)

⭐ Если проект был полезен, поставь звезду на GitHub!
