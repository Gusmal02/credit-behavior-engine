# Credit Behavior Engine 🏦🧠

[![DevSecOps Pipeline](https://github.com/Gusmal02/credit-behavior-engine/actions/workflows/devsecops_pipeline.yml/badge.svg)](https://github.com/Gusmal02/credit-behavior-engine/actions)

Motor de scoring crediticio conductual basado en Deep Learning, construido sobre datos reales de e-commerce (Olist Brasil — 96,096 clientes). Identifica clientes con alta propensión a productos de crédito combinando redes neuronales con reglas de negocio.

## Problema que resuelve

Las instituciones financieras necesitan identificar clientes con propensión a crédito antes de lanzar campañas de originación. Los modelos tradicionales usan solo datos de buró — este motor usa **comportamiento de compra real** como señal predictiva, aplicable a cualquier empresa con historial transaccional.

## Arquitectura

Datos reales Olist Brasil (datos sucios)

↓

Ingesta + Limpieza → MongoDB

↓

Feature Engineering (17 features conductuales)

↓

┌─────────────────────────────────────┐

│  CAPA NEURONAL                      │

│  Embedding geográfico (6 regiones)  │

│  + Dense Network 128→64→32→1        │

│  AUROC: 0.9958 / AUPRC: 0.6024     │

└─────────────────┬───────────────────┘

↓

┌─────────────────────────────────────┐

│  CAPA SIMBÓLICA                     │

│  Motor de reglas de negocio         │

│  PREMIUM / CANDIDATO / POTENCIAL /  │

│  REVISAR / NO_CANDIDATO             │

└─────────────────────────────────────┘

↓

API REST → Score 0-1000 por cliente

## Métricas

| Métrica | Valor |
|---|---|
| AUROC | 0.9958 |
| AUPRC | 0.6024 |
| Recall clase positiva | 100% |
| Desbalance manejado | 146x |
| Clientes evaluados | 96,096 |

## Stack

| Componente | Tecnología |
|---|---|
| Deep Learning | PyTorch — Embeddings + Dense Network |
| Base de datos | MongoDB (perfiles conductuales) |
| API REST | FastAPI + Uvicorn |
| Tracking | MLflow |
| Contenedorización | Docker + Docker Compose |
| CI/CD + SAST | GitHub Actions + Bandit |
| Gestión entorno | uv (Astral) |
| Testing | pytest (10 pruebas) |

## Instalación

```bash
git clone https://github.com/Gusmal02/credit-behavior-engine
cd credit-behavior-engine
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Descarga el dataset desde [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) y colócalo en `data/`.

## Ejecución

```bash
# 1. Ingesta y limpieza → MongoDB
python src/ingesta.py

# 2. Feature engineering
python src/features.py

# 3. Entrenamiento
python src/model.py

# 4. API
uvicorn api.main:app --reload

# 5. MLflow UI
mlflow ui
```

## Docker

```bash
docker-compose up --build
```

## Tests

```bash
pytest tests/ -v
# 10 passed
```

## Roadmap

- [ ] n8n workflow para ingesta automática
- [ ] Terraform + GCP para despliegue cloud
- [ ] Monitoreo de drift del modelo
- [ ] Dashboard de campañas en Grafana