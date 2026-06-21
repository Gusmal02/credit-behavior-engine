"""
Credit Behavior Engine — API REST
Endpoints para scoring crediticio individual y batch.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.scoring import score_cliente, score_batch, cargar_modelo

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "credit_behavior"

app = FastAPI(
    title="Credit Behavior Engine",
    description="""
    Motor de scoring crediticio conductual para servicios financieros.
    Combina Deep Learning (Embeddings + Red Neuronal) con reglas de negocio
    para identificar clientes con alta propensión a productos de crédito.
    
    Desarrollado sobre datos reales de e-commerce (Olist Brasil — 96,096 clientes).
    Aplicable a procesos de originación de crédito en instituciones financieras.
    """,
    version="1.0.0",
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class PerfilCliente(BaseModel):
    customer_unique_id: str
    total_ordenes: int = 1
    ticket_promedio_historico: float = 0.0
    ticket_maximo: float = 0.0
    cuotas_maximas_historico: float = 0.0
    uso_credito_historico: float = 0.0
    score_promedio_historico: float = 3.0
    reviews_negativas: float = 0.0
    dias_entrega_promedio: float = 0.0
    retrasos_promedio: float = 0.0
    categorias_exploradas: float = 0.0
    antiguedad_dias: float = 0.0
    dias_desde_ultima_compra: float = 0.0
    region: str = "Desconocido"

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_unique_id": "cliente_001",
                "total_ordenes": 4,
                "ticket_promedio_historico": 250.0,
                "ticket_maximo": 800.0,
                "cuotas_maximas_historico": 12.0,
                "uso_credito_historico": 1.0,
                "score_promedio_historico": 4.5,
                "reviews_negativas": 0.0,
                "dias_entrega_promedio": 7.0,
                "retrasos_promedio": 0.0,
                "categorias_exploradas": 5.0,
                "antiguedad_dias": 365.0,
                "dias_desde_ultima_compra": 30.0,
                "region": "Sudeste",
            }
        }
    }


class ScoreResponse(BaseModel):
    customer_unique_id: str
    probabilidad: float
    score_credito: int
    segmento: str
    motivo: str
    region: str
    total_ordenes: int
    ticket_maximo: float
    uso_credito_historico: float


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "proyecto": "Credit Behavior Engine",
        "version": "1.0.0",
        "descripcion": "Motor de scoring crediticio conductual",
        "modelo": "Embedding + Dense Network (PyTorch)",
        "datos": "Olist Brasil — 96,096 clientes reales",
        "auroc": 0.9958,
        "auprc": 0.6024,
        "endpoints": ["/score", "/score/batch", "/health", "/docs"],
    }


@app.post("/score", response_model=ScoreResponse)
def score_individual(cliente: PerfilCliente):
    """
    Scoring crediticio para un cliente individual.
    Devuelve probabilidad, score 0-1000 y segmento de negocio.
    """
    try:
        perfil = cliente.model_dump()
        resultado = score_cliente(perfil)
        return ScoreResponse(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/score/batch")
def score_batch_endpoint(limite: int = 50):
    """
    Scoring batch de clientes desde MongoDB.
    Útil para campañas de originación de crédito masivas.
    """
    if limite > 500:
        raise HTTPException(
            status_code=400,
            detail="Límite máximo 500 clientes por request"
        )
    try:
        df = score_batch(limite=limite)
        distribucion = df["segmento"].value_counts().to_dict()
        candidatos = df[df["segmento"].isin(["CANDIDATO", "PREMIUM"])]
        return {
            "total_evaluados": len(df),
            "distribucion_segmentos": distribucion,
            "candidatos_credito": len(candidatos),
            "tasa_conversion": round(len(candidatos) / max(len(df), 1), 4),
            "top_candidatos": candidatos.head(10).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)  # nosec B101
        client.server_info()
        client.close()
        mongo_status = "ok"
    except Exception:
        mongo_status = "no disponible"

    return {
        "status": "ok",
        "modelo": "cargado",
        "mongodb": mongo_status,
    }