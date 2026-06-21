"""
Credit Behavior Engine — Motor de Scoring
Convierte probabilidades del modelo en scores de negocio interpretables.
Capa simbólica: reglas de negocio sobre predicciones del modelo.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "credit_behavior"
MODELS_DIR = "models"
THRESHOLD = 0.3


# ── Arquitectura (debe coincidir con model.py) ─────────────────────────────────
class CreditBehaviorNet(nn.Module):
    def __init__(self, num_features: int, num_regiones: int, embedding_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(num_regiones, embedding_dim)
        input_dim = num_features + embedding_dim
        self.red = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_cat):
        emb = self.embedding(x_cat.squeeze(1))
        x = torch.cat([x_num, emb], dim=1)
        return self.red(x).squeeze(1)


# ── Carga de modelo y artefactos ───────────────────────────────────────────────
def cargar_modelo():
    with open(f"{MODELS_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # nosec B301
    with open(f"{MODELS_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)  # nosec B301
    with open(f"{MODELS_DIR}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)  # nosec B301

    num_features = len(feature_names)
    num_regiones = len(le.classes_)

    device = torch.device("cpu")
    model = CreditBehaviorNet(num_features, num_regiones)
    model.load_state_dict(
        torch.load(
            f"{MODELS_DIR}/credit_model_best.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model, scaler, le, feature_names, device


# ── Scoring de un cliente individual ──────────────────────────────────────────
def score_cliente(perfil: dict) -> dict:
    """
    Recibe perfil de cliente desde MongoDB y devuelve score crediticio.
    Combina probabilidad del modelo con reglas de negocio (capa simbólica).
    """
    model, scaler, le, feature_names, device = cargar_modelo()

    # Construir features igual que en training
    perfil["ratio_flete_ticket"] = 0.0
    perfil["frecuencia_compra"] = (
        perfil.get("total_ordenes", 1) /
        max(perfil.get("antiguedad_dias", 30) / 30, 1)
    )
    perfil["score_comportamiento"] = (
        perfil.get("score_promedio_historico", 3) * 0.4 +
        (1 / max(perfil.get("retrasos_promedio", 1), 1)) * 0.3 +
        perfil["frecuencia_compra"] * 0.3
    )
    perfil["es_recurrente"] = int(perfil.get("total_ordenes", 1) > 1)
    perfil["alto_valor"] = int(
        perfil.get("ticket_maximo", 0) > perfil.get("ticket_promedio_historico", 0)
    )

    X_num = scaler.transform(
        np.array([[perfil.get(f, 0) for f in feature_names]], dtype=np.float32)
    )

    region = perfil.get("region", "Desconocido")
    if region not in le.classes_:
        region = "Desconocido"
    region_enc = le.transform([region])[0]

    X_num_t = torch.tensor(X_num, dtype=torch.float32)
    X_cat_t = torch.tensor([[region_enc]], dtype=torch.long)

    with torch.no_grad():
        logit = model(X_num_t, X_cat_t)
        prob = torch.sigmoid(logit).item()

    # Score 0-1000 para negocio
    score = int(prob * 1000)

    # ── Capa simbólica: reglas de negocio ──────────────────────────────────
    if prob >= THRESHOLD:
        if perfil.get("reviews_negativas", 0) > 3:
            segmento = "REVISAR"
            motivo = "Alta propensión pero historial de insatisfacción"
        elif perfil.get("cuotas_maximas_historico", 0) >= 10:
            segmento = "PREMIUM"
            motivo = "Alta propensión y experiencia con crédito a largo plazo"
        else:
            segmento = "CANDIDATO"
            motivo = "Perfil con alta propensión a crédito"
    else:
        if perfil.get("total_ordenes", 0) >= 3 and prob > 0.15:
            segmento = "POTENCIAL"
            motivo = "Baja propensión actual pero cliente recurrente — seguimiento recomendado"
        else:
            segmento = "NO_CANDIDATO"
            motivo = "Sin señales suficientes de propensión a crédito"

    return {
        "customer_unique_id": perfil.get("customer_unique_id", "N/A"),
        "probabilidad": round(prob, 4),
        "score_credito": score,
        "segmento": segmento,
        "motivo": motivo,
        "region": region,
        "total_ordenes": perfil.get("total_ordenes", 0),
        "ticket_maximo": perfil.get("ticket_maximo", 0),
        "uso_credito_historico": perfil.get("uso_credito_historico", 0),
    }


# ── Scoring masivo desde MongoDB ───────────────────────────────────────────────
def score_batch(limite: int = 100) -> pd.DataFrame:
    """Puntúa un lote de clientes desde MongoDB."""
    client = MongoClient(MONGO_URI)  # nosec B101
    db = client[DB_NAME]
    registros = list(db["perfiles_clientes"].find({}, {"_id": 0}).limit(limite))
    client.close()

    resultados = []
    for perfil in registros:
        try:
            resultado = score_cliente(perfil)
            resultados.append(resultado)
        except Exception as e:
            continue

    df = pd.DataFrame(resultados)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Credit Behavior Engine — Motor de Scoring")
    print("=" * 55)

    print("\nEjemplo: cliente individual")
    perfil_ejemplo = {
        "customer_unique_id": "ejemplo_001",
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

    resultado = score_cliente(perfil_ejemplo)
    print(f"\nCliente: {resultado['customer_unique_id']}")
    print(f"Score crediticio: {resultado['score_credito']}/1000")
    print(f"Probabilidad:     {resultado['probabilidad']*100:.1f}%")
    print(f"Segmento:         {resultado['segmento']}")
    print(f"Motivo:           {resultado['motivo']}")

    print("\nScoring batch — primeros 200 clientes de MongoDB...")
    df_scores = score_batch(limite=200)
    print(f"\nDistribución de segmentos:")
    print(df_scores["segmento"].value_counts().to_string())
    print(f"\n✅ Scoring completo. Siguiente: api/main.py")