"""
Credit Behavior Engine — Feature Engineering
Transforma perfiles de MongoDB en features para Deep Learning.
Incluye codificación de variables categóricas y normalización.
"""

import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv
import pickle

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "credit_behavior"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES_NUMERICAS = [
    "total_ordenes",
    "ticket_promedio_historico",
    "ticket_maximo",
    "cuotas_maximas_historico",
    "uso_credito_historico",
    "score_promedio_historico",
    "reviews_negativas",
    "dias_entrega_promedio",
    "retrasos_promedio",
    "categorias_exploradas",
    "antiguedad_dias",
    "dias_desde_ultima_compra",
]

FEATURES_CATEGORICAS = ["region"]
TARGET = "propenso_credito"


# ── Carga desde MongoDB ────────────────────────────────────────────────────────
def cargar_desde_mongodb() -> pd.DataFrame:
    print("Cargando perfiles desde MongoDB...")
    client = MongoClient(MONGO_URI)  # nosec B101
    db = client[DB_NAME]
    registros = list(db["perfiles_clientes"].find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(registros)
    print(f"  ✅ {len(df):,} perfiles cargados")
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────
def construir_features(df: pd.DataFrame) -> tuple:
    """
    Construye features para Deep Learning.
    Retorna X_num (numéricas), X_cat (categóricas codificadas), y (target).
    """
    print("\nConstruyendo features...")

    # ── Features derivadas ─────────────────────────────────────────────────
    # Ratio flete/ticket — clientes con flete alto relativo son más sensibles al precio
    df["ratio_flete_ticket"] = (
        df.get("flete_total", pd.Series(0, index=df.index)) /
        df["ticket_promedio_historico"].clip(lower=1)
    ).fillna(0)

    # Frecuencia de compra — órdenes por mes activo
    df["frecuencia_compra"] = (
        df["total_ordenes"] /
        (df["antiguedad_dias"] / 30).clip(lower=1)
    ).fillna(0)

    # Score de comportamiento positivo
    df["score_comportamiento"] = (
        df["score_promedio_historico"] * 0.4 +
        (1 / df["retrasos_promedio"].clip(lower=1)) * 0.3 +
        df["frecuencia_compra"] * 0.3
    ).fillna(0)

    # Indicador de cliente recurrente
    df["es_recurrente"] = (df["total_ordenes"] > 1).astype(int)

    # Indicador de comprador de alto valor
    df["alto_valor"] = (
        df["ticket_maximo"] > df["ticket_promedio_historico"].median()
    ).astype(int)

    features_extra = [
        "ratio_flete_ticket",
        "frecuencia_compra",
        "score_comportamiento",
        "es_recurrente",
        "alto_valor",
    ]

    todas_numericas = FEATURES_NUMERICAS + features_extra

    # ── Normalización ──────────────────────────────────────────────────────
    X_num = df[todas_numericas].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    print(f"  ✅ Features numéricas: {X_num.shape[1]}")

    # ── Codificación categórica ────────────────────────────────────────────
    le = LabelEncoder()
    df["region_encoded"] = le.fit_transform(df["region"].fillna("Desconocido"))
    X_cat = df[["region_encoded"]].values.astype(np.int64)
    num_regiones = len(le.classes_)
    print(f"  ✅ Regiones codificadas: {le.classes_.tolist()}")

    # ── Target ────────────────────────────────────────────────────────────
    y = df[TARGET].values.astype(np.float32)
    positivos = int(y.sum())
    negativos = int((1 - y).sum())
    ratio = negativos / max(positivos, 1)
    print(f"  📊 Desbalance — positivos: {positivos:,} | negativos: {negativos:,} | ratio: {ratio:.1f}x")

    # ── Guardar scaler y encoder ───────────────────────────────────────────
    with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)  # nosec B301
    with open(f"{MODELS_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)  # nosec B301
    with open(f"{MODELS_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(todas_numericas, f)  # nosec B301

    print(f"  ✅ Scaler y encoder guardados en {MODELS_DIR}/")

    return X_num, X_cat, y, num_regiones, ratio


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Credit Behavior Engine — Feature Engineering")
    print("=" * 55)

    df = cargar_desde_mongodb()
    X_num, X_cat, y, num_regiones, ratio = construir_features(df)

    print(f"\n📐 Shape features numéricas: {X_num.shape}")
    print(f"📐 Shape features categóricas: {X_cat.shape}")
    print(f"📐 Shape target: {y.shape}")
    print(f"⚖️  Ratio desbalance: {ratio:.1f}x — se aplicará pos_weight en entrenamiento")
    print("\n✅ Features listas. Siguiente: python src/model.py")