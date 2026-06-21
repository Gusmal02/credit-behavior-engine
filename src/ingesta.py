"""
Credit Behavior Engine — Ingesta y limpieza de datos Olist
Combina 8 datasets reales con datos sucios para construir
perfiles de comportamiento crediticio por cliente.
"""

import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "credit_behavior"


# ── Carga de datasets ──────────────────────────────────────────────────────────
def cargar_datasets() -> dict:
    print("Cargando datasets Olist...")
    archivos = {
        "customers":    "olist_customers_dataset.csv",
        "orders":       "olist_orders_dataset.csv",
        "items":        "olist_order_items_dataset.csv",
        "payments":     "olist_order_payments_dataset.csv",
        "reviews":      "olist_order_reviews_dataset.csv",
        "products":     "olist_products_dataset.csv",
        "sellers":      "olist_sellers_dataset.csv",
        "geo":          "olist_geolocation_dataset.csv",
        "translation":  "product_category_name_translation.csv",
    }
    dfs = {}
    for nombre, archivo in archivos.items():
        ruta = os.path.join(DATA_DIR, archivo)
        dfs[nombre] = pd.read_csv(ruta)
        print(f"  ✅ {nombre}: {len(dfs[nombre]):,} registros")
    return dfs


# ── Limpieza de datos sucios ───────────────────────────────────────────────────
def limpiar_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza órdenes — datos sucios reales de Olist."""
    df = df.copy()

    # Convertir fechas — vienen como string con formatos inconsistentes
    cols_fecha = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in cols_fecha:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Datos sucios: órdenes sin fecha de aprobación (~160 registros)
    nulos_aprobacion = df["order_approved_at"].isna().sum()
    print(f"  ⚠️  Órdenes sin fecha aprobación: {nulos_aprobacion} — imputando con purchase_timestamp")
    df["order_approved_at"] = df["order_approved_at"].fillna(df["order_purchase_timestamp"])

    # Datos sucios: entregas sin fecha real (~2,965 registros cancelados/pendientes)
    df["order_delivered_customer_date"] = df["order_delivered_customer_date"].fillna(pd.NaT)

    # Calcular días de entrega real
    df["dias_entrega"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Datos sucios: días negativos por errores de registro
    negativos = (df["dias_entrega"] < 0).sum()
    if negativos > 0:
        print(f"  ⚠️  Días de entrega negativos: {negativos} — corrigiendo a NaN")
        df.loc[df["dias_entrega"] < 0, "dias_entrega"] = np.nan

    # Retraso en entrega vs estimado
    df["dias_retraso"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days.clip(lower=0)

    return df


def limpiar_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia reviews — texto sucio, emojis, nulos."""
    df = df.copy()

    # Datos sucios: reviews sin comentario (~58% del dataset)
    df["review_comment_message"] = df["review_comment_message"].fillna("")
    df["review_comment_title"] = df["review_comment_title"].fillna("")

    # Normalizar score
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["review_score"] = df["review_score"].clip(1, 5).fillna(3)

    # Feature: review negativa
    df["review_negativa"] = (df["review_score"] <= 2).astype(int)

    return df


def limpiar_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia pagos — múltiples métodos por orden."""
    df = df.copy()

    # Datos sucios: installments = 0 en pagos que no son a plazos
    df["payment_installments"] = df["payment_installments"].clip(lower=1)

    # Datos sucios: valores negativos en payment_value (reversiones)
    reversiones = (df["payment_value"] < 0).sum()
    if reversiones > 0:
        print(f"  ⚠️  Pagos con valor negativo (reversiones): {reversiones}")
        df = df[df["payment_value"] >= 0]

    return df


def limpiar_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega región geográfica por estado — variable de riesgo crediticio."""
    df = df.copy()

    # Regiones de Brasil por estado
    norte = ["AM","RR","AP","PA","TO","RO","AC"]
    nordeste = ["MA","PI","CE","RN","PE","PB","SE","AL","BA"]
    centro_oeste = ["MT","MS","GO","DF"]
    sudeste = ["SP","RJ","MG","ES"]
    sul = ["PR","SC","RS"]

    def asignar_region(estado):
        if estado in norte: return "Norte"
        if estado in nordeste: return "Nordeste"
        if estado in centro_oeste: return "Centro-Oeste"
        if estado in sudeste: return "Sudeste"
        if estado in sul: return "Sul"
        return "Desconocido"

    df["region"] = df["geolocation_state"].apply(asignar_region)

    # Coordenadas promedio por zip code
    geo_agg = df.groupby("geolocation_zip_code_prefix").agg(
        lat_promedio=("geolocation_lat", "mean"),
        lng_promedio=("geolocation_lng", "mean"),
        region=("region", "first")
    ).reset_index()

    return geo_agg


# ── Construcción de perfiles de comportamiento ─────────────────────────────────
def construir_perfiles(dfs: dict) -> pd.DataFrame:
    """
    Une todos los datasets para construir un perfil
    de comportamiento crediticio por cliente.
    """
    print("\nConstruyendo perfiles de comportamiento...")

    orders = limpiar_orders(dfs["orders"])
    reviews = limpiar_reviews(dfs["reviews"])
    payments = limpiar_payments(dfs["payments"])
    geo = limpiar_geo(dfs["geo"])

    # Unir categorías de productos en inglés
    products = dfs["products"].merge(
        dfs["translation"],
        on="product_category_name",
        how="left"
    )
    products["product_category_name_english"] = (
        products["product_category_name_english"].fillna("unknown")
    )

    # Agregar items por orden
    items_agg = dfs["items"].merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
    items_agg = items_agg.groupby("order_id").agg(
        num_items=("order_item_id", "count"),
        ticket_promedio=("price", "mean"),
        ticket_total=("price", "sum"),
        flete_total=("freight_value", "sum"),
        categorias_distintas=("product_category_name_english", "nunique"),
    ).reset_index()

    # Agregar pagos por orden
    payments_agg = payments.groupby("order_id").agg(
        metodos_pago=("payment_type", "nunique"),
        cuotas_maximas=("payment_installments", "max"),
        valor_total_pagado=("payment_value", "sum"),
        uso_credito=("payment_type", lambda x: (x == "credit_card").any().astype(int)),
    ).reset_index()

    # Agregar reviews por orden
    reviews_agg = reviews.groupby("order_id").agg(
        score_promedio=("review_score", "mean"),
        review_negativa=("review_negativa", "max"),
    ).reset_index()

    # Unir todo por orden
    df = orders.merge(dfs["customers"], on="customer_id", how="left")
    df = df.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_agg, on="order_id", how="left")

    # Agregar geolocalización por zip code del cliente
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"}),
        on="customer_zip_code_prefix",
        how="left"
    )

    # Agregar perfil por cliente (histórico)
    perfil_cliente = df.groupby("customer_unique_id").agg(
        total_ordenes=("order_id", "count"),
        ticket_promedio_historico=("ticket_total", "mean"),
        ticket_maximo=("ticket_total", "max"),
        cuotas_maximas_historico=("cuotas_maximas", "max"),
        uso_credito_historico=("uso_credito", "max"),
        score_promedio_historico=("score_promedio", "mean"),
        reviews_negativas=("review_negativa", "sum"),
        dias_entrega_promedio=("dias_entrega", "mean"),
        retrasos_promedio=("dias_retraso", "mean"),
        categorias_exploradas=("categorias_distintas", "sum"),
        region=("region", "first"),
        primera_compra=("order_purchase_timestamp", "min"),
        ultima_compra=("order_purchase_timestamp", "max"),
    ).reset_index()

    # Calcular antigüedad en días
    fecha_corte = pd.Timestamp("2018-10-01")
    perfil_cliente["antiguedad_dias"] = (
        fecha_corte - perfil_cliente["primera_compra"]
    ).dt.days.clip(lower=0)

    perfil_cliente["dias_desde_ultima_compra"] = (
        fecha_corte - perfil_cliente["ultima_compra"]
    ).dt.days.clip(lower=0)

    # Variable objetivo: propensión a crédito
    # Clientes que usaron tarjeta de crédito Y tienen ticket alto Y múltiples órdenes
    perfil_cliente["propenso_credito"] = (
        (perfil_cliente["uso_credito_historico"] == 1) &
        (perfil_cliente["ticket_maximo"] > perfil_cliente["ticket_promedio_historico"] * 1.5) &
        (perfil_cliente["total_ordenes"] >= 2)
    ).astype(int)

    # Limpiar nulos residuales
    numericas = perfil_cliente.select_dtypes(include=[np.number]).columns
    perfil_cliente[numericas] = perfil_cliente[numericas].fillna(0)

    dist = perfil_cliente["propenso_credito"].value_counts()
    print(f"  ✅ Perfiles construidos: {len(perfil_cliente):,} clientes")
    print(f"  📊 Propensos a crédito: {dist.get(1,0):,} ({dist.get(1,0)/len(perfil_cliente)*100:.1f}%)")
    print(f"  📊 No propensos:        {dist.get(0,0):,} ({dist.get(0,0)/len(perfil_cliente)*100:.1f}%)")

    return perfil_cliente


# ── Guardado en MongoDB ────────────────────────────────────────────────────────
def guardar_en_mongodb(df: pd.DataFrame) -> None:
    print("\nGuardando en MongoDB...")
    client = MongoClient(MONGO_URI)  # nosec B101
    db = client[DB_NAME]
    col = db["perfiles_clientes"]

    col.drop()  # limpia colección anterior

    registros = df.to_dict("records")

    # Convertir timestamps a string para MongoDB
    for r in registros:
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                r[k] = str(v)

    col.insert_many(registros)
    print(f"  ✅ {len(registros):,} perfiles guardados en MongoDB")
    client.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Credit Behavior Engine — Ingesta y Limpieza")
    print("=" * 55)

    dfs = cargar_datasets()
    perfiles = construir_perfiles(dfs)
    print(perfiles.head(3).to_string())

    try:
        guardar_en_mongodb(perfiles)
    except Exception as e:
        print(f"  ⚠️  MongoDB no disponible: {e}")
        print("  💾 Guardando CSV de respaldo...")
        perfiles.to_csv("data/perfiles_clientes.csv", index=False)
        print("  ✅ CSV guardado en data/perfiles_clientes.csv")

    print("\n✅ Ingesta completa. Siguiente: python src/features.py")