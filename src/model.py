"""
Credit Behavior Engine — Modelo Deep Learning
Arquitectura: Embedding (región) + Dense Network
Manejo de desbalance extremo con pos_weight=146
MLflow para tracking de experimentos
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
from pymongo import MongoClient
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "credit_behavior"
MODELS_DIR = "models"
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Dataset ────────────────────────────────────────────────────────────────────
class CreditDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


# ── Arquitectura ───────────────────────────────────────────────────────────────
class CreditBehaviorNet(nn.Module):
    """
    Red neuronal con Embedding para región geográfica.
    Embedding aprende representaciones densas de cada región
    en lugar de one-hot encoding — captura similitudes geográficas.
    """
    def __init__(self, num_features: int, num_regiones: int, embedding_dim: int = 4):
        super().__init__()

        # Embedding para región — transforma índice entero en vector denso
        self.embedding = nn.Embedding(num_regiones, embedding_dim)

        input_dim = num_features + embedding_dim

        self.red = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_cat):
        # Embedding de región
        emb = self.embedding(x_cat.squeeze(1))
        # Concatenar numéricas + embedding
        x = torch.cat([x_num, emb], dim=1)
        return self.red(x).squeeze(1)


# ── Carga de datos ─────────────────────────────────────────────────────────────
def cargar_datos():
    print("Cargando datos desde MongoDB...")
    client = MongoClient(MONGO_URI)  # nosec B101
    db = client[DB_NAME]
    registros = list(db["perfiles_clientes"].find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(registros)

    with open(f"{MODELS_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # nosec B301
    with open(f"{MODELS_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)  # nosec B301
    with open(f"{MODELS_DIR}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)  # nosec B301

    # Reconstruir features igual que en features.py
    df["ratio_flete_ticket"] = (
        df.get("flete_total", pd.Series(0, index=df.index)) /
        df["ticket_promedio_historico"].clip(lower=1)
    ).fillna(0)
    df["frecuencia_compra"] = (
        df["total_ordenes"] /
        (df["antiguedad_dias"] / 30).clip(lower=1)
    ).fillna(0)
    df["score_comportamiento"] = (
        df["score_promedio_historico"] * 0.4 +
        (1 / df["retrasos_promedio"].clip(lower=1)) * 0.3 +
        df["frecuencia_compra"] * 0.3
    ).fillna(0)
    df["es_recurrente"] = (df["total_ordenes"] > 1).astype(int)
    df["alto_valor"] = (
        df["ticket_maximo"] > df["ticket_promedio_historico"].median()
    ).astype(int)

    X_num = scaler.transform(df[feature_names].fillna(0).values.astype(np.float32))
    df["region_encoded"] = le.transform(df["region"].fillna("Desconocido"))
    X_cat = df[["region_encoded"]].values.astype(np.int64)
    y = df["propenso_credito"].values.astype(np.float32)

    num_regiones = len(le.classes_)
    ratio = int((1 - y).sum() / max(y.sum(), 1))

    print(f"  ✅ {len(df):,} registros | ratio desbalance: {ratio}x")
    return X_num, X_cat, y, num_regiones, ratio


# ── Entrenamiento ──────────────────────────────────────────────────────────────
def entrenar():
    print("\n" + "=" * 55)
    print("Credit Behavior Engine — Entrenamiento Deep Learning")
    print("=" * 55)

    X_num, X_cat, y, num_regiones, ratio = cargar_datos()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Split estratificado — preserva proporción de positivos
    X_num_tr, X_num_te, X_cat_tr, X_cat_te, y_tr, y_te = train_test_split(
        X_num, X_cat, y, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y
    )

    # WeightedRandomSampler — sobremuestrea positivos en cada batch
    pesos_clase = np.where(y_tr == 1, ratio, 1.0)
    sampler = WeightedRandomSampler(
        weights=pesos_clase,
        num_samples=len(y_tr),
        replacement=True
    )

    train_ds = CreditDataset(X_num_tr, X_cat_tr, y_tr)
    test_ds = CreditDataset(X_num_te, X_cat_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=512, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = CreditBehaviorNet(
        num_features=X_num.shape[1],
        num_regiones=num_regiones,
        embedding_dim=4
    ).to(device)

    # pos_weight corrige la función de pérdida para desbalance extremo
    pos_weight = torch.tensor([float(ratio)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # ── MLflow tracking ────────────────────────────────────────────────────
    mlflow.set_experiment("credit-behavior-engine")

    with mlflow.start_run(run_name="CreditBehaviorNet_v1"):
        mlflow.log_params({
            "arquitectura": "Embedding + Dense 128-64-32",
            "epochs": 20,
            "batch_size": 512,
            "learning_rate": 0.001,
            "pos_weight": ratio,
            "embedding_dim": 4,
            "dropout": "0.4/0.3/0.2",
            "num_features": X_num.shape[1],
            "num_regiones": num_regiones,
            "train_size": len(y_tr),
            "test_size": len(y_te),
        })

        num_features = X_num.shape[1]
        EPOCHS = 20
        mejor_auprc = 0.0

        for epoch in range(EPOCHS):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            loss_total = 0.0
            for x_num, x_cat, labels in train_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(x_num, x_cat)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_total += loss.item()

            # ── Eval ───────────────────────────────────────────────────────
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for x_num, x_cat, labels in test_loader:
                    logits = model(x_num.to(device), x_cat.to(device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            auprc = average_precision_score(all_labels, all_probs)
            auroc = roc_auc_score(all_labels, all_probs)
            loss_avg = loss_total / len(train_loader)

            scheduler.step(loss_avg)

            mlflow.log_metrics({
                "loss": round(loss_avg, 4),
                "auprc": round(auprc, 4),
                "auroc": round(auroc, 4),
            }, step=epoch)

            print(f"Época {epoch+1:02d}/{EPOCHS} | "
                  f"Loss: {loss_avg:.4f} | "
                  f"AUPRC: {auprc:.4f} | "
                  f"AUROC: {auroc:.4f}")

            # Guardar mejor modelo
            if auprc > mejor_auprc:
                mejor_auprc = auprc
                torch.save(
                    model.state_dict(),
                    f"{MODELS_DIR}/credit_model_best.pth"
                )

        # ── Evaluación final ───────────────────────────────────────────────
        print("\n── Evaluación Final ──")
        threshold = 0.3  # umbral bajo para capturar más positivos
        preds = (all_probs >= threshold).astype(int)

        print(classification_report(
            all_labels, preds,
            target_names=["No Propenso", "Propenso Crédito"],
            zero_division=0
        ))

        cm = confusion_matrix(all_labels, preds)
        print(f"Matriz de confusión:\n{cm}")

        mlflow.log_metrics({
            "mejor_auprc": round(mejor_auprc, 4),
            "threshold": threshold,
        })
        # Guardar modelo con ejemplo de input para trazabilidad
        example_num = torch.zeros(1, num_features)
        example_cat = torch.zeros(1, 1, dtype=torch.long)
        mlflow.pytorch.log_model(
            model,
            name="credit_behavior_model",
            input_example=(example_num, example_cat),
            serialization_format="pickle",
        )

        print(f"\n✅ Mejor AUPRC: {mejor_auprc:.4f}")
        print(f"✅ Modelo guardado en {MODELS_DIR}/credit_model_best.pth")
        print("✅ Experimento registrado en MLflow")


if __name__ == "__main__":
    entrenar()
    print("\n✅ Entrenamiento completo. Siguiente: python src/scoring.py")
    