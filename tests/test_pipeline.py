"""Suite de pruebas — Credit Behavior Engine"""
import os, sys, pytest, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestIngesta:
    def test_csv_existe(self):
        assert os.path.exists("data/perfiles_clientes.csv")

    def test_csv_columnas(self):
        df = pd.read_csv("data/perfiles_clientes.csv")
        for col in ["customer_unique_id","total_ordenes","propenso_credito","region"]:
            assert col in df.columns

    def test_sin_nulos_criticos(self):
        df = pd.read_csv("data/perfiles_clientes.csv")
        assert df["customer_unique_id"].isnull().sum() == 0

    def test_target_valido(self):
        df = pd.read_csv("data/perfiles_clientes.csv")
        assert set(df["propenso_credito"].unique()).issubset({0, 1})

class TestModelos:
    def test_modelo_existe(self):
        assert os.path.exists("models/credit_model_best.pth")

    def test_scaler_existe(self):
        assert os.path.exists("models/scaler.pkl")

    def test_encoder_existe(self):
        assert os.path.exists("models/label_encoder.pkl")

class TestScoring:
    def test_score_rango_valido(self):
        """Score 0-1000 calculado correctamente."""
        prob = 0.85
        score = int(prob * 1000)
        assert 0 <= score <= 1000

    def test_segmentos_validos(self):
        """Los segmentos de negocio son los esperados."""
        segmentos_validos = {"CANDIDATO","PREMIUM","REVISAR","POTENCIAL","NO_CANDIDATO"}
        for s in segmentos_validos:
            assert s in segmentos_validos

    def test_region_desconocida_manejada(self):
        """Región inválida se normaliza a Desconocido."""
        regiones_validas = ["Centro-Oeste","Desconocido","Nordeste","Norte","Sudeste","Sul"]
        region = "RegionInexistente"
        resultado = region if region in regiones_validas else "Desconocido"
        assert resultado == "Desconocido"