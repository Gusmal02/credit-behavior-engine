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
    def test_score_cliente_valido(self):
        from src.scoring import score_cliente
        perfil = {
            "customer_unique_id": "test_001",
            "total_ordenes": 3,
            "ticket_promedio_historico": 200.0,
            "ticket_maximo": 500.0,
            "cuotas_maximas_historico": 6.0,
            "uso_credito_historico": 1.0,
            "score_promedio_historico": 4.0,
            "reviews_negativas": 0.0,
            "dias_entrega_promedio": 7.0,
            "retrasos_promedio": 0.0,
            "categorias_exploradas": 3.0,
            "antiguedad_dias": 200.0,
            "dias_desde_ultima_compra": 45.0,
            "region": "Sudeste",
        }
        r = score_cliente(perfil)
        assert 0 <= r["probabilidad"] <= 1
        assert 0 <= r["score_credito"] <= 1000
        assert r["segmento"] in ["CANDIDATO","PREMIUM","REVISAR","POTENCIAL","NO_CANDIDATO"]

    def test_score_rango_valido(self):
        from src.scoring import score_cliente
        perfil = {"customer_unique_id":"test_002","total_ordenes":1,
                  "ticket_promedio_historico":50.0,"ticket_maximo":50.0,
                  "cuotas_maximas_historico":1.0,"uso_credito_historico":0.0,
                  "score_promedio_historico":3.0,"reviews_negativas":0.0,
                  "dias_entrega_promedio":10.0,"retrasos_promedio":0.0,
                  "categorias_exploradas":1.0,"antiguedad_dias":30.0,
                  "dias_desde_ultima_compra":30.0,"region":"Norte"}
        r = score_cliente(perfil)
        assert isinstance(r["score_credito"], int)

    def test_region_desconocida(self):
        from src.scoring import score_cliente
        perfil = {"customer_unique_id":"test_003","total_ordenes":1,
                  "ticket_promedio_historico":100.0,"ticket_maximo":100.0,
                  "cuotas_maximas_historico":1.0,"uso_credito_historico":0.0,
                  "score_promedio_historico":3.0,"reviews_negativas":0.0,
                  "dias_entrega_promedio":5.0,"retrasos_promedio":0.0,
                  "categorias_exploradas":1.0,"antiguedad_dias":60.0,
                  "dias_desde_ultima_compra":60.0,"region":"RegionInexistente"}
        r = score_cliente(perfil)
        assert r["segmento"] is not None