"""Script de entrenamiento para CI — genera modelos mínimos para tests."""
import os, pickle, torch, torch.nn as nn
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('data/perfiles_clientes.csv')
features = ['total_ordenes','ticket_promedio_historico','ticket_maximo',
            'cuotas_maximas_historico','uso_credito_historico',
            'score_promedio_historico','reviews_negativas',
            'dias_entrega_promedio','retrasos_promedio',
            'categorias_exploradas','antiguedad_dias',
            'dias_desde_ultima_compra','ratio_flete_ticket',
            'frecuencia_compra','score_comportamiento',
            'es_recurrente','alto_valor']

df['ratio_flete_ticket'] = 0.0
df['frecuencia_compra'] = df['total_ordenes'] / (df['antiguedad_dias']/30).clip(lower=1)
df['score_comportamiento'] = df['score_promedio_historico'] * 0.4
df['es_recurrente'] = (df['total_ordenes'] > 1).astype(int)
df['alto_valor'] = (df['ticket_maximo'] > df['ticket_promedio_historico']).astype(int)

scaler = StandardScaler()
scaler.fit(df[features].fillna(0))
le = LabelEncoder()
le.fit(['Centro-Oeste','Desconocido','Nordeste','Norte','Sudeste','Sul'])

os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl','wb') as f: pickle.dump(scaler, f)
with open('models/label_encoder.pkl','wb') as f: pickle.dump(le, f)
with open('models/feature_names.pkl','wb') as f: pickle.dump(features, f)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 4)
        self.red = nn.Sequential(
            nn.Linear(21, 128),
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
    def forward(self, x, c):
        e = self.embedding(c.squeeze(1))
        return self.red(torch.cat([x, e], dim=1)).squeeze(1)

model = Net()
torch.save(model.state_dict(), "models/credit_model_best.pth")
print("Modelos CI listos")