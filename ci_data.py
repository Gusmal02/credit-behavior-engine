"""Genera dataset sintético para CI."""
import os, pandas as pd, numpy as np
np.random.seed(42)
n = 1000
os.makedirs('data', exist_ok=True)
df = pd.DataFrame({
    'customer_unique_id': [f'c{i}' for i in range(n)],
    'total_ordenes': np.random.randint(1,5,n),
    'ticket_promedio_historico': np.random.exponential(200,n),
    'ticket_maximo': np.random.exponential(300,n),
    'cuotas_maximas_historico': np.random.randint(1,12,n).astype(float),
    'uso_credito_historico': np.random.randint(0,2,n).astype(float),
    'score_promedio_historico': np.random.uniform(1,5,n),
    'reviews_negativas': np.random.randint(0,3,n).astype(float),
    'dias_entrega_promedio': np.random.uniform(3,20,n),
    'retrasos_promedio': np.random.uniform(0,5,n),
    'categorias_exploradas': np.random.randint(1,5,n).astype(float),
    'antiguedad_dias': np.random.randint(30,500,n).astype(float),
    'dias_desde_ultima_compra': np.random.randint(1,200,n).astype(float),
    'region': np.random.choice(['Sudeste','Sul','Nordeste','Norte','Centro-Oeste'],n),
    'propenso_credito': np.random.randint(0,2,n),
})
df.to_csv('data/perfiles_clientes.csv', index=False)
print('Dataset CI generado')