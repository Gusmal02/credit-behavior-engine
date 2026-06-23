# Documentación Técnica — Credit Behavior Engine

## Decisiones de arquitectura

### ¿Por qué Embeddings y no One-Hot Encoding?
One-Hot encoding trata las 6 regiones como categorías independientes sin relación entre ellas. El Embedding aprende representaciones densas de 4 dimensiones donde regiones geográficamente similares (Sudeste/Sul) terminan cercanas en el espacio vectorial. Con solo 96K registros, 4 dimensiones son suficientes para capturar la señal geográfica sin overfitting.

### ¿Por qué pos_weight=146 y no SMOTE?
SMOTE genera ejemplos sintéticos interpolando entre positivos existentes — con solo 653 positivos reales el riesgo de overfitting es alto. pos_weight penaliza directamente la función de pérdida BCEWithLogitsLoss, obligando al modelo a aprender de cada positivo real sin distorsionar la distribución del dataset.

### ¿Por qué WeightedRandomSampler además de pos_weight?
Doble defensa contra el desbalance: pos_weight corrige la función de pérdida, WeightedRandomSampler garantiza que cada batch contenga suficientes ejemplos positivos para que el gradiente sea informativo. Sin esto, batches de 512 muestras tendrían en promedio solo 3-4 positivos — señal insuficiente.

### ¿Por qué threshold=0.3 y no 0.5?
Con desbalance 146x, el modelo tiende a ser conservador en sus predicciones de probabilidad. Bajar el threshold de 0.5 a 0.3 captura más verdaderos positivos a costa de más falsos positivos — trade-off aceptable en originación de crédito donde el costo de no contactar a un candidato es mayor que el de contactar a alguien no interesado.

### ¿Por qué MongoDB y no PostgreSQL?
Los perfiles conductuales son documentos con estructura variable — un cliente con 10 órdenes tiene más campos que uno con 1. MongoDB permite almacenar perfiles heterogéneos sin definir esquema rígido. Además, la integración con n8n (roadmap) es más natural con MongoDB para flujos de eventos.

## Datos sucios encontrados y tratamiento

| Dato sucio | Cantidad | Tratamiento |
|---|---|---|
| Fechas de aprobación faltantes | 160 | Imputación con purchase_timestamp |
| Entregas sin fecha real | ~2,965 | NaT — órdenes canceladas/pendientes |
| Reviews sin comentario | ~58% | String vacío |
| Días de entrega negativos | Verificado: 0 | N/A |
| Geolocalización duplicada | 1,000,163 raw | Agregación por zip_code_prefix |

## Notas de seguridad

### B301 — pickle
Los modelos se cargan desde almacenamiento interno controlado (disco local o volumen Docker), nunca desde input externo no confiable. El riesgo de deserialización maliciosa es aceptado y documentado.

### B101 — MongoClient
Conexión a instancia MongoDB interna del mismo entorno Docker. No expuesta a internet en configuración de producción.

## MLflow

Experimento: `credit-behavior-engine`
Run: `CreditBehaviorNet_v1`
Parámetros tracked: arquitectura, epochs, batch_size, learning_rate, pos_weight, embedding_dim, dropout, num_features, num_regiones, train_size, test_size
Métricas tracked por época: loss, auprc, auroc