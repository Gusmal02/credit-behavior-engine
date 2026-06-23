# Hallazgos — Credit Behavior Engine

## Hallazgo 1: Desbalance extremo de clases (146x)

Solo 653 de 96,096 clientes califican como propensos a crédito (0.7%). Este desbalance es real y esperado — Olist es un marketplace general donde la mayoría compra ocasionalmente sin señales de propensión financiera.

**Impacto:** Un modelo sin corrección aprende a decir "no propenso" siempre y obtiene 99.3% de accuracy sin aprender nada útil.

**Solución implementada:** Triple defensa — pos_weight=146, WeightedRandomSampler y threshold=0.3. Resultado: Recall 100% en clase positiva.

## Hallazgo 2: AUROC 0.9958 con desbalance 146x

El modelo distingue propensos de no propensos con 99.58% de certeza en ranking. Esto significa que si ordenamos todos los clientes por score descendente, los propensos reales aparecen consistentemente en los primeros lugares — exactamente lo que necesita una campaña de originación masiva.

## Hallazgo 3: AUPRC 0.6024 — 86x mejor que azar

La línea base aleatoria daría AUPRC=0.007 (proporción de positivos). Logramos 0.6024 — 86 veces mejor. En contextos de desbalance extremo, AUPRC es la métrica honesta porque mide precisión en todos los umbrales posibles.

## Hallazgo 4: Segmentación geográfica real

El Embedding aprendió que clientes del Sudeste (São Paulo, Rio de Janeiro) tienen perfiles diferentes a clientes del Nordeste o Norte. Esto se refleja en los scores — los dos candidatos detectados en el batch de 200 son Nordeste (score 881) y Sudeste (score 999), con tickets acordes al poder adquisitivo regional.

## Hallazgo 5: Recall 100% con 374 falsos positivos

La matriz de confusión muestra:
- 131 propensos detectados de 131 — ninguno escapó
- 374 falsos positivos de 19,089 negativos (2%)

En originación de crédito, el costo de no contactar a un cliente que sí quiere crédito (falso negativo) es mayor que el costo de contactar a alguien que no está interesado (falso positivo). La configuración actual prioriza correctamente el recall.

## Hallazgo 6: Datos sucios reales de Olist

El dataset contiene imperfecciones reales: 160 fechas de aprobación faltantes, ~2,965 entregas sin fecha, 58% de reviews sin comentario. El pipeline de limpieza los maneja explícitamente con estrategias documentadas — no los elimina.