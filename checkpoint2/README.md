# Checkpoint 2: Clasificación de reseñas y vectorización

Este subproyecto continúa el pipeline de reseñas de restaurantes con las técnicas de vectorización vistas en clase: Bag-of-Words y TF-IDF.

## Archivos

- `Restaurant_Reviews.tsv`: corpus etiquetado (`Liked`: 1 positiva, 0 negativa).
- `train.py`: separa datos de entrenamiento y prueba, compara CountVectorizer y TfidfVectorizer con el mismo clasificador y selecciona el de mejor F1 macro.
- `resultados_vectorizacion.json`: accuracy, F1 macro y reporte de clasificación de cada representación.
- `modelo_nlp.pkl`: pipeline elegido, listo para cargar en `app.py`.
- `app.py`: API de predicción usada en la actividad anterior.
- `requirements.txt`: dependencias congeladas del entorno del proyecto.

## ¿Qué hace cada vectorizador?

- **Bag-of-Words (`CountVectorizer`)** convierte cada reseña en conteos de términos. Es sencillo e interpretable, pero términos frecuentes pueden dominar.
- **TF-IDF (`TfidfVectorizer`)** pondera términos por su frecuencia en una reseña y reduce el peso de los que aparecen en muchas reseñas. Conserva la representación dispersa y suele destacar términos más distintivos.

Ambos pipelines usan el mismo clasificador y la misma partición estratificada para comparar resultados de manera justa. El modelo con mayor F1 macro se guarda como `modelo_nlp.pkl`.

## Ejecutar

Desde esta carpeta, con el entorno virtual del repositorio activado:

```powershell
python train.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

La API recibe una reseña en `POST /predict`; la documentación interactiva está en `http://127.0.0.1:8000/docs`.

Las métricas sirven para esta partición de prueba y este conjunto de datos; no garantizan el mismo rendimiento con reseñas de otros restaurantes o dominios.
