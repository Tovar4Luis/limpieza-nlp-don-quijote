# API de análisis de sentimientos

Carga `modelo_nlp.pkl`, generado con el vectorizador ganador entre Bag-of-Words y TF-IDF, y clasifica reseñas como positivas o negativas.

## Iniciar el servicio

Desde esta carpeta:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: <http://127.0.0.1:8000/docs>

## Consultar una predicción

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"review":"The food was amazing and the service was excellent!"}'
```

`liked` vale `1` para una reseña positiva y `0` para una negativa.
