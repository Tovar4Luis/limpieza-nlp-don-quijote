# Normalización y lematización de texto en español

Este proyecto aplica al texto de *Don Quijote de la Mancha*, de Miguel de Cervantes, el flujo de normalización y lematización visto en clase.

## Archivos

- `don_quijote.txt`: texto plano del libro, procedente del libro electrónico 2000 de Project Gutenberg: https://www.gutenberg.org/ebooks/2000. La obra es de dominio público.
- `limpieza.py`: código Python del procesamiento.
- `requirements.txt`: versiones congeladas de las dependencias instaladas en el entorno virtual.

## Proceso aplicado

1. Carga el modelo español `es_core_news_sm` de spaCy (lo descarga al ejecutarse por primera vez, si hace falta).
2. Tokeniza el texto y muestra los primeros tokens.
3. Cuenta y elimina stop words, puntuación y espacios.
4. Convierte cada lema a minúsculas y guarda el texto normalizado en `don_quijote_normalizado.txt`.
5. Compara el stemming de Snowball en español con la lematización de spaCy; guarda la tabla CSV en `comparacion_stemming_lematizacion.csv`.

El libro se procesa en bloques para evitar el límite de longitud de spaCy.

## Crear el entorno y ejecutar (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python limpieza.py
```

También se pueden indicar rutas de entrada y salida:

```powershell
python limpieza.py --input don_quijote.txt --output don_quijote_normalizado.txt --comparison comparacion_stemming_lematizacion.csv
```
