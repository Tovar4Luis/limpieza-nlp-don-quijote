# Procesamiento de texto y semántica distribucional

Proyecto de procesamiento de *Don Quijote de la Mancha*, de Miguel de Cervantes. El archivo fuente se obtuvo del libro electrónico 2000 de Project Gutenberg (obra de dominio público): https://www.gutenberg.org/ebooks/2000.

## Entregables

- `don_quijote.txt`: libro original.
- `don_quijote_normalizado.txt`: texto tokenizado, en minúsculas, lematizado y filtrado de stop words y signos.
- `proyecto_nlp.py`: pipeline completo de carga, limpieza, lematización, entrenamiento Word2Vec y gráficas.
- `espacio_word2vec_skipgram.png` y `espacio_word2vec_cbow.png`: visualizaciones 2D de palabras y sus contextos en ventanas locales y amplias.
- `requirements.txt`: dependencias congeladas del entorno virtual.

## Proceso

1. Se retira la cabecera y el cierre editorial de Project Gutenberg.
2. spaCy tokeniza en español y lematiza; se pasan los lemas a minúsculas y se filtran stop words, puntuación y espacios.
3. Se guarda el texto procesado en `don_quijote_normalizado.txt`.
4. Gensim entrena dos modelos Word2Vec sobre las oraciones: Skip-gram (`sg=1`) y CBOW (`sg=0`), con ventana de contexto 5.
5. Las 60 palabras más frecuentes se proyectan a dos dimensiones con PCA para crear una gráfica por modelo.

Las imágenes incluidas muestran proyecciones distribucionales obtenidas directamente del texto limpio. Al ejecutar el pipeline, se generan de nuevo como proyecciones PCA etiquetadas a partir de los modelos Word2Vec entrenados.

## Ejecución en PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python proyecto_nlp.py
```

En la primera ejecución, si falta el modelo `es_core_news_sm`, spaCy intenta descargarlo. El script permite indicar otros archivos de entrada y salida:

```powershell
python proyecto_nlp.py --input don_quijote.txt --output don_quijote_normalizado.txt
```
