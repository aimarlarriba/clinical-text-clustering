"""
Configuración centralizada - Práctica 2 (Clasificación Supervisada) - Adrián (v5).
Rejillas ampliadas para evitar óptimos en frontera y decisión basada en coste (Tema 3).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "500_Reddit_users_posts_labels.csv"
CACHE_EMB_COMUN = DATA_DIR / "embeddings_384d_comun.npy"

OUTPUT_DIR = PROJECT_ROOT / "resultados_adrian"
FIGURES_DIR = OUTPUT_DIR / "figuras_poster"

RANDOM_SEED_OFICIAL = 42
SEEDS_ESTABILIDAD = [42, 7, 13, 21, 99, 123, 256, 512, 777, 1024]

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_MAX_TOKENS = 256

# Rejillas ampliadas para evitar óptimos truncados en los bordes (5-Fold CV sobre Train)
GRID_C_384D = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
GRID_C_TFIDF = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
GRID_ALPHA_PESOS = [0.5, 0.75, 1.0, 1.25, 1.5]
GRID_PESO_ESTILOMETRIA = [0.05, 0.15, 0.30, 0.50, 0.80]

# Multiplicador de coste lambda >= 1.0 para la clase minoritaria (Tema 3: Decisión basada en coste)
# Regla: c_hat(x) = argmax_c [ lambda_c * P(y=c | x) ], con lambda_min = lambda y lambda_resto = 1.0
GRID_LAMBDA_COSTE = [
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40,
    1.50, 1.60, 1.75, 1.90, 2.00, 2.25, 2.50, 3.00,
]

# Rejilla TF-IDF por caracteres (min_df=4 y 5 frenan memorización de nombres propios aislados)
GRID_TFIDF_CHAR = [
    {"analyzer": "char_wb", "ngram_range": (2, 4), "min_df": 3, "max_features": 25000, "limpiar_artefactos": True},
    {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 4, "max_features": 30000, "limpiar_artefactos": True},
    {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5, "max_features": 35000, "limpiar_artefactos": True},
]

NIVELES_RUIDO_LEXICO = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
REPETICIONES_RUIDO = 5


def crear_directorios_salida() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)