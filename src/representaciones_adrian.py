"""
Módulo de preprocesamiento, representaciones vectoriales y patrones estilométricos (Adrián - v5).
Novedades v5:
  - Conteo exacto de tokens WordPiece reales del tokenizador de all-MiniLM-L6-v2.
  - Estadísticas léxicas exactas del corpus (media ± StDev de palabras y vocabulario |Sigma|).
  - Conteo explícito de artefactos Zenodo ('Tired') y protección en Kruskal-Wallis.
"""

import re
from typing import Iterable, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.feature_extraction.text import TfidfVectorizer


class PreprocesadorTextoClinico:
    """Corrige puntuación pegada ('relationships.It'), URLs y artefactos Zenodo ('Tired')."""

    RE_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    RE_PUNT_PEGADA = re.compile(r"([.!?;:,)])([A-Za-z])")
    RE_PAR_PEGADO = re.compile(r"([A-Za-z0-9])\(")
    # Artefacto del CSV original de Zenodo: saltos de línea/marcadores sustituidos por 'Tired' en mitad de frase
    RE_ARTEFACTO_TIRED = re.compile(r"(?<=[a-z,]\s)Tired(?=\s+[a-z])")
    RE_ESPACIOS = re.compile(r"\s+")

    def __init__(self, limpiar_artefactos_zenodo: bool = True):
        self.limpiar_artefactos_zenodo = limpiar_artefactos_zenodo

    def limpiar(self, texto: str) -> str:
        t = str(texto) if texto is not None else ""
        t = self.RE_URL.sub(" <url> ", t)
        t = self.RE_PUNT_PEGADA.sub(r"\1 \2", t)
        t = self.RE_PAR_PEGADO.sub(r"\1 (", t)
        if self.limpiar_artefactos_zenodo:
            t = self.RE_ARTEFACTO_TIRED.sub(" ", t)
        return self.RE_ESPACIOS.sub(" ", t).strip()

    def limpiar_lote(self, textos: Iterable[str]) -> List[str]:
        return [self.limpiar(t) for t in textos]

    def contar_artefactos_zenodo(self, textos: Iterable[str]) -> Dict[str, int]:
        """Cuantifica cuántos documentos y ocurrencias contienen el artefacto 'Tired'."""
        docs_afectados = 0
        ocurrencias = 0
        for t in textos:
            n = len(self.RE_ARTEFACTO_TIRED.findall(str(t) if t is not None else ""))
            if n > 0:
                docs_afectados += 1
                ocurrencias += n
        return {"docs_afectados": docs_afectados, "ocurrencias_totales": ocurrencias}


def calcular_estadisticas_corpus(textos: Iterable[str]) -> Dict[str, Any]:
    """
    Calcula estadísticas exactas del corpus requeridas por la plantilla del póster:
    N instancias, longitud documental en palabras (media ± StDev, mediana, mín, máx)
    y tamaño real del vocabulario único |Sigma|.
    """
    re_pal = re.compile(r"[a-zA-Z]+")
    vocab = set()
    longitudes = []
    for t in textos:
        pals = re_pal.findall(str(t).lower())
        longitudes.append(len(pals))
        vocab.update(pals)
    arr = np.asarray(longitudes, dtype=float)
    return {
        "n_documentos": int(len(arr)),
        "vocabulario_palabras_unico": int(len(vocab)),
        "palabras_media": round(float(np.mean(arr)), 2),
        "palabras_std": round(float(np.std(arr, ddof=1)), 2),
        "palabras_mediana": round(float(np.median(arr)), 2),
        "palabras_min": int(np.min(arr)),
        "palabras_max": int(np.max(arr)),
    }


class ExtractorPatronesEstilometricos:
    """
    Extrae 12 patrones estructurales y psicolingüísticos directamente del texto
    (ingeniería de atributos pura, sin ningún modelo supervisado auxiliar).
    """

    PRONOMBRES_1A = {"i", "im", "ive", "id", "me", "my", "mine", "myself"}
    PRONOMBRES_2A = {"you", "youre", "youve", "youd", "your", "yours", "yourself", "u"}
    PRONOMBRES_3A = {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs"}
    ABSOLUTISTAS = {
        "always", "never", "nothing", "everything", "everyone", "nobody",
        "completely", "totally", "forever", "constantly", "entire", "whole", "all", "none",
    }
    NEGACIONES = {
        "not", "no", "dont", "cant", "cannot", "wont", "didnt", "doesnt",
        "isnt", "arent", "wasnt", "werent", "never", "none", "nothing", "neither", "nobody",
    }

    NOMBRES_FEATURES = [
        "log_tokens",
        "log_frases",
        "tokens_por_frase",
        "long_media_palabra",
        "riqueza_lexica_guiraud",
        "pct_1a_persona",
        "pct_2a_persona",
        "balance_yo_vs_tu",
        "pct_3a_persona",
        "pct_absolutistas",
        "pct_negacion",
        "tasa_interrog_exclam",
    ]

    ETIQUETAS_LEGIBLES = {
        "log_tokens": "Volumen texto (log tokens)",
        "log_frases": "Nº de frases (log)",
        "tokens_por_frase": "Tokens por frase (media)",
        "long_media_palabra": "Longitud media de palabra",
        "riqueza_lexica_guiraud": "Riqueza léxica (Guiraud)",
        "pct_1a_persona": "% 1ª persona (I/me/my)",
        "pct_2a_persona": "% 2ª persona (you/your)",
        "balance_yo_vs_tu": "Balance Yo − Tú (1ª − 2ª)",
        "pct_3a_persona": "% 3ª persona (he/she/they)",
        "pct_absolutistas": "% Palabras absolutistas",
        "pct_negacion": "% Palabras de negación",
        "tasa_interrog_exclam": "Signos ? y ! por frase",
    }

    RE_PALABRAS = re.compile(r"[a-zA-Z']+")
    RE_FRASES = re.compile(r"[.!?]+")

    def __init__(self, limpiar_artefactos: bool = True):
        self.preprocesador = PreprocesadorTextoClinico(limpiar_artefactos_zenodo=limpiar_artefactos)

    def _extraer_uno(self, texto: str) -> List[float]:
        t = self.preprocesador.limpiar(texto)
        palabras_raw = self.RE_PALABRAS.findall(t.lower())
        palabras = [w.replace("'", "") for w in palabras_raw if w.replace("'", "")]
        n_pal = len(palabras)
        n_pal_safe = max(n_pal, 1)

        frases = [f.strip() for f in self.RE_FRASES.split(t) if len(f.strip()) >= 3]
        n_frases = max(len(frases), 1)

        tokens_por_frase = n_pal / float(n_frases)
        long_media_pal = float(np.mean([len(w) for w in palabras])) if n_pal > 0 else 0.0
        guiraud = len(set(palabras)) / float(np.sqrt(n_pal_safe))

        pct_1a = sum(1 for w in palabras if w in self.PRONOMBRES_1A) / float(n_pal_safe) * 100.0
        pct_2a = sum(1 for w in palabras if w in self.PRONOMBRES_2A) / float(n_pal_safe) * 100.0
        pct_3a = sum(1 for w in palabras if w in self.PRONOMBRES_3A) / float(n_pal_safe) * 100.0
        pct_abs = sum(1 for w in palabras if w in self.ABSOLUTISTAS) / float(n_pal_safe) * 100.0
        pct_neg = sum(1 for w in palabras if w in self.NEGACIONES) / float(n_pal_safe) * 100.0

        n_ie = t.count("?") + t.count("!")
        tasa_ie = n_ie / float(n_frases)

        return [
            float(np.log1p(n_pal)),
            float(np.log1p(n_frases)),
            float(min(tokens_por_frase, 120.0)),
            long_media_pal,
            float(guiraud),
            pct_1a,
            pct_2a,
            pct_1a - pct_2a,
            pct_3a,
            pct_abs,
            pct_neg,
            float(min(tasa_ie, 5.0)),
        ]

    def transform(self, textos: Iterable[str]) -> np.ndarray:
        """Devuelve matriz numpy (N, 12) con los rasgos estilométricos."""
        return np.asarray([self._extraer_uno(t) for t in textos], dtype=np.float32)

    def analizar_significancia_clases(
        self, textos: Iterable[str], y: np.ndarray, clases_ordenadas: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evalúa Kruskal-Wallis estrictamente sobre Train (sin tocar Dev/Test)."""
        X_est = self.transform(textos)
        df_raw = pd.DataFrame(X_est, columns=self.NOMBRES_FEATURES)
        y_arr = np.asarray(y)

        filas_kw = []
        for col in self.NOMBRES_FEATURES:
            if df_raw[col].nunique() <= 1:
                h_stat, p_val = 0.0, 1.0
            else:
                grupos = [df_raw.loc[y_arr == cls, col].values for cls in clases_ordenadas]
                h_stat, p_val = kruskal(*grupos)
            filas_kw.append({
                "feature": col,
                "etiqueta": self.ETIQUETAS_LEGIBLES[col],
                "H_kruskal": round(float(h_stat), 3),
                "p_value": float(p_val),
                "neg_log10_p": round(float(-np.log10(max(p_val, 1e-15))), 3),
                "significativo_05": bool(p_val < 0.05),
            })

        df_stats = pd.DataFrame(filas_kw).sort_values("H_kruskal", ascending=False).reset_index(drop=True)

        df_z = (df_raw - df_raw.mean()) / df_raw.std(ddof=0).replace(0, 1.0)
        df_z["clase"] = y_arr
        df_zscores = df_z.groupby("clase").mean().reindex(clases_ordenadas)

        return df_stats, df_zscores


class RepresentadorTFIDFCaracteres:
    """Transforma textos en vectores dispersos TF-IDF a nivel de caracteres."""

    def __init__(
        self,
        analyzer: str = "char_wb",
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 4,
        max_features: int = 30000,
        sublinear_tf: bool = True,
        limpiar_artefactos: bool = True,
    ):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.limpiar_artefactos = limpiar_artefactos

        self.preprocesador = PreprocesadorTextoClinico(limpiar_artefactos_zenodo=limpiar_artefactos)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            strip_accents="unicode",
        )
        self._ajustado = False

    def fit_transform(self, textos_train: Iterable[str]):
        limpios = self.preprocesador.limpiar_lote(textos_train)
        X_train = self.vectorizer.fit_transform(limpios)
        self._ajustado = True
        return X_train

    def transform(self, textos: Iterable[str]):
        if not self._ajustado:
            raise RuntimeError("Debes llamar a fit_transform(textos_train) antes de transform().")
        limpios = self.preprocesador.limpiar_lote(textos)
        return self.vectorizer.transform(limpios)

    def obtener_info(self) -> dict:
        vocab_size = len(self.vectorizer.vocabulary_) if self._ajustado else 0
        return {
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "limpiar_artefactos": self.limpiar_artefactos,
            "vocabulario_real": vocab_size,
        }


class RepresentadorSBERT384D:
    """Wrapper para codificar textos con all-MiniLM-L6-v2 y medir tokens WordPiece reales."""

    def __init__(self, nombre_modelo: str = "all-MiniLM-L6-v2"):
        self.nombre_modelo = nombre_modelo
        self._modelo = None

    def _cargar_modelo(self):
        if self._modelo is None:
            import os
            import warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
            from sentence_transformers import SentenceTransformer
            self._modelo = SentenceTransformer(self.nombre_modelo)

    def encode(self, textos: Iterable[str], show_progress_bar: bool = False) -> np.ndarray:
        self._cargar_modelo()
        lista_textos = [str(t) if t is not None else "" for t in textos]
        vectores = self._modelo.encode(lista_textos, show_progress_bar=show_progress_bar)
        return np.asarray(vectores, dtype=np.float32)

    def medir_tokens_reales(self, textos: Iterable[str], limite_tokens: int = 256) -> Dict[str, Any]:
        """
        Cuenta los tokens WordPiece reales del tokenizador interno de all-MiniLM-L6-v2
        (sin truncar) para obtener el porcentaje exacto de usuarios truncados por SBERT.
        """
        self._cargar_modelo()
        tok = self._modelo.tokenizer
        lista = [str(t) if t is not None else "" for t in textos]
        longitudes = [
            len(tok.encode(t, add_special_tokens=True, truncation=False))
            for t in lista
        ]
        arr = np.asarray(longitudes, dtype=int)
        return {
            "metodo": "wordpiece_real_all-MiniLM-L6-v2",
            "limite_tokens": int(limite_tokens),
            "longitudes_tokens": longitudes,
            "pct_truncados": round(float(np.mean(arr > limite_tokens) * 100.0), 2),
            "n_truncados": int(np.sum(arr > limite_tokens)),
            "media_tokens": round(float(np.mean(arr)), 2),
            "std_tokens": round(float(np.std(arr, ddof=1)), 2),
            "mediana_tokens": int(np.median(arr)),
            "min_tokens": int(np.min(arr)),
            "max_tokens": int(np.max(arr)),
        }