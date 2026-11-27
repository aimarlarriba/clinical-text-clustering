"""
Módulo del Experimento RQ1: Robustez ante Ruido Léxico - Adrián (v5).
Mejoras v5:
  - Inyecta erratas estrictamente sobre palabras alfabéticas (preserva puntuación externa).
  - Garantiza mutación real (nunca sustituye por la misma letra ni transpone letras iguales).
  - Reporta la degradación tanto en F1-Macro como en F1-Minoritaria (Attempt).
"""

import re
import string
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.representaciones_adrian import RepresentadorSBERT384D
from src.evaluador_adrian import EvaluadorClasificacion
from src.modelos_adrian import ClasificadorBase


class InyectorRuidoLexico:
    """Inyecta erratas léxicas garantizadas (sustitución, borrado, inserción, transposición)."""

    ALFABETO = string.ascii_lowercase
    RE_PALABRA = re.compile(r"\b[A-Za-z]{3,}\b")

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _sustituir_distinta(self, c: str) -> str:
        c_low = c.lower()
        candidatas = [ch for ch in self.ALFABETO if ch != c_low]
        nueva = candidatas[int(self.rng.randint(0, len(candidatas)))]
        return nueva.upper() if c.isupper() else nueva

    def _corromper_palabra(self, palabra: str) -> str:
        if len(palabra) < 3:
            return palabra
        operacion = self.rng.choice(["sustituir", "borrar", "insertar", "transponer"])
        chars = list(palabra)

        if operacion == "transponer":
            pares_distintos = [i for i in range(len(chars) - 1) if chars[i].lower() != chars[i + 1].lower()]
            if pares_distintos:
                pos = pares_distintos[int(self.rng.randint(0, len(pares_distintos)))]
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                return "".join(chars)
            operacion = "sustituir"

        if operacion == "sustituir":
            pos = int(self.rng.randint(0, len(chars)))
            chars[pos] = self._sustituir_distinta(chars[pos])
        elif operacion == "borrar":
            chars.pop(int(self.rng.randint(0, len(chars))))
        elif operacion == "insertar":
            pos = int(self.rng.randint(0, len(chars) + 1))
            chars.insert(pos, self.ALFABETO[int(self.rng.randint(0, len(self.ALFABETO)))])
        return "".join(chars)

    def corromper_texto(self, texto: str, tasa_ruido: float) -> str:
        if tasa_ruido <= 0.0:
            return str(texto)

        def _reemplazar(match):
            w = match.group(0)
            return self._corromper_palabra(w) if self.rng.rand() < tasa_ruido else w

        return self.RE_PALABRA.sub(_reemplazar, str(texto))

    def corromper_corpus(self, textos: List[str], tasa_ruido: float) -> List[str]:
        return [self.corromper_texto(t, tasa_ruido) for t in textos]


class ExperimentoRobustezRuido:
    """Compara la degradación ante ruido léxico entre TF-IDF Caracteres y SBERT 384D."""

    def __init__(
        self,
        evaluador: EvaluadorClasificacion,
        modelo_tfidf: ClasificadorBase,
        modelo_sbert: ClasificadorBase,
        modelo_sbert_bal: ClasificadorBase,
        sbert_encoder: RepresentadorSBERT384D,
        modelo_tfidf_bal: Optional[ClasificadorBase] = None,
    ):
        self.evaluador = evaluador
        self.modelo_tfidf = modelo_tfidf
        self.modelo_sbert = modelo_sbert
        self.modelo_sbert_bal = modelo_sbert_bal
        self.modelo_tfidf_bal = modelo_tfidf_bal
        self.sbert_encoder = sbert_encoder

    def ejecutar(
        self,
        textos_test: np.ndarray,
        X_384_test_limpio: np.ndarray,
        y_test: np.ndarray,
        niveles_ruido: List[float],
        n_repeticiones: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        registros = []
        modelos_eval = [
            (self.modelo_sbert, "embeddings"),
            (self.modelo_sbert_bal, "embeddings"),
            (self.modelo_tfidf, "texto"),
        ]
        if self.modelo_tfidf_bal is not None:
            modelos_eval.append((self.modelo_tfidf_bal, "texto"))

        for nivel in niveles_ruido:
            reps = 1 if nivel == 0.0 else n_repeticiones
            print(f"    -> Evaluando ruido léxico {int(nivel * 100):2d}% ({reps} rep.)...")

            for r in range(reps):
                if nivel == 0.0:
                    textos_eval = list(textos_test)
                    X_384_eval = X_384_test_limpio
                else:
                    inyector = InyectorRuidoLexico(seed=1000 + int(nivel * 1000) + r)
                    textos_eval = inyector.corromper_corpus(list(textos_test), tasa_ruido=nivel)
                    X_384_eval = self.sbert_encoder.encode(textos_eval, show_progress_bar=False)

                for mod, tipo in modelos_eval:
                    X_in = X_384_eval if tipo == "embeddings" else textos_eval
                    res = self.evaluador.evaluar_modelo(mod, X_in, y_test)
                    registros.append({
                        "nivel_ruido_pct": int(round(nivel * 100)),
                        "rep": r,
                        "modelo": mod.nombre_corto,
                        "f1_macro": res["f1_macro"],
                        "f1_weighted": res["f1_weighted"],
                        "f1_minoritaria": res["f1_minoritaria"],
                        "kappa": res["kappa"],
                    })

        df_detalle = pd.DataFrame(registros)
        df_resumen = (
            df_detalle.groupby(["modelo", "nivel_ruido_pct"], as_index=False)
            .agg(
                f1_macro_mean=("f1_macro", "mean"),
                f1_macro_std=("f1_macro", "std"),
                f1_weighted_mean=("f1_weighted", "mean"),
                f1_weighted_std=("f1_weighted", "std"),
                f1_min_mean=("f1_minoritaria", "mean"),
                f1_min_std=("f1_minoritaria", "std"),
            )
            .fillna(0.0)
        )

        caidas = {}
        nivel_max = int(round(max(niveles_ruido) * 100))
        for mod, g in df_resumen.groupby("modelo"):
            base_m = float(g.loc[g["nivel_ruido_pct"] == 0, "f1_macro_mean"].values[0])
            fin_m = float(g.loc[g["nivel_ruido_pct"] == nivel_max, "f1_macro_mean"].values[0])
            delta_rel_m = ((fin_m - base_m) / base_m * 100.0) if base_m > 0 else 0.0

            base_min = float(g.loc[g["nivel_ruido_pct"] == 0, "f1_min_mean"].values[0])
            fin_min = float(g.loc[g["nivel_ruido_pct"] == nivel_max, "f1_min_mean"].values[0])
            delta_rel_min = ((fin_min - base_min) / base_min * 100.0) if base_min > 0 else 0.0

            caidas[mod] = {
                "f1_macro_0pct": round(base_m, 4),
                f"f1_macro_{nivel_max}pct": round(fin_m, 4),
                "delta_abs_f1_macro": round(fin_m - base_m, 4),
                "delta_rel_f1_macro_pct": round(delta_rel_m, 2),
                "f1_min_0pct": round(base_min, 4),
                f"f1_min_{nivel_max}pct": round(fin_min, 4),
                "delta_abs_f1_min": round(fin_min - base_min, 4),
                "delta_rel_f1_min_pct": round(delta_rel_min, 2),
            }

        return df_detalle, df_resumen, caidas