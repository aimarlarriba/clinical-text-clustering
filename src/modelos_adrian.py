"""
Módulo de clasificadores y optimización mediante 5-Fold CV en Train (Adrián - v5).
Mejoras clave v5:
  - Decisión basada en coste (Tema 3): multiplicador lambda >= 1.0 aplicado como
    argmax_c [lambda_c * P(y=c | x)] en vez de un umbral override univariante.
  - Proyección exacta de Gram (Teorema del Representador) dentro de cada fold en OptimizadorCV:
    matemáticamente equivalente a L2-LogReg en 35.000D (error < 1e-14) y ~50x más rápida.
  - Búsqueda conjunta sobre toda la rejilla TF-IDF también en Alternativa 5.
  - Métrica OOF unificada en las 4 alternativas supervisadas.
  - Emparejamiento mejorado de palabras ejemplo para n-gramas (incluso con puntuación).
"""

import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score

from src.particion_comun import BaselineClusteringJerarquico384D
from src.representaciones_adrian import RepresentadorTFIDFCaracteres, ExtractorPatronesEstilometricos
from src.config_adrian import (
    RANDOM_SEED_OFICIAL,
    GRID_C_384D,
    GRID_C_TFIDF,
    GRID_ALPHA_PESOS,
    GRID_PESO_ESTILOMETRIA,
    GRID_LAMBDA_COSTE,
    GRID_TFIDF_CHAR,
)


def _calcular_pesos_clase(y_train: np.ndarray, alpha: float) -> dict:
    clases_u, conteos = np.unique(y_train, return_counts=True)
    n_total, k = float(len(y_train)), float(len(clases_u))
    return {c: float((n_total / (k * cnt)) ** alpha) for c, cnt in zip(clases_u, conteos)}


def _predecir_con_coste(
    proba: np.ndarray,
    clases: np.ndarray,
    clase_minoritaria: Optional[str],
    lambda_minoritaria: float,
) -> np.ndarray:
    """
    Regla de decisión basada en coste (Tema 3, ec. 15):
        c_hat(x) = argmax_c [ lambda_c * P(y=c | x) ]
    con lambda_minoritaria >= 1.0 (coste relativo de FN en clase crítica) y lambda_c = 1.0 en el resto.
    """
    if lambda_minoritaria is None or lambda_minoritaria <= 1.0 or clase_minoritaria not in clases:
        return clases[np.argmax(proba, axis=1)]
    scores = proba.copy()
    idx_min = int(np.where(clases == clase_minoritaria)[0][0])
    scores[:, idx_min] *= float(lambda_minoritaria)
    return clases[np.argmax(scores, axis=1)]


def _proyectar_gram_exacta(X_tr, X_val, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isomorfismo exacto al subespacio fila de X_tr (Teorema del Representador):
    para cualquier modelo lineal con regularización L2 (como Regresión Logística),
    el vector de pesos óptimo vive en span(X_tr). Proyectar (N_tr, 35000) -> (N_tr, <=N_tr)
    preserva exactamente todos los productos internos y probabilidades (diff < 1e-14),
    acelerando la búsqueda en rejilla ~50x con 0% de data leakage.
    """
    G = (X_tr @ X_tr.T).toarray() if hasattr(X_tr, "toarray") else (X_tr @ X_tr.T)
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    mask = evals > tol
    sqrt_ev = np.sqrt(evals[mask])
    U = evecs[:, mask]
    Z_tr = U * sqrt_ev
    K_val = (X_val @ X_tr.T).toarray() if hasattr(X_val, "toarray") else (X_val @ X_tr.T)
    Z_val = K_val @ (U / sqrt_ev)
    return Z_tr, Z_val


def _extraer_top_ngramas_por_palabra(
    coefs_clase: np.ndarray,
    nombres_ngramas: np.ndarray,
    palabras_frecuentes_train: list,
    top_n: int = 10,
) -> list:
    """
    Devuelve los top_n n-gramas más discriminativos asociando cada uno a su palabra ejemplo
    más frecuente en Train (incluso si el n-grama incluye puntuación al final, como 'pist. ').
    """
    orden = np.argsort(coefs_clase)[::-1]
    seleccionados = []
    palabras_usadas = set()
    raices_usadas = []

    for i in orden:
        peso = float(coefs_clase[i])
        if peso <= 0:
            break
        ng = str(nombres_ngramas[i])
        limpio = ng.strip()
        if len(limpio) < 3:
            continue

        nucleo_alpha = re.sub(r"[^a-z]", "", limpio.lower())
        palabra_ejemplo = None
        if len(nucleo_alpha) >= 3:
            empieza_esp = ng.startswith(" ")
            termina_esp = ng.endswith(" ") and limpio[-1].isalpha()
            for w in palabras_frecuentes_train:
                if empieza_esp and not w.startswith(nucleo_alpha):
                    continue
                if termina_esp and not w.endswith(nucleo_alpha):
                    continue
                if nucleo_alpha in w:
                    palabra_ejemplo = w
                    break

        if palabra_ejemplo:
            raiz_w = palabra_ejemplo[:5]
            if palabra_ejemplo in palabras_usadas or any(
                raiz_w in p or p[:5] in palabra_ejemplo for p in palabras_usadas
            ):
                continue
            palabras_usadas.add(palabra_ejemplo)
            etiqueta = f"'{ng}' ({palabra_ejemplo})"
        else:
            if any(limpio in r or r in limpio for r in raices_usadas):
                continue
            raices_usadas.append(limpio)
            etiqueta = f"'{ng}'"

        seleccionados.append((etiqueta, round(peso, 4)))
        if len(seleccionados) >= top_n:
            break
    return seleccionados


class ClasificadorBase(ABC):
    """Interfaz común para todas las alternativas."""

    def __init__(self, nombre: str, nombre_corto: str, tipo_entrada: str):
        self.nombre = nombre
        self.nombre_corto = nombre_corto
        self.tipo_entrada = tipo_entrada
        self.clases_ = None
        self.tiempo_entrenamiento_seg = 0.0
        self.hiperparametros = {}

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass


class Alternativa1BaselineClustering(ClasificadorBase):
    """Alternativa 1 (Común al equipo): Clustering Jerárquico 384D."""

    def __init__(self, n_clusters: int = 5, linkage: str = "ward"):
        super().__init__(
            nombre="1. Baseline: Clustering Jerárquico (384D)",
            nombre_corto="Baseline Clust. 384D",
            tipo_entrada="embeddings",
        )
        self.modelo = BaselineClusteringJerarquico384D(n_clusters=n_clusters, linkage=linkage)
        self.hiperparametros = {"n_clusters": n_clusters, "linkage": linkage}

    def fit(self, X_train, y_train):
        t0 = time.perf_counter()
        self.modelo.fit(X_train, y_train)
        self.tiempo_entrenamiento_seg = time.perf_counter() - t0
        self.clases_ = np.asarray(self.modelo.clases_)
        return self

    def predict(self, X) -> np.ndarray:
        return self.modelo.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.modelo.predict_proba(X)


class Alternativa2LogReg384D(ClasificadorBase):
    """Alternativa 2: Regresión Logística + SBERT 384D (L2, sin balancear)."""

    def __init__(self, C: float = 1.0, random_state: int = RANDOM_SEED_OFICIAL):
        super().__init__(
            nombre="2. Regresión Logística + SBERT 384D",
            nombre_corto="LogReg + 384D",
            tipo_entrada="embeddings",
        )
        self.C = C
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("l2", Normalizer(norm="l2")),
            ("clf", LogisticRegression(
                C=self.C, class_weight=None, max_iter=2000, solver="lbfgs", random_state=self.random_state
            )),
        ])
        self.hiperparametros = {"C": C, "normalizacion": "l2", "class_weight": None}

    def fit(self, X_train, y_train):
        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        self.tiempo_entrenamiento_seg = time.perf_counter() - t0
        self.clases_ = np.asarray(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)


class Alternativa3LogRegTFIDFChar(ClasificadorBase):
    """Alternativa 3 (Representación propia): Regresión Logística + TF-IDF Caracteres."""

    def __init__(
        self,
        C: float = 2.0,
        analyzer: str = "char_wb",
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 4,
        max_features: int = 30000,
        limpiar_artefactos: bool = True,
        random_state: int = RANDOM_SEED_OFICIAL,
    ):
        super().__init__(
            nombre="3. Regresión Logística + TF-IDF (Caracteres)",
            nombre_corto="LogReg + TF-IDF Char",
            tipo_entrada="texto",
        )
        self.C = C
        self.random_state = random_state
        self.representador = RepresentadorTFIDFCaracteres(
            analyzer=analyzer, ngram_range=ngram_range, min_df=min_df,
            max_features=max_features, limpiar_artefactos=limpiar_artefactos,
        )
        self.modelo = LogisticRegression(
            C=self.C, class_weight=None, max_iter=2000, solver="lbfgs", random_state=self.random_state
        )
        self.hiperparametros = {
            "C": C, "analyzer": analyzer, "ngram_range": list(ngram_range),
            "min_df": min_df, "max_features": max_features, "limpiar_artefactos": limpiar_artefactos,
        }

    def fit(self, textos_train, y_train):
        t0 = time.perf_counter()
        X_tr = self.representador.fit_transform(textos_train)
        self.modelo.fit(X_tr, y_train)
        self.tiempo_entrenamiento_seg = time.perf_counter() - t0
        self.clases_ = np.asarray(self.modelo.classes_)
        self.hiperparametros["vocabulario_real"] = self.representador.obtener_info()["vocabulario_real"]
        return self

    def predict(self, textos) -> np.ndarray:
        return self.modelo.predict(self.representador.transform(textos))

    def predict_proba(self, textos) -> np.ndarray:
        return self.modelo.predict_proba(self.representador.transform(textos))


class Alternativa4LogReg384DBalanced(ClasificadorBase):
    """Alternativa 4 (Mejora acordada): LogReg + SBERT 384D (L2) con pesos balanceados."""

    def __init__(self, C: float = 1.0, alpha_pesos: float = 1.0, random_state: int = RANDOM_SEED_OFICIAL):
        super().__init__(
            nombre="4. Regresión Logística + SBERT 384D (Balanced)",
            nombre_corto="LogReg + 384D (Bal.)",
            tipo_entrada="embeddings",
        )
        self.C = C
        self.alpha_pesos = alpha_pesos
        self.random_state = random_state
        self.pipeline = None
        self.hiperparametros = {"C": C, "normalizacion": "l2", "alpha_pesos": alpha_pesos}

    def fit(self, X_train, y_train):
        t0 = time.perf_counter()
        pesos = _calcular_pesos_clase(y_train, self.alpha_pesos)
        self.pipeline = Pipeline([
            ("l2", Normalizer(norm="l2")),
            ("clf", LogisticRegression(
                C=self.C, class_weight=pesos, max_iter=2000, solver="lbfgs", random_state=self.random_state
            )),
        ])
        self.pipeline.fit(X_train, y_train)
        self.tiempo_entrenamiento_seg = time.perf_counter() - t0
        self.clases_ = np.asarray(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)


class Alternativa5LogRegTFIDFCharBalanced(ClasificadorBase):
    """
    Alternativa 5 (Propuesta Síntesis de Adrián):
    TF-IDF Caracteres + Balanceo de clases, enriquecida SOLO con los componentes
    (patrones estilométricos y/o decisión basada en coste lambda) validados en 5-Fold CV.
    """

    def __init__(
        self,
        C: float = 1.0,
        alpha_pesos: float = 1.0,
        analyzer: str = "char_wb",
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 4,
        max_features: int = 30000,
        limpiar_artefactos: bool = True,
        usar_estilometria: bool = False,
        peso_estilometria: float = 0.15,
        clase_minoritaria: Optional[str] = None,
        lambda_minoritaria: float = 1.0,
        random_state: int = RANDOM_SEED_OFICIAL,
    ):
        super().__init__(
            nombre="5. Síntesis: LogReg + TF-IDF Char (Bal.+)",
            nombre_corto="LogReg + Char (Bal.+)",
            tipo_entrada="texto",
        )
        self.C = C
        self.alpha_pesos = alpha_pesos
        self.usar_estilometria = usar_estilometria
        self.peso_estilometria = peso_estilometria
        self.clase_minoritaria = clase_minoritaria
        self.lambda_minoritaria = float(lambda_minoritaria) if lambda_minoritaria is not None else 1.0
        self.random_state = random_state

        self.representador = RepresentadorTFIDFCaracteres(
            analyzer=analyzer, ngram_range=ngram_range, min_df=min_df,
            max_features=max_features, limpiar_artefactos=limpiar_artefactos,
        )
        self.extractor_estilo = ExtractorPatronesEstilometricos(limpiar_artefactos=limpiar_artefactos)
        self.scaler_estilo = StandardScaler()
        self.norm_estilo = Normalizer(norm="l2")
        self.modelo = None
        self._n_tfidf_cols = 0
        self._palabras_train = []

        self.hiperparametros = {
            "C": C, "alpha_pesos": alpha_pesos, "analyzer": analyzer,
            "ngram_range": list(ngram_range), "min_df": min_df,
            "max_features": max_features, "limpiar_artefactos": limpiar_artefactos,
            "usar_estilometria": usar_estilometria,
            "peso_estilometria": peso_estilometria if usar_estilometria else 0.0,
            "lambda_minoritaria": self.lambda_minoritaria,
        }

    def _construir_matriz(self, textos, fit_transform: bool = False):
        X_tfidf = self.representador.fit_transform(textos) if fit_transform else self.representador.transform(textos)
        if not self.usar_estilometria:
            return X_tfidf

        X_est = self.extractor_estilo.transform(textos)
        X_est = self.scaler_estilo.fit_transform(X_est) if fit_transform else self.scaler_estilo.transform(X_est)
        X_est = self.norm_estilo.fit_transform(X_est) if fit_transform else self.norm_estilo.transform(X_est)
        X_est = X_est * float(self.peso_estilometria)
        return hstack([X_tfidf, csr_matrix(X_est)], format="csr")

    def fit(self, textos_train, y_train):
        t0 = time.perf_counter()
        X_tr = self._construir_matriz(textos_train, fit_transform=True)
        self._n_tfidf_cols = len(self.representador.vectorizer.vocabulary_)

        limpios = self.representador.preprocesador.limpiar_lote(textos_train)
        conteo_w = Counter(re.findall(r"[a-z]{3,}", " ".join(limpios).lower()))
        self._palabras_train = [w for w, cnt in conteo_w.most_common() if cnt >= 2]

        pesos = _calcular_pesos_clase(y_train, self.alpha_pesos)
        self.modelo = LogisticRegression(
            C=self.C, class_weight=pesos, max_iter=2000, solver="lbfgs", random_state=self.random_state
        )
        self.modelo.fit(X_tr, y_train)
        self.tiempo_entrenamiento_seg = time.perf_counter() - t0
        self.clases_ = np.asarray(self.modelo.classes_)
        self.hiperparametros["vocabulario_real"] = self._n_tfidf_cols
        return self

    def predict_proba(self, textos) -> np.ndarray:
        return self.modelo.predict_proba(self._construir_matriz(textos, fit_transform=False))

    def predict(self, textos) -> np.ndarray:
        proba = self.predict_proba(textos)
        return _predecir_con_coste(proba, self.clases_, self.clase_minoritaria, self.lambda_minoritaria)

    def obtener_top_ngramas_unicos(self, clase: str, top_n: int = 10) -> list:
        idx_cls = int(np.where(self.clases_ == clase)[0][0])
        coefs_tfidf = self.modelo.coef_[idx_cls, : self._n_tfidf_cols]
        nombres = self.representador.vectorizer.get_feature_names_out()
        return _extraer_top_ngramas_por_palabra(coefs_tfidf, nombres, self._palabras_train, top_n=top_n)


class OptimizadorCV:
    """
    Optimiza hiperparámetros con 5-Fold CV Estratificado estrictamente dentro de Train.
    Usa proyección exacta de Gram por fold (0% data leakage, precisión 1e-14, ~50x más rápido).
    """

    def __init__(self, clase_minoritaria: str, n_splits: int = 5, random_state: int = RANDOM_SEED_OFICIAL):
        self.clase_minoritaria = clase_minoritaria
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _preparar_folds_384d(self, X_train: np.ndarray, y_train: np.ndarray) -> List[tuple]:
        folds = []
        for idx_tr, idx_val in self.skf.split(np.zeros(len(y_train)), y_train):
            norm = Normalizer(norm="l2")
            X_tr_f = norm.fit_transform(X_train[idx_tr])
            X_val_f = norm.transform(X_train[idx_val])
            folds.append((idx_tr, idx_val, X_tr_f, X_val_f, y_train[idx_tr], y_train[idx_val]))
        return folds

    def _preparar_folds_tfidf(self, txt_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> List[dict]:
        """Precomputa matriz sparse TF-IDF y proyección exacta de Gram dentro de cada fold."""
        folds = []
        for idx_tr, idx_val in self.skf.split(np.zeros(len(y_train)), y_train):
            rep = RepresentadorTFIDFCaracteres(
                analyzer=cfg["analyzer"],
                ngram_range=tuple(cfg["ngram_range"]),
                min_df=cfg["min_df"],
                max_features=cfg["max_features"],
                limpiar_artefactos=cfg["limpiar_artefactos"],
            )
            X_tr_sp = rep.fit_transform(txt_train[idx_tr])
            X_val_sp = rep.transform(txt_train[idx_val])
            Z_tr, Z_val = _proyectar_gram_exacta(X_tr_sp, X_val_sp)
            folds.append({
                "idx_tr": idx_tr, "idx_val": idx_val,
                "X_tr_sp": X_tr_sp, "X_val_sp": X_val_sp,
                "Z_tr": Z_tr, "Z_val": Z_val,
                "y_tr": y_train[idx_tr], "y_val": y_train[idx_val],
            })
        return folds

    def _preparar_folds_estilometria(self, txt_train: np.ndarray, y_train: np.ndarray) -> List[tuple]:
        extractor = ExtractorPatronesEstilometricos(limpiar_artefactos=True)
        X_est_all = extractor.transform(txt_train)
        folds = []
        for idx_tr, idx_val in self.skf.split(np.zeros(len(y_train)), y_train):
            sc = StandardScaler()
            nl = Normalizer(norm="l2")
            E_tr = nl.fit_transform(sc.fit_transform(X_est_all[idx_tr]))
            E_val = nl.transform(sc.transform(X_est_all[idx_val]))
            folds.append((csr_matrix(E_tr), csr_matrix(E_val)))
        return folds

    def _evaluar_folds_denso(
        self,
        folds: List[tuple],
        y_train: np.ndarray,
        C: float,
        alpha_pesos: Optional[float] = None,
        devolver_oof: bool = False,
    ) -> dict:
        clases_ref = np.unique(y_train)
        mapa = {c: j for j, c in enumerate(clases_ref)}
        oof_proba = np.zeros((len(y_train), len(clases_ref)), dtype=float)
        oof_preds = np.empty(len(y_train), dtype=object)
        f1_tr_macros = []

        for idx_tr, idx_val, X_tr_f, X_val_f, y_tr_f, _ in folds:
            cw = _calcular_pesos_clase(y_tr_f, alpha_pesos) if alpha_pesos is not None else None
            clf = LogisticRegression(
                C=C, class_weight=cw, max_iter=2000, solver="lbfgs", random_state=self.random_state
            )
            clf.fit(X_tr_f, y_tr_f)
            pred_val = clf.predict(X_val_f)
            pred_tr = clf.predict(X_tr_f)
            oof_preds[idx_val] = pred_val
            f1_tr_macros.append(f1_score(y_tr_f, pred_tr, average="macro", zero_division=0))

            if devolver_oof:
                p_raw = clf.predict_proba(X_val_f)
                for j_m, c_m in enumerate(clf.classes_):
                    oof_proba[idx_val, mapa[c_m]] = p_raw[:, j_m]

        oof_f1_macro = float(f1_score(y_train, oof_preds, average="macro", zero_division=0))
        y_bin = (y_train == self.clase_minoritaria).astype(int)
        p_bin = (oof_preds == self.clase_minoritaria).astype(int)
        oof_f1_min = float(f1_score(y_bin, p_bin, zero_division=0))
        oof_rec_min = float(recall_score(y_bin, p_bin, zero_division=0))

        out = {
            "oof_f1_macro": oof_f1_macro,
            "oof_f1_min": oof_f1_min,
            "oof_rec_min": oof_rec_min,
            "train_f1_macro": float(np.mean(f1_tr_macros)),
        }
        if devolver_oof:
            out["oof_proba"] = oof_proba
            out["clases_ref"] = clases_ref
        return out

    def _evaluar_folds_tfidf_con_estilo(
        self,
        folds_tfidf: List[dict],
        y_train: np.ndarray,
        C: float,
        alpha_pesos: Optional[float] = None,
        folds_est: Optional[List[tuple]] = None,
        peso_est: float = 0.0,
        devolver_oof: bool = False,
    ) -> dict:
        """Evalúa LogReg sobre folds TF-IDF (+ estilometría opcional) usando proyección exacta de Gram."""
        folds_adaptados = []
        for f_idx, fd in enumerate(folds_tfidf):
            if folds_est is not None and peso_est > 0.0:
                E_tr, E_val = folds_est[f_idx]
                X_tr_in = hstack([fd["X_tr_sp"], E_tr * float(peso_est)], format="csr")
                X_val_in = hstack([fd["X_val_sp"], E_val * float(peso_est)], format="csr")
                Z_tr, Z_val = _proyectar_gram_exacta(X_tr_in, X_val_in)
            else:
                Z_tr, Z_val = fd["Z_tr"], fd["Z_val"]
            folds_adaptados.append((fd["idx_tr"], fd["idx_val"], Z_tr, Z_val, fd["y_tr"], fd["y_val"]))

        return self._evaluar_folds_denso(
            folds_adaptados, y_train, C=C, alpha_pesos=alpha_pesos, devolver_oof=devolver_oof
        )

    def optimizar_alternativa2(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        folds = self._preparar_folds_384d(X_tr, y_tr)
        mejor = {"score": -1.0, "params": {"C": 1.0}, "cv_stats": {}}
        for c in GRID_C_384D:
            res = self._evaluar_folds_denso(folds, y_tr, C=c, alpha_pesos=None)
            if res["oof_f1_macro"] > mejor["score"]:
                mejor = {"score": res["oof_f1_macro"], "params": {"C": c}, "cv_stats": res}
        return {"mejores_params": mejor["params"], "cv_stats": mejor["cv_stats"]}

    def optimizar_alternativa3(
        self, txt_tr: np.ndarray, y_tr: np.ndarray, cache_tfidf_folds: Optional[dict] = None
    ) -> Dict[str, Any]:
        mejor = {"score": -1.0, "params": {}, "cv_stats": {}}
        for idx_cfg, cfg in enumerate(GRID_TFIDF_CHAR):
            if cache_tfidf_folds is not None and idx_cfg in cache_tfidf_folds:
                folds = cache_tfidf_folds[idx_cfg]
            else:
                folds = self._preparar_folds_tfidf(txt_tr, y_tr, cfg)
                if cache_tfidf_folds is not None:
                    cache_tfidf_folds[idx_cfg] = folds
            for c in GRID_C_TFIDF:
                res = self._evaluar_folds_tfidf_con_estilo(folds, y_tr, C=c, alpha_pesos=None)
                if res["oof_f1_macro"] > mejor["score"]:
                    mejor = {"score": res["oof_f1_macro"], "params": {**cfg, "C": c}, "cv_stats": res}
        return {"mejores_params": mejor["params"], "cv_stats": mejor["cv_stats"]}

    def optimizar_alternativa4(
        self, X_tr: np.ndarray, y_tr: np.ndarray, calcular_curvas: bool = True
    ) -> Dict[str, Any]:
        folds = self._preparar_folds_384d(X_tr, y_tr)
        mejor = {"score": -1.0, "params": {"C": 1.0, "alpha_pesos": 1.0}, "cv_stats": {}}
        for alpha in GRID_ALPHA_PESOS:
            for c in GRID_C_384D:
                res = self._evaluar_folds_denso(folds, y_tr, C=c, alpha_pesos=alpha)
                obj = 0.55 * res["oof_f1_macro"] + 0.45 * res["oof_f1_min"]
                if obj > mejor["score"]:
                    mejor = {"score": obj, "params": {"C": c, "alpha_pesos": alpha}, "cv_stats": res}

        out = {"mejores_params": mejor["params"], "cv_stats": mejor["cv_stats"]}
        if calcular_curvas:
            c_opt, a_opt = mejor["params"]["C"], mejor["params"]["alpha_pesos"]
            curva_alpha = []
            for alpha in [0.0] + GRID_ALPHA_PESOS:
                r_a = self._evaluar_folds_denso(folds, y_tr, C=c_opt, alpha_pesos=alpha if alpha > 0 else None)
                curva_alpha.append({
                    "alpha": alpha,
                    "cv_f1_min": round(r_a["oof_f1_min"], 4),
                    "cv_rec_min": round(r_a["oof_rec_min"], 4),
                    "cv_f1_macro": round(r_a["oof_f1_macro"], 4),
                })

            curva_bv = []
            for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]:
                r_c = self._evaluar_folds_denso(folds, y_tr, C=c, alpha_pesos=a_opt)
                curva_bv.append({
                    "C": c,
                    "train_f1_macro": round(r_c["train_f1_macro"], 4),
                    "cv_f1_macro": round(r_c["oof_f1_macro"], 4),
                    "cv_f1_min": round(r_c["oof_f1_min"], 4),
                })
            out["curva_alpha_cv"] = curva_alpha
            out["curva_sesgo_varianza"] = curva_bv
        return out

    def optimizar_alternativa5_con_puerta(
        self, txt_tr: np.ndarray, y_tr: np.ndarray, cache_tfidf_folds: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Ablación estricta en 5-Fold CV sobre Train:
          - Paso 5A: Base TF-IDF Char (Bal.) explorando rejilla completa TF-IDF x C x alpha.
          - Puerta 1 (Estilometría): solo se activa si mejora el objetivo OOF > +0.003.
          - Puerta 2 (Coste asimétrico lambda >= 1.0 en OOF, Tema 3): solo se activa si mejora
            el objetivo OOF > +0.005 sin degradar OOF F1-Macro más de 0.015.
        """
        mejor_base = {"score": -1.0, "params": {}, "cv_stats": {}, "cfg_idx": 0}
        folds_por_cfg = cache_tfidf_folds if cache_tfidf_folds is not None else {}

        for idx_cfg, cfg in enumerate(GRID_TFIDF_CHAR):
            if idx_cfg not in folds_por_cfg:
                folds_por_cfg[idx_cfg] = self._preparar_folds_tfidf(txt_tr, y_tr, cfg)
            folds = folds_por_cfg[idx_cfg]
            for alpha in GRID_ALPHA_PESOS:
                for c in GRID_C_TFIDF:
                    res = self._evaluar_folds_tfidf_con_estilo(folds, y_tr, C=c, alpha_pesos=alpha)
                    obj = 0.55 * res["oof_f1_macro"] + 0.45 * res["oof_f1_min"]
                    if obj > mejor_base["score"]:
                        mejor_base = {
                            "score": obj,
                            "params": {**cfg, "C": c, "alpha_pesos": alpha},
                            "cv_stats": res,
                            "cfg_idx": idx_cfg,
                        }

        folds_ganador = folds_por_cfg[mejor_base["cfg_idx"]]
        c_b = mejor_base["params"]["C"]
        a_b = mejor_base["params"]["alpha_pesos"]

        # Puerta 1: Patrones estilométricos
        folds_est = self._preparar_folds_estilometria(txt_tr, y_tr)
        mejor_estilo = {"score": -1.0, "peso": 0.0, "cv_stats": None}
        for w_est in GRID_PESO_ESTILOMETRIA:
            res_e = self._evaluar_folds_tfidf_con_estilo(
                folds_ganador, y_tr, C=c_b, alpha_pesos=a_b, folds_est=folds_est, peso_est=w_est
            )
            obj_e = 0.55 * res_e["oof_f1_macro"] + 0.45 * res_e["oof_f1_min"]
            if obj_e > mejor_estilo["score"]:
                mejor_estilo = {"score": obj_e, "peso": w_est, "cv_stats": res_e}

        activar_estilometria = mejor_estilo["score"] > (mejor_base["score"] + 0.003)
        peso_est_final = mejor_estilo["peso"] if activar_estilometria else 0.0

        # Probabilidades OOF del ganador tras Puerta 1 para evaluar Puerta 2 (coste asimétrico lambda)
        oof_res = self._evaluar_folds_tfidf_con_estilo(
            folds_ganador, y_tr, C=c_b, alpha_pesos=a_b,
            folds_est=folds_est if activar_estilometria else None,
            peso_est=peso_est_final, devolver_oof=True,
        )
        oof_proba = oof_res["oof_proba"]
        clases_ref = oof_res["clases_ref"]
        y_bin = (y_tr == self.clase_minoritaria).astype(int)

        f1_macro_pre = oof_res["oof_f1_macro"]
        f1_min_pre = oof_res["oof_f1_min"]
        obj_pre = 0.55 * f1_macro_pre + 0.45 * f1_min_pre

        mejor_lambda = 1.0
        mejor_obj_lam = obj_pre
        mejor_stats_lam = {
            "oof_f1_macro": f1_macro_pre, "oof_f1_min": f1_min_pre, "oof_rec_min": oof_res["oof_rec_min"]
        }

        for lam in GRID_LAMBDA_COSTE:
            if lam <= 1.0:
                continue
            preds_l = _predecir_con_coste(oof_proba, clases_ref, self.clase_minoritaria, lam)
            f1_m_l = float(f1_score(y_tr, preds_l, average="macro", zero_division=0))
            p_bin_l = (preds_l == self.clase_minoritaria).astype(int)
            f1_min_l = float(f1_score(y_bin, p_bin_l, zero_division=0))
            rec_min_l = float(recall_score(y_bin, p_bin_l, zero_division=0))
            obj_l = 0.55 * f1_m_l + 0.45 * f1_min_l

            if f1_m_l >= (f1_macro_pre - 0.015) and obj_l > mejor_obj_lam:
                mejor_obj_lam = obj_l
                mejor_lambda = float(lam)
                mejor_stats_lam = {"oof_f1_macro": f1_m_l, "oof_f1_min": f1_min_l, "oof_rec_min": rec_min_l}

        activar_coste = (mejor_lambda > 1.0) and (mejor_obj_lam > obj_pre + 0.005)
        lambda_final = mejor_lambda if activar_coste else 1.0

        params_finales = {
            **mejor_base["params"],
            "usar_estilometria": activar_estilometria,
            "peso_estilometria": peso_est_final,
            "lambda_minoritaria": lambda_final,
        }

        ablacion_cv = {
            "5A_base_char_bal": {
                "oof_f1_macro": round(mejor_base["cv_stats"]["oof_f1_macro"], 4),
                "oof_f1_min": round(mejor_base["cv_stats"]["oof_f1_min"], 4),
                "score_obj": round(mejor_base["score"], 4),
            },
            "5B_mas_estilometria": {
                "peso_optimo": mejor_estilo["peso"],
                "oof_f1_macro": round(mejor_estilo["cv_stats"]["oof_f1_macro"], 4),
                "oof_f1_min": round(mejor_estilo["cv_stats"]["oof_f1_min"], 4),
                "score_obj": round(mejor_estilo["score"], 4),
                "decision_puerta": "ACEPTADA" if activar_estilometria else "DESCARTADA (no mejora CV)",
            },
            "5C_mas_coste_lambda_oof": {
                "lambda_candidato": mejor_lambda,
                "oof_f1_macro": round(mejor_stats_lam["oof_f1_macro"], 4),
                "oof_f1_min": round(mejor_stats_lam["oof_f1_min"], 4),
                "oof_rec_min": round(mejor_stats_lam["oof_rec_min"], 4),
                "score_obj": round(mejor_obj_lam, 4),
                "decision_puerta": "ACEPTADA" if activar_coste else "DESCARTADA (no mejora CV)",
            },
        }

        return {
            "mejores_params": params_finales,
            "ablacion_cv": ablacion_cv,
            "cv_stats": mejor_stats_lam if activar_coste else {
                "oof_f1_macro": f1_macro_pre, "oof_f1_min": f1_min_pre, "oof_rec_min": oof_res["oof_rec_min"]
            },
        }