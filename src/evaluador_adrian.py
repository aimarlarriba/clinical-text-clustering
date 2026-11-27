"""
Módulo de evaluación científica, estabilidad estadística anidada (0% data leakage)
y análisis cualitativo de errores reales (Tema 3) - Adrián (v5).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.modelos_adrian import (
    ClasificadorBase,
    Alternativa1BaselineClustering,
    Alternativa2LogReg384D,
    Alternativa3LogRegTFIDFChar,
    Alternativa4LogReg384DBalanced,
    Alternativa5LogRegTFIDFCharBalanced,
    OptimizadorCV,
)


class EvaluadorClasificacion:
    """Calcula métricas completas del Tema 3, estabilidad anidada en 10 runs y casos de error."""

    def __init__(self, clases_ordenadas: np.ndarray, clase_minoritaria: str):
        self.clases = np.asarray(clases_ordenadas)
        self.clase_minoritaria = clase_minoritaria
        self.idx_minoritaria = int(np.where(self.clases == clase_minoritaria)[0][0])

    def _alinear_probabilidades(self, y_proba: np.ndarray, clases_modelo: np.ndarray) -> np.ndarray:
        proba_alineada = np.zeros((y_proba.shape[0], len(self.clases)), dtype=float)
        mapa = {c: idx for idx, c in enumerate(self.clases)}
        for col_modelo, cls_nombre in enumerate(clases_modelo):
            if cls_nombre in mapa:
                proba_alineada[:, mapa[cls_nombre]] = y_proba[:, col_modelo]
        return proba_alineada

    def evaluar_modelo(self, modelo: ClasificadorBase, X_eval, y_true: np.ndarray) -> Dict[str, Any]:
        y_pred = modelo.predict(X_eval)
        y_proba = self._alinear_probabilidades(modelo.predict_proba(X_eval), modelo.clases_)

        N = len(y_true)
        acc = float(accuracy_score(y_true, y_pred))
        ic95_acc = float(1.96 * np.sqrt((acc * (1.0 - acc)) / max(N, 1)))
        kappa = float(cohen_kappa_score(y_true, y_pred))

        prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
            y_true, y_pred, labels=self.clases, zero_division=0
        )

        prec_macro, rec_macro, f1_macro = float(np.mean(prec_cls)), float(np.mean(rec_cls)), float(np.mean(f1_cls))
        pesos = sup_cls / np.maximum(np.sum(sup_cls), 1)
        prec_weighted = float(np.sum(prec_cls * pesos))
        rec_weighted = float(np.sum(rec_cls * pesos))
        f1_weighted = float(np.sum(f1_cls * pesos))

        prec_min = float(prec_cls[self.idx_minoritaria])
        rec_min = float(rec_cls[self.idx_minoritaria])
        f1_min = float(f1_cls[self.idx_minoritaria])

        y_true_bin_min = (y_true == self.clase_minoritaria).astype(int)
        scores_min = y_proba[:, self.idx_minoritaria]

        try:
            auc_macro = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=self.clases))
        except ValueError:
            auc_macro = 0.0

        try:
            auc_min = float(roc_auc_score(y_true_bin_min, scores_min))
            fpr_min, tpr_min, _ = roc_curve(y_true_bin_min, scores_min)
            curva_roc_min = {"fpr": fpr_min.tolist(), "tpr": tpr_min.tolist()}
        except ValueError:
            auc_min = 0.0
            curva_roc_min = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

        try:
            ap_min = float(average_precision_score(y_true_bin_min, scores_min))
            pr_prec, pr_rec, _ = precision_recall_curve(y_true_bin_min, scores_min)
            curva_pr_min = {"precision": pr_prec.tolist(), "recall": pr_rec.tolist()}
        except ValueError:
            ap_min = 0.0
            curva_pr_min = {"precision": [0.0, 1.0], "recall": [1.0, 0.0]}

        cm = confusion_matrix(y_true, y_pred, labels=self.clases)

        por_clase = {
            str(cls_name): {
                "precision": round(float(prec_cls[i]), 4),
                "recall": round(float(rec_cls[i]), 4),
                "f1": round(float(f1_cls[i]), 4),
                "soporte": int(sup_cls[i]),
            }
            for i, cls_name in enumerate(self.clases)
        }

        return {
            "modelo": modelo.nombre,
            "modelo_corto": modelo.nombre_corto,
            "hiperparametros": modelo.hiperparametros,
            "tiempo_entrenamiento_seg": round(float(modelo.tiempo_entrenamiento_seg), 4),
            "accuracy": round(acc, 4),
            "accuracy_ic95": round(ic95_acc, 4),
            "kappa": round(kappa, 4),
            "f1_minoritaria": round(f1_min, 4),
            "precision_minoritaria": round(prec_min, 4),
            "recall_minoritaria": round(rec_min, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "roc_auc_macro": round(auc_macro, 4),
            "roc_auc_minoritaria": round(auc_min, 4),
            "ap_minoritaria": round(ap_min, 4),
            "curva_roc_minoritaria": curva_roc_min,
            "curva_pr_minoritaria": curva_pr_min,
            "por_clase": por_clase,
            "matriz_confusion": cm.tolist(),
        }

    def extraer_casos_analisis_error(
        self,
        usuarios_test: np.ndarray,
        textos_test: np.ndarray,
        X384_test: np.ndarray,
        y_test: np.ndarray,
        modelo_sbert_bal: ClasificadorBase,
        modelo_char_bal: ClasificadorBase,
        max_chars: int = 260,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extrae ejemplos textuales reales de Test para el análisis cualitativo de errores del póster:
          - falsos_negativos_minoritaria: usuarios Attempt reales no detectados por Alt. 5.
          - falsos_positivos_minoritaria: usuarios no-Attempt alertados como Attempt por Alt. 5.
          - ventaja_sin_truncamiento: textos largos donde Alt. 5 acierta y Alt. 4 (SBERT) falla.
        """
        pred_4 = modelo_sbert_bal.predict(X384_test)
        pred_5 = modelo_char_bal.predict(textos_test)
        proba_5 = self._alinear_probabilidades(modelo_char_bal.predict_proba(textos_test), modelo_char_bal.clases_)

        def _formatear(i: int) -> Dict[str, Any]:
            txt = str(textos_test[i]).replace("\n", " ").strip()
            extracto = (txt[:max_chars] + "...") if len(txt) > max_chars else txt
            return {
                "usuario": str(usuarios_test[i]),
                "real": str(y_test[i]),
                "pred_alt4_384d": str(pred_4[i]),
                "pred_alt5_char": str(pred_5[i]),
                "prob_attempt_alt5": round(float(proba_5[i, self.idx_minoritaria]), 4),
                "n_palabras": int(len(txt.split())),
                "extracto_inicio": extracto,
            }

        fn_min, fp_min, ventaja_char = [], [], []
        for i in range(len(y_test)):
            if y_test[i] == self.clase_minoritaria and pred_5[i] != self.clase_minoritaria:
                fn_min.append(_formatear(i))
            if y_test[i] != self.clase_minoritaria and pred_5[i] == self.clase_minoritaria:
                fp_min.append(_formatear(i))
            if pred_5[i] == y_test[i] and pred_4[i] != y_test[i]:
                ventaja_char.append(_formatear(i))

        ventaja_char.sort(key=lambda d: d["n_palabras"], reverse=True)
        return {
            "falsos_negativos_minoritaria": fn_min[:3],
            "falsos_positivos_minoritaria": fp_min[:3],
            "ventaja_sin_truncamiento": ventaja_char[:3],
        }

    def evaluar_estabilidad_multirun(
        self,
        textos_todos: np.ndarray,
        X_384_todos: np.ndarray,
        y_todos: np.ndarray,
        semillas: List[int],
        nested_cv: bool = True,
        params_fijos: Optional[Dict[str, dict]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evalúa estabilidad estadística en 10 particiones estratificadas 70/15/15.
        - nested_cv=True (POR DEFECTO, 0% data leakage): re-optimiza todos los hiperparámetros
          mediante 5-Fold CV estrictamente dentro del Train (70%) de cada semilla.
        - nested_cv=False: usa params_fijos (solo para depuración rápida).
        """
        indices = np.arange(len(y_todos))
        registros = []

        for r_idx, seed in enumerate(semillas, start=1):
            idx_tr, idx_temp, y_tr, y_temp = train_test_split(
                indices, y_todos, test_size=0.30, random_state=seed, stratify=y_todos
            )
            _, idx_te, _, y_te = train_test_split(
                idx_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
            )

            if nested_cv:
                print(f"    -> Run {r_idx:2d}/{len(semillas)} (seed={seed:4d}) [Nested 5-Fold CV en Train]...")
                opt_run = OptimizadorCV(clase_minoritaria=self.clase_minoritaria, n_splits=5, random_state=seed)
                cache_run = {}
                p2 = opt_run.optimizar_alternativa2(X_384_todos[idx_tr], y_tr)["mejores_params"]
                p3 = opt_run.optimizar_alternativa3(textos_todos[idx_tr], y_tr, cache_tfidf_folds=cache_run)["mejores_params"]
                p4 = opt_run.optimizar_alternativa4(X_384_todos[idx_tr], y_tr, calcular_curvas=False)["mejores_params"]
                p5 = opt_run.optimizar_alternativa5_con_puerta(textos_todos[idx_tr], y_tr, cache_tfidf_folds=cache_run)["mejores_params"]
            else:
                p2, p3, p4, p5 = (
                    params_fijos["alt2"], params_fijos["alt3"],
                    params_fijos["alt4"], params_fijos["alt5"],
                )

            modelos_run = [
                Alternativa1BaselineClustering(n_clusters=5, linkage="ward"),
                Alternativa2LogReg384D(C=p2["C"], random_state=seed),
                Alternativa3LogRegTFIDFChar(
                    C=p3["C"], analyzer=p3["analyzer"],
                    ngram_range=tuple(p3["ngram_range"]), min_df=p3["min_df"],
                    max_features=p3["max_features"],
                    limpiar_artefactos=p3["limpiar_artefactos"], random_state=seed,
                ),
                Alternativa4LogReg384DBalanced(
                    C=p4["C"], alpha_pesos=p4["alpha_pesos"], random_state=seed
                ),
                Alternativa5LogRegTFIDFCharBalanced(
                    C=p5["C"], alpha_pesos=p5["alpha_pesos"],
                    analyzer=p5["analyzer"], ngram_range=tuple(p5["ngram_range"]),
                    min_df=p5["min_df"], max_features=p5["max_features"],
                    limpiar_artefactos=p5["limpiar_artefactos"],
                    usar_estilometria=p5["usar_estilometria"],
                    peso_estilometria=p5["peso_estilometria"],
                    clase_minoritaria=self.clase_minoritaria,
                    lambda_minoritaria=p5["lambda_minoritaria"],
                    random_state=seed,
                ),
            ]

            for m in modelos_run:
                if m.tipo_entrada == "embeddings":
                    m.fit(X_384_todos[idx_tr], y_tr)
                    res = self.evaluar_modelo(m, X_384_todos[idx_te], y_te)
                else:
                    m.fit(textos_todos[idx_tr], y_tr)
                    res = self.evaluar_modelo(m, textos_todos[idx_te], y_te)

                registros.append({
                    "seed": seed,
                    "modelo_corto": m.nombre_corto,
                    "f1_minoritaria": res["f1_minoritaria"],
                    "recall_minoritaria": res["recall_minoritaria"],
                    "f1_macro": res["f1_macro"],
                    "f1_weighted": res["f1_weighted"],
                    "accuracy": res["accuracy"],
                    "kappa": res["kappa"],
                    "roc_auc_macro": res["roc_auc_macro"],
                    "roc_auc_minoritaria": res["roc_auc_minoritaria"],
                })

        df_runs = pd.DataFrame(registros)
        metricas = [
            "f1_minoritaria", "recall_minoritaria", "f1_macro", "f1_weighted",
            "accuracy", "kappa", "roc_auc_macro", "roc_auc_minoritaria",
        ]

        resumen = {}
        for mod_corto, grupo in df_runs.groupby("modelo_corto", sort=False):
            resumen[mod_corto] = {}
            for met in metricas:
                resumen[mod_corto][f"{met}_mean"] = round(float(grupo[met].mean()), 4)
                resumen[mod_corto][f"{met}_std"] = round(float(grupo[met].std(ddof=1)), 4)

        return df_runs, resumen