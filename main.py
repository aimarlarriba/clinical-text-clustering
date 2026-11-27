"""
Orquestador Principal - Práctica 2: Clasificación Supervisada de Texto (Adrián - v3)
===================================================================================
Alternativas evaluadas sobre la partición oficial 70/15/15 (seed=42, stratify=y):
  1. Baseline común: Clustering Jerárquico 384D.
  2. Regresión Logística + SBERT 384D (L2, sin balancear).
  3. Regresión Logística + TF-IDF Caracteres (sin balancear).
  4. Mejora 384D: Regresión Logística + SBERT 384D (L2, pesos balanceados α).
  5. Propuesta Síntesis: Regresión Logística + TF-IDF Caracteres (pesos balanceados α).

Uso:
    python main_adrian.py
    python main_adrian.py --reps_ruido 3   # Más rápido en RQ1 (~3 min)
"""

import sys
import json
import time
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.particion_comun import cargar_particion_oficial_equipo
from src.config_adrian import (
    RAW_DATA_FILE,
    CACHE_EMB_COMUN,
    OUTPUT_DIR,
    FIGURES_DIR,
    RANDOM_SEED_OFICIAL,
    SEEDS_ESTABILIDAD,
    SBERT_MODEL_NAME,
    SBERT_MAX_TOKENS,
    NIVELES_RUIDO_LEXICO,
    REPETICIONES_RUIDO,
    crear_directorios_salida,
)
from src.representaciones_adrian import RepresentadorSBERT384D
from src.modelos_adrian import (
    Alternativa1BaselineClustering,
    Alternativa2LogReg384D,
    Alternativa3LogRegTFIDFChar,
    Alternativa4LogReg384DBalanced,
    Alternativa5LogRegTFIDFCharBalanced,
    OptimizadorCV,
)
from src.evaluador_adrian import EvaluadorClasificacion
from src.experimento_ruido_adrian import ExperimentoRobustezRuido
from src.visualizador_adrian import VisualizadorPoster


def detectar_clase_minoritaria(y_train: np.ndarray) -> str:
    conteo = Counter(y_train)
    clase_min, _ = conteo.most_common()[-1]
    print("\nDistribución de clases en Train (70%, N=350):")
    for cls, cnt in conteo.most_common():
        marca = "  <-- CLASE MINORITARIA" if cls == clase_min else ""
        print(f"  - {cls:<12s}: {cnt:3d} ({cnt / len(y_train) * 100:5.1f}%){marca}")
    return str(clase_min)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Clasificación Supervisada - Adrián (v3)")
    parser.add_argument("--csv", type=str, default=str(RAW_DATA_FILE))
    parser.add_argument("--reps_ruido", type=int, default=REPETICIONES_RUIDO)
    args = parser.parse_args()

    t_inicio = time.time()
    crear_directorios_salida()

    print("=" * 82)
    print(" PRÁCTICA 2: CLASIFICACIÓN SUPERVISADA DE TEXTO — ADRIÁN (v3: 5-Fold CV + Síntesis)")
    print("=" * 82)

    # 1. Partición oficial común del equipo (70/15/15, seed=42, stratify=y)
    print("\n[PASO 1/6] Cargando partición oficial común del equipo...")
    datos = cargar_particion_oficial_equipo(
        ruta_csv=args.csv, ruta_cache_emb=str(CACHE_EMB_COMUN), seed=RANDOM_SEED_OFICIAL
    )
    txt_tr, txt_dev, txt_te = datos["textos"]
    X384_tr, X384_dev, X384_te = datos["X_384"]
    y_tr, y_dev, y_te = datos["y"]
    clases = datos["clases"]

    clase_minoritaria = detectar_clase_minoritaria(y_tr)
    textos_todos = np.concatenate([txt_tr, txt_dev, txt_te])
    X384_todos = np.vstack([X384_tr, X384_dev, X384_te])
    y_todos = np.concatenate([y_tr, y_dev, y_te])

    evaluador = EvaluadorClasificacion(clases_ordenadas=clases, clase_minoritaria=clase_minoritaria)

    # 2. Selección robusta de hiperparámetros mediante 5-Fold CV en Train (sin tocar Dev ni Test)
    print("\n[PASO 2/6] Optimizando hiperparámetros con 5-Fold CV Estratificado en Train (350)...")
    opt = OptimizadorCV(clase_minoritaria=clase_minoritaria, n_splits=5, random_state=RANDOM_SEED_OFICIAL)

    opt_alt2 = opt.optimizar_alternativa2(X384_tr, y_tr)
    opt_alt3 = opt.optimizar_alternativa3(txt_tr, y_tr)
    opt_alt4 = opt.optimizar_alternativa4(X384_tr, y_tr)
    opt_alt5 = opt.optimizar_alternativa5(txt_tr, y_tr, opt_alt3["mejores_params"])

    p2 = opt_alt2["mejores_params"]
    p3 = opt_alt3["mejores_params"]
    p4 = opt_alt4["mejores_params"]
    p5 = opt_alt5["mejores_params"]

    print(f"  -> Alt.2 (LogReg + 384D):       {p2} | CV F1-Macro={opt_alt2['cv_stats']['cv_f1_macro']:.4f}")
    print(f"  -> Alt.3 (LogReg + TF-IDF Char):{p3} | CV F1-Macro={opt_alt3['cv_stats']['cv_f1_macro']:.4f}")
    print(f"  -> Alt.4 (LogReg + 384D Bal.):  {p4} | CV F1-Min={opt_alt4['cv_stats']['cv_f1_min']:.4f}, F1-Macro={opt_alt4['cv_stats']['cv_f1_macro']:.4f}")
    print(f"  -> Alt.5 (Síntesis Char Bal.):  {{'C': {p5['C']}, 'alpha_pesos': {p5['alpha_pesos']}}} | CV F1-Min={opt_alt5['cv_stats']['cv_f1_min']:.4f}, F1-Macro={opt_alt5['cv_stats']['cv_f1_macro']:.4f}")

    # 3. Entrenar en Train (350) y evaluar limpiamente en Dev (75) y Test oficial (75)
    print("\n[PASO 3/6] Entrenando modelos finales en Train (350) y evaluando en Dev (75) y Test oficial (75)...")
    m1 = Alternativa1BaselineClustering(n_clusters=5, linkage="ward").fit(X384_tr, y_tr)
    m2 = Alternativa2LogReg384D(C=p2["C"], random_state=RANDOM_SEED_OFICIAL).fit(X384_tr, y_tr)
    m3 = Alternativa3LogRegTFIDFChar(
        C=p3["C"], analyzer=p3["analyzer"], ngram_range=tuple(p3["ngram_range"]),
        min_df=p3["min_df"], max_features=p3["max_features"],
        limpiar_artefactos=p3["limpiar_artefactos"], random_state=RANDOM_SEED_OFICIAL,
    ).fit(txt_tr, y_tr)
    m4 = Alternativa4LogReg384DBalanced(
        C=p4["C"], alpha_pesos=p4["alpha_pesos"], random_state=RANDOM_SEED_OFICIAL
    ).fit(X384_tr, y_tr)
    m5 = Alternativa5LogRegTFIDFCharBalanced(
        C=p5["C"], alpha_pesos=p5["alpha_pesos"],
        analyzer=p5["analyzer"], ngram_range=tuple(p5["ngram_range"]),
        min_df=p5["min_df"], max_features=p5["max_features"],
        limpiar_artefactos=p5["limpiar_artefactos"], random_state=RANDOM_SEED_OFICIAL,
    ).fit(txt_tr, y_tr)

    modelos = [m1, m2, m3, m4, m5]
    resultados_dev = [evaluador.evaluar_modelo(m, X384_dev if m.tipo_entrada == "embeddings" else txt_dev, y_dev) for m in modelos]
    resultados_test = [evaluador.evaluar_modelo(m, X384_te if m.tipo_entrada == "embeddings" else txt_te, y_te) for m in modelos]

    def _a_dataframe(lista_res):
        return pd.DataFrame([
            {
                "Alternativa": r["modelo_corto"],
                f"F1_Min ({clase_minoritaria})": r["f1_minoritaria"],
                "Rec_Min": r["recall_minoritaria"],
                "F1_Macro": r["f1_macro"],
                "F1_Weighted": r["f1_weighted"],
                "Accuracy": r["accuracy"],
                "Kappa": r["kappa"],
                "AUC_Macro": r["roc_auc_macro"],
                "AUC_Min": r["roc_auc_minoritaria"],
                "AP_Min": r["ap_minoritaria"],
            }
            for r in lista_res
        ])

    df_dev = _a_dataframe(resultados_dev)
    df_test = _a_dataframe(resultados_test)

    print("\n=== TABLA 1A: VERIFICACIÓN INDEPENDIENTE EN DEV (seed=42, N=75) ===")
    print(df_dev.to_string(index=False))
    print("\n=== TABLA 1B: RESULTADOS FINALES EN TEST OFICIAL DEL EQUIPO (seed=42, N=75) ===")
    print(df_test.to_string(index=False))
    df_dev.to_csv(OUTPUT_DIR / "tabla_dev_oficial.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "tabla_test_oficial.csv", index=False)

    # 4. Estabilidad en 10 runs estratificados (Media ± StDev)
    print(f"\n[PASO 4/6] Evaluando estabilidad estadística ({len(SEEDS_ESTABILIDAD)} runs estratificados)...")
    df_runs, resumen_estabilidad = evaluador.evaluar_estabilidad_multirun(
        textos_todos=textos_todos, X_384_todos=X384_todos, y_todos=y_todos,
        params_alt2=p2, params_alt3=p3, params_alt4=p4, params_alt5=p5, semillas=SEEDS_ESTABILIDAD,
    )
    df_estab = pd.DataFrame.from_dict(resumen_estabilidad, orient="index")
    cols_estab = [
        "f1_minoritaria_mean", "f1_minoritaria_std",
        "recall_minoritaria_mean", "recall_minoritaria_std",
        "f1_macro_mean", "f1_macro_std",
        "kappa_mean", "kappa_std",
        "roc_auc_minoritaria_mean", "roc_auc_minoritaria_std",
    ]
    print("\n=== TABLA 2: ESTABILIDAD ESTADÍSTICA (10 RUNS — MEDIA ± STDEV) ===")
    print(df_estab[cols_estab].to_string())
    df_estab.to_csv(OUTPUT_DIR / "tabla_estabilidad_10runs.csv")
    df_runs.to_csv(OUTPUT_DIR / "detalle_10runs.csv", index=False)

    # 5. Experimento RQ1 (Robustez ante Ruido Léxico)
    print("\n[PASO 5/6] Ejecutando Experimento RQ1 (Robustez ante Ruido Léxico: 0% a 30%)...")
    sbert_encoder = RepresentadorSBERT384D(nombre_modelo=SBERT_MODEL_NAME)
    exp_ruido = ExperimentoRobustezRuido(
        evaluador=evaluador, modelo_tfidf=m3, modelo_sbert=m2, modelo_sbert_bal=m4,
        sbert_encoder=sbert_encoder, modelo_tfidf_bal=m5,
    )
    df_ruido_det, df_ruido_res, caidas_rq1 = exp_ruido.ejecutar(
        textos_test=txt_te, X_384_test_limpio=X384_te, y_test=y_te,
        niveles_ruido=NIVELES_RUIDO_LEXICO, n_repeticiones=args.reps_ruido,
    )
    print("\n=== RESPUESTA RQ1 (Degradación relativa al 30% de ruido léxico) ===")
    for mod, info in caidas_rq1.items():
        print(
            f"  * {mod:<22s} | F1-Macro: {info['f1_macro_0pct']:.4f} -> {info['f1_macro_30pct']:.4f} "
            f"({info['delta_rel_f1_macro_pct']:+.1f}%)"
        )
    df_ruido_res.to_csv(OUTPUT_DIR / "tabla_ruido_lexico_rq1.csv", index=False)

    # 6. Generar las 9 figuras 100% autoexplicativas (300 DPI)
    print("\n[PASO 6/6] Generando 9 figuras autoexplicativas (300 DPI) para el póster DIN-A3...")
    viz = VisualizadorPoster(carpeta_salida=FIGURES_DIR, clases=clases, clase_minoritaria=clase_minoritaria)

    prevalencia_min_test = float(np.mean(y_te == clase_minoritaria))
    clase_contraste = "Supportive" if "Supportive" in clases else str(clases[-1])
    top_ng_min = m5.obtener_top_ngramas_unicos(clase_minoritaria, top_n=10)
    top_ng_sup = m5.obtener_top_ngramas_unicos(clase_contraste, top_n=10)

    rutas = [
        viz.fig1_diagnostico_dataset_y_truncamiento(textos_todos, y_todos, limite_tokens_sbert=SBERT_MAX_TOKENS),
        viz.fig2_comparativa_oficial(resultados_test),
        viz.fig3_estabilidad_multirun(df_runs),
        viz.fig4_matrices_confusion(resultados_test),
        viz.fig5_roc_y_pr_minoritaria(resultados_test, prevalencia_min=prevalencia_min_test),
        viz.fig6_desglose_clases_rq2(resultados_test),
        viz.fig7_robustez_ruido_rq1(df_ruido_res, caidas_rq1),
        viz.fig8_balanceo_y_sesgo_varianza(
            opt_alt4["curva_alpha_cv"], p4["alpha_pesos"],
            opt_alt4["curva_sesgo_varianza"], p4["C"],
        ),
        viz.fig9_interpretabilidad_logreg_char(top_ng_min, top_ng_sup),
    ]
    for i, rf in enumerate(rutas, start=1):
        print(f"  {i}) {rf}")

    informe = {
        "autor": "Adrián",
        "clase_minoritaria": clase_minoritaria,
        "optimizacion_cv": {"alt2": opt_alt2, "alt3": opt_alt3, "alt4": opt_alt4, "alt5": opt_alt5},
        "resultados_dev": resultados_dev,
        "resultados_test_oficial": resultados_test,
        "estabilidad_10runs": resumen_estabilidad,
        "rq1_robustez_ruido": caidas_rq1,
        "top_ngramas_minoritaria": top_ng_min,
        "top_ngramas_supportive": top_ng_sup,
    }
    with open(OUTPUT_DIR / "resultados_completos_adrian.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 82)
    print(f" [OK] FINALIZADO EN {time.time() - t_inicio:.1f}s | Figuras en: {FIGURES_DIR}")
    print("=" * 82)


if __name__ == "__main__":
    main()