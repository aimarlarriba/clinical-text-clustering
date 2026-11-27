"""
Orquestador Principal - Práctica 2: Clasificación Supervisada de Texto (Adrián - v5)
===================================================================================
Novedades v5:
  - Sin dependencias rotas: genera las 7 figuras editoriales directamente con
    GeneradorPosterAesthetic (generar_graficos_poster.py).
  - Conteo real de tokens WordPiece de all-MiniLM-L6-v2 (sin heurística x1.3).
  - Decisión basada en coste (lambda >= 1.0, Tema 3) + búsqueda completa de TF-IDF en Alt. 5.
  - Nested 5-Fold CV en los 10 runs por defecto (0% data leakage real; ~2 min gracias a Gram exacto).
  - Extracción automática de casos textuales reales de análisis de error en Test.

Uso:
    python main_adrian.py
    python main_adrian.py --reps_ruido 5
    python main_adrian.py --sin_nested   (solo depuración rápida con hiperparámetros fijos)
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
from src.representaciones_adrian import (
    PreprocesadorTextoClinico,
    RepresentadorSBERT384D,
    ExtractorPatronesEstilometricos,
    calcular_estadisticas_corpus,
)
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
from generar_graficos_poster import GeneradorPosterAesthetic


def detectar_clase_minoritaria(y_train: np.ndarray) -> str:
    conteo = Counter(y_train)
    clase_min, _ = conteo.most_common()[-1]
    print("\nDistribución de clases en Train (70%, N=350):")
    for cls, cnt in conteo.most_common():
        marca = "  <-- CLASE MINORITARIA" if cls == clase_min else ""
        print(f"  - {cls:<12s}: {cnt:3d} ({cnt / len(y_train) * 100:5.1f}%){marca}")
    return str(clase_min)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Clasificación Supervisada - Adrián (v5)")
    parser.add_argument("--csv", type=str, default=str(RAW_DATA_FILE))
    parser.add_argument("--reps_ruido", type=int, default=REPETICIONES_RUIDO)
    parser.add_argument(
        "--sin_nested",
        action="store_true",
        help="Desactiva Nested 5-Fold CV en los 10 runs (usa hiperparámetros de seed=42; solo depuración).",
    )
    args = parser.parse_args()

    t_inicio = time.time()
    crear_directorios_salida()

    print("=" * 84)
    print(" PRÁCTICA 2: CLASIFICACIÓN SUPERVISADA — ADRIÁN (v5: Rigor + Coste + Nested CV)")
    print("=" * 84)

    # 1. Partición oficial común del equipo
    print("\n[PASO 1/7] Cargando partición oficial común (70/15/15, seed=42, stratify=y)...")
    datos = cargar_particion_oficial_equipo(
        ruta_csv=args.csv, ruta_cache_emb=str(CACHE_EMB_COMUN), seed=RANDOM_SEED_OFICIAL
    )
    usr_tr, usr_dev, usr_te = datos["usuarios"]
    txt_tr, txt_dev, txt_te = datos["textos"]
    X384_tr, X384_dev, X384_te = datos["X_384"]
    y_tr, y_dev, y_te = datos["y"]
    clases = datos["clases"]

    clase_minoritaria = detectar_clase_minoritaria(y_tr)
    textos_todos = np.concatenate([txt_tr, txt_dev, txt_te])
    X384_todos = np.vstack([X384_tr, X384_dev, X384_te])
    y_todos = np.concatenate([y_tr, y_dev, y_te])

    # Estadísticas exactas del corpus y conteo real de tokens WordPiece de SBERT
    stats_corpus = calcular_estadisticas_corpus(textos_todos)
    stats_artefactos = PreprocesadorTextoClinico().contar_artefactos_zenodo(textos_todos)
    sbert_encoder = RepresentadorSBERT384D(nombre_modelo=SBERT_MODEL_NAME)
    tokens_sbert = sbert_encoder.medir_tokens_reales(textos_todos, limite_tokens=SBERT_MAX_TOKENS)
    print(
        f"  * Corpus: |Sigma|={stats_corpus['vocabulario_palabras_unico']:,} palabras únicas | "
        f"Longitud: {stats_corpus['palabras_media']:.1f} ± {stats_corpus['palabras_std']:.1f} pal."
    )
    print(
        f"  * Truncamiento REAL WordPiece (>256 tok.): {tokens_sbert['n_truncados']}/{len(textos_todos)} "
        f"({tokens_sbert['pct_truncados']:.1f}%) | Mediana={tokens_sbert['mediana_tokens']} | Máx={tokens_sbert['max_tokens']:,}"
    )
    print(
        f"  * Artefacto Zenodo 'Tired': {stats_artefactos['ocurrencias_totales']} ocurrencias "
        f"en {stats_artefactos['docs_afectados']} usuarios."
    )

    # 2. Análisis estadístico Kruskal-Wallis de los 12 patrones en Train (sin tocar Dev/Test)
    print("\n[PASO 2/7] Analizando significancia Kruskal-Wallis de 12 patrones en Train...")
    extractor_est = ExtractorPatronesEstilometricos(limpiar_artefactos=True)
    df_kw, _ = extractor_est.analizar_significancia_clases(txt_tr, y_tr, clases)
    print(df_kw[["etiqueta", "H_kruskal", "p_value", "significativo_05"]].to_string(index=False))
    df_kw.to_csv(OUTPUT_DIR / "tabla_patrones_kruskal_wallis.csv", index=False)

    evaluador = EvaluadorClasificacion(clases_ordenadas=clases, clase_minoritaria=clase_minoritaria)

    # 3. Optimización 5-Fold CV en Train + Puerta de Ablación para Alt. 5
    print("\n[PASO 3/7] Optimizando modelos con 5-Fold CV en Train (proyección exacta de Gram)...")
    opt = OptimizadorCV(clase_minoritaria=clase_minoritaria, n_splits=5, random_state=RANDOM_SEED_OFICIAL)
    cache_folds = {}

    opt_alt2 = opt.optimizar_alternativa2(X384_tr, y_tr)
    opt_alt3 = opt.optimizar_alternativa3(txt_tr, y_tr, cache_tfidf_folds=cache_folds)
    opt_alt4 = opt.optimizar_alternativa4(X384_tr, y_tr, calcular_curvas=True)
    opt_alt5 = opt.optimizar_alternativa5_con_puerta(txt_tr, y_tr, cache_tfidf_folds=cache_folds)

    p2, p3, p4, p5 = (
        opt_alt2["mejores_params"], opt_alt3["mejores_params"],
        opt_alt4["mejores_params"], opt_alt5["mejores_params"],
    )
    print(f"  * Alt. 2 óptimos: {p2}")
    print(f"  * Alt. 3 óptimos: {p3}")
    print(f"  * Alt. 4 óptimos: {p4}")
    print(f"  * Alt. 5 óptimos: {p5}")

    abl = opt_alt5["ablacion_cv"]
    print("\n=== PUERTA DE ABLACIÓN EN 5-FOLD CV (SOLO SE ACTIVA LO QUE MEJORA) ===")
    print(f"  5A) Base TF-IDF Char (Bal.):    OOF F1-Macro={abl['5A_base_char_bal']['oof_f1_macro']:.4f} | OOF F1-Min={abl['5A_base_char_bal']['oof_f1_min']:.4f} | Obj={abl['5A_base_char_bal']['score_obj']:.4f}")
    print(f"  5B) + Patrones Estilométricos:  OOF F1-Macro={abl['5B_mas_estilometria']['oof_f1_macro']:.4f} | OOF F1-Min={abl['5B_mas_estilometria']['oof_f1_min']:.4f} | Obj={abl['5B_mas_estilometria']['score_obj']:.4f} -> {abl['5B_mas_estilometria']['decision_puerta']}")
    print(f"  5C) + Coste Asimétrico (λ OOF): OOF F1-Macro={abl['5C_mas_coste_lambda_oof']['oof_f1_macro']:.4f} | OOF F1-Min={abl['5C_mas_coste_lambda_oof']['oof_f1_min']:.4f} | Obj={abl['5C_mas_coste_lambda_oof']['score_obj']:.4f} (λ={abl['5C_mas_coste_lambda_oof']['lambda_candidato']}) -> {abl['5C_mas_coste_lambda_oof']['decision_puerta']}")

    # 4. Entrenamiento definitivo en Train y evaluación honesta en Dev y Test oficial
    print("\n[PASO 4/7] Evaluando las 5 alternativas en Dev (75) y Test oficial (75)...")
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
        limpiar_artefactos=p5["limpiar_artefactos"],
        usar_estilometria=p5["usar_estilometria"],
        peso_estilometria=p5["peso_estilometria"],
        clase_minoritaria=clase_minoritaria,
        lambda_minoritaria=p5["lambda_minoritaria"],
        random_state=RANDOM_SEED_OFICIAL,
    ).fit(txt_tr, y_tr)

    modelos = [m1, m2, m3, m4, m5]
    resultados_dev = [evaluador.evaluar_modelo(m, X384_dev if m.tipo_entrada == "embeddings" else txt_dev, y_dev) for m in modelos]
    resultados_test = [evaluador.evaluar_modelo(m, X384_te if m.tipo_entrada == "embeddings" else txt_te, y_te) for m in modelos]
    casos_error_test = evaluador.extraer_casos_analisis_error(usr_te, txt_te, X384_te, y_te, m4, m5)

    def _a_df(lista):
        return pd.DataFrame([
            {
                "Alternativa": r["modelo_corto"],
                f"F1_Min ({clase_minoritaria})": r["f1_minoritaria"],
                "Rec_Min": r["recall_minoritaria"],
                "F1_Macro": r["f1_macro"],
                "F1_Weighted": r["f1_weighted"],
                "Accuracy": r["accuracy"],
                "IC95_Acc": r["accuracy_ic95"],
                "Kappa": r["kappa"],
                "AUC_Macro": r["roc_auc_macro"],
                "AUC_Min": r["roc_auc_minoritaria"],
                "AP_Min": r["ap_minoritaria"],
            }
            for r in lista
        ])

    df_dev, df_test = _a_df(resultados_dev), _a_df(resultados_test)
    print("\n=== TABLA 1A: VERIFICACIÓN INDEPENDIENTE EN DEV (seed=42, N=75) ===")
    print(df_dev.to_string(index=False))
    print("\n=== TABLA 1B: RESULTADOS FINALES EN TEST OFICIAL (seed=42, N=75) ===")
    print(df_test.to_string(index=False))
    df_dev.to_csv(OUTPUT_DIR / "tabla_dev_oficial.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "tabla_test_oficial.csv", index=False)

    # 5. Estabilidad en 10 runs estratificados (Nested 5-Fold CV por defecto = 0% leakage)
    usar_nested = not args.sin_nested
    modo_txt = "Nested 5-Fold CV (0% leakage)" if usar_nested else "Hiperparámetros fijos (depuración)"
    print(f"\n[PASO 5/7] Evaluando estabilidad ({len(SEEDS_ESTABILIDAD)} runs — {modo_txt})...")
    df_runs, resumen_estabilidad = evaluador.evaluar_estabilidad_multirun(
        textos_todos=textos_todos,
        X_384_todos=X384_todos,
        y_todos=y_todos,
        semillas=SEEDS_ESTABILIDAD,
        nested_cv=usar_nested,
        params_fijos={"alt2": p2, "alt3": p3, "alt4": p4, "alt5": p5},
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

    # 6. Experimento RQ1 (Ruido Léxico)
    print(f"\n[PASO 6/7] Ejecutando Experimento RQ1 (Ruido Léxico 0%..30%, {args.reps_ruido} rep./nivel)...")
    exp_ruido = ExperimentoRobustezRuido(
        evaluador=evaluador, modelo_tfidf=m3, modelo_sbert=m2, modelo_sbert_bal=m4,
        sbert_encoder=sbert_encoder, modelo_tfidf_bal=m5,
    )
    _, df_ruido_res, caidas_rq1 = exp_ruido.ejecutar(
        textos_test=txt_te, X_384_test_limpio=X384_te, y_test=y_te,
        niveles_ruido=NIVELES_RUIDO_LEXICO, n_repeticiones=args.reps_ruido,
    )
    print("\n=== RESPUESTA RQ1 (Degradación relativa al 30% de ruido léxico) ===")
    for mod, info in caidas_rq1.items():
        print(
            f"  * {mod:<22s} | F1-Macro: {info['f1_macro_0pct']:.4f} -> {info['f1_macro_30pct']:.4f} "
            f"({info['delta_rel_f1_macro_pct']:+.1f}%) | "
            f"F1-Min: {info['f1_min_0pct']:.4f} -> {info['f1_min_30pct']:.4f} "
            f"({info['delta_rel_f1_min_pct']:+.1f}%)"
        )
    df_ruido_res.to_csv(OUTPUT_DIR / "tabla_ruido_lexico_rq1.csv", index=False)

    # Guardar informe JSON completo ANTES de generar figuras
    clase_contraste = "Supportive" if "Supportive" in clases else str(clases[-1])
    top_ng_min = m5.obtener_top_ngramas_unicos(clase_minoritaria, top_n=10)
    top_ng_sup = m5.obtener_top_ngramas_unicos(clase_contraste, top_n=10)

    informe = {
        "autor": "Adrián",
        "version_pipeline": "v5",
        "clase_minoritaria": clase_minoritaria,
        "estadisticas_corpus": stats_corpus,
        "artefactos_zenodo": stats_artefactos,
        "tokens_sbert_reales": tokens_sbert,
        "nested_cv_10runs": usar_nested,
        "n_repeticiones_ruido": int(args.reps_ruido),
        "patrones_kruskal_wallis": df_kw.to_dict(orient="records"),
        "optimizacion_cv": {"alt2": opt_alt2, "alt3": opt_alt3, "alt4": opt_alt4, "alt5": opt_alt5},
        "resultados_dev": resultados_dev,
        "resultados_test_oficial": resultados_test,
        "casos_analisis_error_test": casos_error_test,
        "estabilidad_10runs": resumen_estabilidad,
        "rq1_robustez_ruido": caidas_rq1,
        "top_ngramas_minoritaria": top_ng_min,
        "top_ngramas_supportive": top_ng_sup,
    }
    with open(OUTPUT_DIR / "resultados_completos_adrian.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)

    # 7. Generar las 7 figuras definitivas del póster (300 DPI + SVG)
    print("\n[PASO 7/7] Generando las 7 figuras editoriales del póster (PNG 300 DPI + SVG)...")
    gen = GeneradorPosterAesthetic(dir_datos=OUTPUT_DIR, dir_salida=FIGURES_DIR)
    gen.generar_todas()

    print("\n" + "=" * 84)
    print(f" [OK] FINALIZADO EN {time.time() - t_inicio:.1f}s | Resultados en: {OUTPUT_DIR}")
    print("=" * 84)


if __name__ == "__main__":
    main()