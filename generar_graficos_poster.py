"""
Generador de Figuras Editorial/Aesthetic para Póster (Adrián - v5)
==================================================================
Lee los resultados YA COMPILADOS en 'resultados_adrian/' (sin re-entrenar nada)
y genera las 7 figuras definitivas (PNG 300 DPI + SVG vectorial) sin solapes.

Arreglos v5:
  - Fig 1 usa longitudes reales en tokens WordPiece de all-MiniLM-L6-v2 (sin heurística x1.3).
  - Fig 3 lee dinámicamente el multiplicador de coste lambda* del JSON.
  - Fig 4 lee dinámicamente n_repeticiones_ruido del JSON (sin hardcodear '3 repeticiones').
  - Fig 5 corrige el solapamiento entre las etiquetas del eje X y 'Clase Predicha por el Modelo'.
  - Fig 6 especifica explícitamente en título/subtítulo que corresponde a Alt. 4 (SBERT 384D).

Uso:
    python generar_graficos_poster.py
"""

import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_adrian import (
    RAW_DATA_FILE,
    OUTPUT_DIR,
    FIGURES_DIR,
    RANDOM_SEED_OFICIAL,
    SBERT_MAX_TOKENS,
)
from src.particion_comun import limpiar_texto_reddit
from src.representaciones_adrian import ExtractorPatronesEstilometricos


THEME = {
    "bg_fig": "#FFFFFF",
    "ink_dark": "#0F172A",
    "ink_mid": "#334155",
    "ink_light": "#64748B",
    "grid": "#E2E8F0",
    "spine": "#CBD5E1",
    "alert": "#E11D48",
    "alert_soft": "#FFE4E6",
    "ok": "#059669",
    "accent": "#4338CA",
    "accent_soft": "#EEF2FF",
}

COLOR_MODELO = {
    "Baseline Clust. 384D":  "#94A3B8",
    "LogReg + 384D":         "#60A5FA",
    "LogReg + TF-IDF Char":  "#F59E0B",
    "LogReg + 384D (Bal.)":  "#0D9488",
    "LogReg + Char (Bal.+)": "#4338CA",
}

LABEL_LIMPIO = {
    "Baseline Clust. 384D":  "1. Baseline (Clust. 384D)",
    "LogReg + 384D":         "2. LogReg + 384D",
    "LogReg + TF-IDF Char":  "3. LogReg + TF-IDF Char",
    "LogReg + 384D (Bal.)":  "4. LogReg + 384D (Bal.)",
    "LogReg + Char (Bal.+)": "5. LogReg + Char (Bal.+)",
}

ORDEN_MODELOS = [
    "Baseline Clust. 384D",
    "LogReg + 384D",
    "LogReg + TF-IDF Char",
    "LogReg + 384D (Bal.)",
    "LogReg + Char (Bal.+)",
]


def configurar_estilo_editorial():
    plt.rcParams.update({
        "figure.facecolor": THEME["bg_fig"],
        "axes.facecolor": THEME["bg_fig"],
        "axes.edgecolor": THEME["spine"],
        "axes.linewidth": 0.9,
        "axes.labelcolor": THEME["ink_mid"],
        "axes.titlecolor": THEME["ink_dark"],
        "xtick.color": THEME["ink_mid"],
        "ytick.color": THEME["ink_mid"],
        "text.color": THEME["ink_dark"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.0,
        "axes.titlesize": 11.4,
        "axes.labelsize": 9.6,
        "legend.frameon": False,
        "savefig.facecolor": THEME["bg_fig"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.20,
    })


def estilar_ejes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["spine"])
    ax.spines["bottom"].set_color(THEME["spine"])
    ax.tick_params(length=0, pad=6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=THEME["grid"], linestyle="-", linewidth=0.75, alpha=0.9)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def titulo_editorial(fig, titulo: str, subtitulo: str, y_title: float = 1.01, y_sub: float = 0.94):
    fig.text(0.012, y_title, titulo, ha="left", va="top", fontsize=13.8, fontweight="bold", color=THEME["ink_dark"])
    fig.text(0.012, y_sub, subtitulo, ha="left", va="top", fontsize=9.6, color=THEME["ink_light"])


def evitar_colision_y(posiciones: Dict[str, float], min_gap: float) -> Dict[str, float]:
    items = sorted(posiciones.items(), key=lambda kv: kv[1])
    ajustadas = {}
    prev_y = -1e9
    for k, y in items:
        y_new = max(y, prev_y + min_gap)
        ajustadas[k] = y_new
        prev_y = y_new
    return ajustadas


class GeneradorPosterAesthetic:
    """Genera las 7 figuras definitivas del póster en PNG (300 DPI) + SVG sin solapes."""

    def __init__(self, dir_datos: Path, dir_salida: Path):
        self.dir_datos = Path(dir_datos)
        self.dir_salida = Path(dir_salida)
        self.dir_salida.mkdir(parents=True, exist_ok=True)

        with open(self.dir_datos / "resultados_completos_adrian.json", "r", encoding="utf-8") as f:
            self.informe = json.load(f)

        self.df_runs = pd.read_csv(self.dir_datos / "detalle_10runs.csv")
        self.df_ruido = pd.read_csv(self.dir_datos / "tabla_ruido_lexico_rq1.csv")

        self.clase_min = self.informe["clase_minoritaria"]
        self.res_test = self.informe["resultados_test_oficial"]
        self.clases = list(self.res_test[0]["por_clase"].keys())

        df_csv = pd.read_csv(RAW_DATA_FILE)
        self.textos_todos = df_csv["Post"].fillna("").apply(limpiar_texto_reddit).values
        self.y_todos = df_csv["Label"].astype(str).values

        idx = np.arange(len(self.y_todos))
        idx_tr, _, y_tr, _ = train_test_split(
            idx, self.y_todos, test_size=0.30, random_state=RANDOM_SEED_OFICIAL, stratify=self.y_todos
        )
        self.textos_train = self.textos_todos[idx_tr]
        self.y_train = y_tr

    def _guardar(self, fig, nombre_base: str) -> Path:
        p_png = self.dir_salida / f"{nombre_base}.png"
        p_svg = self.dir_salida / f"{nombre_base}.svg"
        fig.savefig(p_png, dpi=300)
        fig.savefig(p_svg)
        plt.close(fig)
        return p_png

    # -------------------------------------------------------------------------
    # FIGURA 1: Diagnóstico del Corpus (tokens WordPiece REALES de SBERT)
    # -------------------------------------------------------------------------
    def fig1_diagnostico(self) -> Path:
        tok_info = self.informe.get("tokens_sbert_reales")
        if tok_info and "longitudes_tokens" in tok_info:
            tokens = np.asarray(tok_info["longitudes_tokens"], dtype=int)
            etq_eje_x = "Longitud real del historial (tokens WordPiece all-MiniLM-L6-v2, escala log)"
        else:
            tokens = np.array([max(1, int(len(str(t).split()) * 1.3)) for t in self.textos_todos])
            etq_eje_x = "Longitud estimada por usuario (tokens, escala log)"

        pct_trunc = float(np.mean(tokens > SBERT_MAX_TOKENS) * 100.0)
        mediana = float(np.median(tokens))
        max_tok = int(np.max(tokens))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.0, 1.32]})
        titulo_editorial(
            fig,
            f"Dos barreras del corpus: desbalanceo 3,8× y {pct_trunc:.1f}% de textos truncados por SBERT",
            "Corpus clínico Reddit C-SSRS (N = 500 usuarios) · Partición común estratificada 70% Train / 15% Dev / 15% Test (seed = 42)",
            y_title=1.01, y_sub=0.94,
        )

        # Panel A: Clases
        conteos = pd.Series(self.y_todos).value_counts().sort_values(ascending=True)
        colores = [THEME["alert"] if c == self.clase_min else "#94A3B8" if c != "Ideation" else "#334155" for c in conteos.index]
        y_pos = np.arange(len(conteos))
        bars = ax1.barh(y_pos, conteos.values, height=0.60, color=colores, edgecolor="none")

        for b, cls, cnt in zip(bars, conteos.index, conteos.values):
            pct = cnt / len(self.y_todos) * 100.0
            sufijo = "  ★ Minoritaria" if cls == self.clase_min else ""
            col_t = THEME["alert"] if cls == self.clase_min else THEME["ink_dark"]
            ax1.text(
                cnt + 4, b.get_y() + b.get_height() / 2.0,
                f"{cnt} ({pct:.1f}%){sufijo}", va="center", ha="left",
                fontsize=9.1, fontweight="bold" if cls in [self.clase_min, "Ideation"] else "normal", color=col_t,
            )

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(conteos.index, fontsize=9.8, fontweight="bold")
        ax1.set_xlim(0, conteos.max() * 1.50)
        ax1.set_xlabel("Nº de usuarios en el corpus")
        ax1.set_title("A) Distribución de las 5 clases clínicas", loc="left", fontweight="bold", pad=10)
        estilar_ejes(ax1, grid_axis="x")

        # Panel B: Histograma log de tokens WordPiece reales
        xmax = max(13000, int(max_tok * 1.35))
        bins_log = np.logspace(np.log10(max(15, tokens.min())), np.log10(max_tok * 1.05), 30)
        counts_hist, _, _ = ax2.hist(tokens, bins=bins_log, color="#64748B", edgecolor="#FFFFFF", linewidth=0.6, alpha=0.84, zorder=3)
        ax2.set_xscale("log")

        max_bar = float(np.max(counts_hist))
        ymax = max_bar * 1.62
        ax2.set_ylim(0, ymax)
        ax2.set_xlim(15, xmax)

        ax2.axvspan(SBERT_MAX_TOKENS, xmax, color=THEME["alert_soft"], alpha=0.50, zorder=1)
        ax2.axvline(SBERT_MAX_TOKENS, color=THEME["alert"], linestyle="--", linewidth=2.0, zorder=4)
        ax2.axvline(mediana, color=THEME["accent"], linestyle=":", linewidth=2.0, zorder=4)

        ax2.text(
            SBERT_MAX_TOKENS * 0.85, ymax * 0.94,
            "Límite SBERT\n256 tokens", ha="right", va="top",
            fontsize=8.5, fontweight="bold", color=THEME["alert"],
        )
        ax2.text(
            mediana * 1.10, ymax * 0.94,
            f"Mediana: {mediana:.0f} (Máx: {max_tok:,})", ha="left", va="top",
            fontsize=8.3, fontweight="bold", color=THEME["accent"],
        )
        ax2.text(
            mediana * 1.15, max_bar + (ymax - max_bar) * 0.36,
            f"{pct_trunc:.1f}% superan 256 tokens\nSBERT recorta esta zona;\nTF-IDF Char lee el 100%",
            ha="left", va="center", fontsize=8.1, fontweight="bold", color=THEME["alert"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor=THEME["alert"], lw=1.1, alpha=0.98),
            zorder=5,
        )

        ticks = [25, 100, 256, 500, 1000, 2500, 5000, 10000]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{t:,}" for t in ticks], fontsize=8.4)
        ax2.set_xlabel(etq_eje_x)
        ax2.set_ylabel("Nº de usuarios")
        ax2.set_title("B) Longitud WordPiece real vs. ventana de SBERT", loc="left", fontweight="bold", pad=10)
        estilar_ejes(ax2, grid_axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        return self._guardar(fig, "poster_01_diagnostico")

    # -------------------------------------------------------------------------
    # FIGURA 2: Resultados Oficiales + Estabilidad 10 Runs
    # -------------------------------------------------------------------------
    def fig2_resultados_y_estabilidad(self) -> Path:
        res_map = {r["modelo_corto"]: r for r in self.res_test}
        ratio_mac = res_map["LogReg + Char (Bal.+)"]["f1_macro"] / max(res_map["Baseline Clust. 384D"]["f1_macro"], 1e-6)
        es_nested = self.informe.get("nested_cv_10runs", False)
        sub_nested = "Nested 5-Fold CV en 10 particiones" if es_nested else "10 particiones estratificadas"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5), gridspec_kw={"width_ratios": [1.06, 1.12]})
        titulo_editorial(
            fig,
            f"RQ2: Balancear pesos rompe el colapso del baseline y eleva F1-Macro ×{ratio_mac:.1f}",
            f"Izquierda: examen común del equipo (Test seed=42, N=75) · Derecha: estabilidad real ({sub_nested}, Media ± StDev)",
            y_title=1.01, y_sub=0.945,
        )

        y_base = np.arange(len(ORDEN_MODELOS))
        bar_h = 0.34

        for i, mod in enumerate(ORDEN_MODELOS):
            r = res_map[mod]
            col_mod = COLOR_MODELO[mod]

            ax1.barh(i + bar_h / 2 + 0.02, r["f1_macro"], height=bar_h, color=col_mod, alpha=0.95)
            ax1.text(
                r["f1_macro"] + 0.012, i + bar_h / 2 + 0.02,
                f"{r['f1_macro']:.3f}  (κ={r['kappa']:.2f})",
                va="center", ha="left", fontsize=8.6, fontweight="bold", color=THEME["ink_dark"],
            )

            val_min = r["f1_minoritaria"]
            ax1.barh(
                i - bar_h / 2 - 0.02, val_min, height=bar_h,
                color=THEME["alert"] if val_min > 0 else "#E2E8F0", alpha=0.88,
            )
            txt_min = f"{val_min:.3f}" if val_min > 0 else "0.000 (colapso)"
            ax1.text(
                val_min + 0.012, i - bar_h / 2 - 0.02, txt_min,
                va="center", ha="left", fontsize=8.3,
                fontweight="bold" if val_min > 0 else "normal",
                color=THEME["alert"] if val_min > 0 else THEME["ink_light"],
            )

        ax1.set_yticks(y_base)
        ax1.set_yticklabels([LABEL_LIMPIO[m] for m in ORDEN_MODELOS], fontsize=9.3, fontweight="bold")
        ax1.set_xlim(0.0, 0.58)
        ax1.set_xlabel("Puntuación en Test Oficial (↑ Mayor es mejor)")
        ax1.set_title("A) Rendimiento en Test Oficial (seed = 42)", loc="left", fontweight="bold", pad=12)
        estilar_ejes(ax1, grid_axis="x")

        ley_a = [
            patches.Patch(facecolor="#4338CA", label="Barra superior: F1-Macro (y κ)"),
            patches.Patch(facecolor=THEME["alert"], label=f"Barra inferior: F1 Minoritaria ({self.clase_min})"),
        ]
        ax1.legend(
            handles=ley_a, loc="upper center", bbox_to_anchor=(0.48, -0.17),
            ncol=2, fontsize=8.4, frameon=True, facecolor="#F8FAFC", edgecolor=THEME["grid"],
        )

        rng = np.random.RandomState(42)
        max_x_b = 0.55
        for i, mod in enumerate(ORDEN_MODELOS):
            sub = self.df_runs[self.df_runs["modelo_corto"] == mod]
            col_mod = COLOR_MODELO[mod]

            vals_m = sub["f1_macro"].values
            mu_m, sd_m = float(np.mean(vals_m)), float(np.std(vals_m, ddof=1))
            y_m = i + 0.17
            ax2.scatter(vals_m, y_m + rng.uniform(-0.04, 0.04, len(vals_m)), color=col_mod, alpha=0.28, s=22, zorder=3)
            ax2.errorbar(mu_m, y_m, xerr=sd_m, fmt="o", color=col_mod, markersize=7.5, elinewidth=2.2, capsize=3.5, zorder=5)
            x_txt_m = max(mu_m + sd_m, float(np.max(vals_m))) + 0.016
            max_x_b = max(max_x_b, x_txt_m + 0.14)
            ax2.text(
                x_txt_m, y_m, f"{mu_m:.3f} ± {sd_m:.3f}",
                va="center", ha="left", fontsize=8.3, fontweight="bold", color=THEME["ink_dark"], zorder=6,
            )

            vals_a = sub["f1_minoritaria"].values
            mu_a, sd_a = float(np.mean(vals_a)), float(np.std(vals_a, ddof=1))
            y_a = i - 0.17
            ax2.scatter(vals_a, y_a + rng.uniform(-0.04, 0.04, len(vals_a)), color=THEME["alert"], alpha=0.24, s=18, zorder=3)
            ax2.errorbar(
                mu_a, y_a, xerr=sd_a, fmt="D",
                color=THEME["alert"] if mu_a > 0 else "#CBD5E1",
                markersize=6.2, elinewidth=1.8, capsize=3.0, zorder=5,
            )
            txt_a = f"{mu_a:.3f} ± {sd_a:.3f}" if mu_a > 0 else "0.000"
            x_txt_a = (max(mu_a + sd_a, float(np.max(vals_a))) + 0.016) if mu_a > 0 else 0.018
            max_x_b = max(max_x_b, x_txt_a + 0.14)
            ax2.text(
                x_txt_a, y_a, txt_a,
                va="center", ha="left", fontsize=8.1,
                fontweight="bold" if mu_a > 0 else "normal",
                color=THEME["alert"] if mu_a > 0 else THEME["ink_light"], zorder=6,
            )

            if i < len(ORDEN_MODELOS) - 1:
                ax2.axhline(i + 0.5, color=THEME["grid"], linewidth=0.7, linestyle="-")

        ax2.set_yticks(y_base)
        ax2.set_yticklabels([""] * len(ORDEN_MODELOS))
        ax2.set_xlim(-0.02, min(0.75, max_x_b))
        ax2.set_xlabel("Media ± StDev en 10 particiones (↑ Mayor y ↓ bigote más corto = mejor)")
        ax2.set_title("B) Estabilidad Estadística (10 Runs Estratificados)", loc="left", fontweight="bold", pad=12)
        estilar_ejes(ax2, grid_axis="x")

        ley_b = [
            Line2D([0], [0], marker="o", color=THEME["accent"], lw=2, markersize=6.8, label="F1-Macro (Media ± StDev)"),
            Line2D([0], [0], marker="D", color=THEME["alert"], lw=2, markersize=6.0, label=f"F1-{self.clase_min} (Media ± StDev)"),
        ]
        ax2.legend(
            handles=ley_b, loc="upper center", bbox_to_anchor=(0.48, -0.17),
            ncol=2, fontsize=8.4, frameon=True, facecolor="#F8FAFC", edgecolor=THEME["grid"],
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.89])
        return self._guardar(fig, "poster_02_resultados_estabilidad")

    # -------------------------------------------------------------------------
    # FIGURA 3: ROC y Precision-Recall (dinámica con lambda* del JSON)
    # -------------------------------------------------------------------------
    def fig3_roc_pr(self) -> Path:
        res_map = {r["modelo_corto"]: r for r in self.res_test}
        r5 = res_map["LogReg + Char (Bal.+)"]
        prevalencia = self.res_test[0]["por_clase"][self.clase_min]["soporte"] / 75.0
        ratio_ap = r5["ap_minoritaria"] / max(prevalencia, 1e-6)
        lam5 = r5["hiperparametros"].get("lambda_minoritaria", 1.0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
        titulo_editorial(
            fig,
            f"TF-IDF por caracteres ordena mejor la clase '{self.clase_min}' (AUC {r5['roc_auc_minoritaria']:.2f} · AP ×{ratio_ap:.1f} azar)",
            f"Evaluación probabilística independiente del umbral (Tema 3) en Test oficial (N = 75, prevalencia real = {prevalencia*100:.1f}%)",
            y_title=1.01, y_sub=0.94,
        )

        for mod in ORDEN_MODELOS:
            r = res_map[mod]
            col = COLOR_MODELO[mod]
            es_hero = mod in ["LogReg + Char (Bal.+)", "LogReg + 384D (Bal.)"]
            lw = 2.6 if mod == "LogReg + Char (Bal.+)" else (2.1 if es_hero else 1.7)
            ls = "--" if mod == "Baseline Clust. 384D" else "-"
            alpha = 1.0 if es_hero else 0.78

            fpr, tpr = r["curva_roc_minoritaria"]["fpr"], r["curva_roc_minoritaria"]["tpr"]
            ax1.plot(fpr, tpr, color=col, lw=lw, ls=ls, alpha=alpha,
                     label=f"{LABEL_LIMPIO[mod]}  (AUC {r['roc_auc_minoritaria']:.3f})")
            if mod == "LogReg + Char (Bal.+)":
                ax1.fill_between(fpr, tpr, color=col, alpha=0.07)

            rec, prec = r["curva_pr_minoritaria"]["recall"], r["curva_pr_minoritaria"]["precision"]
            ax2.plot(rec, prec, color=col, lw=lw, ls=ls, alpha=alpha,
                     label=f"{LABEL_LIMPIO[mod]}  (AP {r['ap_minoritaria']:.3f})")
            if mod == "LogReg + Char (Bal.+)":
                ax2.fill_between(rec, prec, color=col, alpha=0.07)

        ax1.plot([0, 1], [0, 1], color="#CBD5E1", ls=":", lw=1.4, label="Azar (AUC 0.500)")
        ax1.annotate(
            "Óptimo (0, 1)\n↖ Más arriba-izq",
            xy=(0.01, 0.99), xytext=(0.08, 0.80),
            arrowprops=dict(arrowstyle="->", color=THEME["ok"], lw=1.5),
            fontsize=8.2, fontweight="bold", color=THEME["ok"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=THEME["ok"], lw=0.8, alpha=0.92),
        )
        ax1.set_title(f"A) Curva ROC One-vs-Rest ('{self.clase_min}')", loc="left", fontweight="bold", pad=10)
        ax1.set_xlabel("Tasa de Falsos Positivos (FPR) [↓ Mejor]")
        ax1.set_ylabel("Tasa de Verdaderos Positivos (Recall) [↑ Mejor]")
        ax1.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor=THEME["grid"])
        estilar_ejes(ax1, grid_axis="both")

        ax2.axhline(prevalencia, color="#CBD5E1", ls=":", lw=1.4, label=f"Prevalencia azar ({prevalencia:.3f})")
        txt_coste = f"Decisión por coste\nλ* = {lam5:.2f} (Tema 3)" if lam5 > 1.0 else "Mejor ranking PR\n(AP más alto en Test)"
        ax2.text(
            0.74, 0.44, txt_coste, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=THEME["accent"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor=THEME["accent_soft"], edgecolor=THEME["accent"], lw=0.9),
            zorder=6,
        )
        ax2.set_title(f"B) Curva Precision-Recall ('{self.clase_min}')", loc="left", fontweight="bold", pad=10)
        ax2.set_xlabel("Recall / Cobertura de casos reales [↑ Mejor]")
        ax2.set_ylabel("Precision / Acierto al alertar [↑ Mejor]")
        ax2.set_xlim(0, 1.02)
        ax2.set_ylim(0, 1.06)
        ax2.legend(loc="upper right", fontsize=7.7, labelspacing=0.28, frameon=True, facecolor="#FFFFFF", edgecolor=THEME["grid"])
        estilar_ejes(ax2, grid_axis="both")

        plt.tight_layout(rect=[0, 0, 1, 0.89])
        return self._guardar(fig, "poster_03_roc_pr")

    # -------------------------------------------------------------------------
    # FIGURA 4: RQ1 — Robustez ante Ruido Léxico (lee n_repeticiones dinámicamente)
    # -------------------------------------------------------------------------
    def fig4_rq1_ruido(self) -> Path:
        caidas = self.informe["rq1_robustez_ruido"]
        n_reps = self.informe.get("n_repeticiones_ruido", 5)
        peor_sbert = min(caidas["LogReg + 384D"]["delta_rel_f1_macro_pct"], caidas["LogReg + 384D (Bal.)"]["delta_rel_f1_macro_pct"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
        titulo_editorial(
            fig,
            f"RQ1: TF-IDF por caracteres resiste un 30% de erratas; SBERT cae hasta un {abs(peor_sbert):.0f}%",
            f"Entrenamiento limpio y evaluación con erratas crecientes en Test ({n_reps} repeticiones/nivel) · Línea más plana y alta = más robusto",
            y_title=1.01, y_sub=0.94,
        )

        modelos_ruido = [
            "LogReg + Char (Bal.+)",
            "LogReg + TF-IDF Char",
            "LogReg + 384D",
            "LogReg + 384D (Bal.)",
        ]
        marcadores = {
            "LogReg + Char (Bal.+)": "D",
            "LogReg + TF-IDF Char":  "s",
            "LogReg + 384D":         "o",
            "LogReg + 384D (Bal.)":  "^",
        }
        niveles = sorted(self.df_ruido["nivel_ruido_pct"].unique())

        end_y_abs, end_y_rel = {}, {}
        for mod in modelos_ruido:
            g = self.df_ruido[self.df_ruido["modelo"] == mod].sort_values("nivel_ruido_pct")
            x = g["nivel_ruido_pct"].values
            mu = g["f1_macro_mean"].values
            sd = g["f1_macro_std"].values
            col = COLOR_MODELO[mod]
            es_char = "Char" in mod

            ax1.plot(x, mu, color=col, marker=marcadores[mod], lw=2.6 if es_char else 2.0,
                     ls="-" if es_char else "--", markersize=6.2, zorder=4)
            ax1.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.12, zorder=2)
            end_y_abs[mod] = float(mu[-1])

            ret = (mu / max(mu[0], 1e-6)) * 100.0
            ax2.plot(x, ret, color=col, marker=marcadores[mod], lw=2.6 if es_char else 2.0,
                     ls="-" if es_char else "--", markersize=6.2, zorder=4)
            end_y_rel[mod] = float(ret[-1])

        pos_abs = evitar_colision_y(end_y_abs, min_gap=0.034)
        for mod in modelos_ruido:
            nombre_corto = mod.replace("LogReg + ", "")
            ax1.text(
                31.0, pos_abs[mod], f"{nombre_corto} ({end_y_abs[mod]:.2f})",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color=COLOR_MODELO[mod],
            )

        pos_rel = evitar_colision_y(end_y_rel, min_gap=6.5)
        for mod in modelos_ruido:
            delta = caidas[mod]["delta_rel_f1_macro_pct"]
            nombre_corto = mod.replace("LogReg + ", "")
            ax2.text(
                31.0, pos_rel[mod], f"{nombre_corto} ({delta:+.1f}%)",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color=COLOR_MODELO[mod],
            )

        ax1.set_title("A) Rendimiento absoluto (F1-Macro ± StDev)", loc="left", fontweight="bold", pad=10)
        ax1.set_xlabel("% de palabras con erratas en Test")
        ax1.set_ylabel("F1-Macro [↑ Mejor]")
        ax1.set_xticks(niveles)
        ax1.set_xlim(-1.2, 47.0)
        ax1.set_ylim(0.10, max(0.475, max(end_y_abs.values()) + 0.12))
        estilar_ejes(ax1, grid_axis="y")

        ax2.axhline(100.0, color="#94A3B8", ls=":", lw=1.4, zorder=1)
        ax2.text(
            24.5, 99.6, "100% (sin caída)", ha="center", va="center",
            fontsize=7.9, color=THEME["ink_light"], style="italic",
            bbox=dict(boxstyle="square,pad=0.18", facecolor="#FFFFFF", edgecolor="none", alpha=0.95),
            zorder=3,
        )
        ax2.set_title("B) Retención relativa del rendimiento original", loc="left", fontweight="bold", pad=10)
        ax2.set_xlabel("% de palabras con erratas en Test")
        ax2.set_ylabel("% de F1-Macro conservado [↑ Mejor]")
        ax2.set_xticks(niveles)
        ax2.set_xlim(-1.2, 48.0)
        ax2.set_ylim(45, 115)
        estilar_ejes(ax2, grid_axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.89])
        return self._guardar(fig, "poster_04_rq1_ruido")

    # -------------------------------------------------------------------------
    # FIGURA 5: Matrices de Confusión (ARREGLADO EL SOLAPE CON EL EJE X)
    # -------------------------------------------------------------------------
    def fig5_matrices_historia(self) -> Path:
        res_map = {r["modelo_corto"]: r for r in self.res_test}
        rec4 = res_map["LogReg + 384D (Bal.)"]["recall_minoritaria"] * 100.0
        seleccion = [
            ("Baseline Clust. 384D",  "Paso 1 · Baseline\nColapso en 'Ideation'"),
            ("LogReg + 384D (Bal.)",  f"Paso 2 · Alt. 4 (384D Bal.)\nDesbloquea '{self.clase_min}' ({rec4:.0f}%)"),
            ("LogReg + Char (Bal.+)", "Paso 3 · Alt. 5 (Char Bal.+)\nMejor equilibrio global"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.6))
        titulo_editorial(
            fig,
            f"Anatomía del error (RQ2): del colapso en 'Ideation' a la detección real de '{self.clase_min}'",
            "Matrices de confusión en Test oficial (N = 75) · Color = Recall por fila (ideal: diagonal oscura) · Recuadro rojo = Clase Minoritaria",
            y_title=0.985, y_sub=0.928,
        )

        # bottom=0.26 deja espacio generoso para las etiquetas rotadas SIN tocar el título X ni la colorbar
        fig.subplots_adjust(left=0.08, right=0.985, top=0.75, bottom=0.26, wspace=0.14)

        idx_min = self.clases.index(self.clase_min)
        im_ref = None
        for ax, (mod, subtit) in zip(axes, seleccion):
            r = res_map[mod]
            M = np.asarray(r["matriz_confusion"], dtype=float)
            M_norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)

            im_ref = ax.imshow(M_norm, cmap="PuBu", vmin=0.0, vmax=1.0)
            for row in range(M.shape[0]):
                for col in range(M.shape[1]):
                    cnt = int(M[row, col])
                    pct = M_norm[row, col] * 100.0
                    c_txt = "#FFFFFF" if M_norm[row, col] > 0.55 else THEME["ink_dark"]
                    ax.text(col, row, f"{cnt}\n{pct:.0f}%", ha="center", va="center",
                            color=c_txt, fontsize=8.5, fontweight="bold" if row == col or cnt > 0 else "normal")

            rect = patches.Rectangle(
                (-0.48, idx_min - 0.48), len(self.clases) - 0.04, 0.96,
                lw=2.2, edgecolor=THEME["alert"], facecolor="none", zorder=5,
            )
            ax.add_patch(rect)
            ax.set_title(
                f"{subtit}\nF1-Min={r['f1_minoritaria']:.2f} · F1-Mac={r['f1_macro']:.2f} · κ={r['kappa']:.2f}",
                fontsize=9.6, fontweight="bold", pad=12,
                color=COLOR_MODELO[mod] if mod != "Baseline Clust. 384D" else THEME["ink_mid"],
            )
            ax.set_xticks(range(len(self.clases)))
            ax.set_yticks(range(len(self.clases)))
            ax.set_xticklabels(self.clases, rotation=28, ha="right", fontsize=8.8, fontweight="bold")
            ax.set_yticklabels(self.clases if ax is axes[0] else [""] * len(self.clases), fontsize=9.0, fontweight="bold")
            if ax is axes[0]:
                ax.set_ylabel("Clase Real (C-SSRS)", fontsize=9.4, fontweight="bold", labelpad=8)
            ax.tick_params(length=0, pad=5)
            for sp in ax.spines.values():
                sp.set_color(THEME["grid"])

        # Título X y colorbar colocados muy por debajo de las etiquetas rotadas (cero colisiones)
        fig.text(0.53, 0.11, "Clase Predicha por el Modelo", ha="center", va="center",
                 fontsize=9.6, fontweight="bold", color=THEME["ink_mid"])

        cax = fig.add_axes([0.39, 0.045, 0.28, 0.022])
        cbar = fig.colorbar(im_ref, cax=cax, orientation="horizontal")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0% Recall", "50%", "100% Recall"])
        cbar.ax.tick_params(labelsize=8.4, length=0)
        cbar.outline.set_edgecolor(THEME["grid"])

        return self._guardar(fig, "poster_05_matrices_rq2")

    # -------------------------------------------------------------------------
    # FIGURA 6: Tuning 5-Fold CV de Alt. 4 (explicitado claramente en el título)
    # -------------------------------------------------------------------------
    def fig6_tuning_cv(self) -> Path:
        opt4 = self.informe["optimizacion_cv"]["alt4"]
        df_a = pd.DataFrame(opt4["curva_alpha_cv"])
        df_bv = pd.DataFrame(opt4["curva_sesgo_varianza"])
        a_opt = opt4["mejores_params"]["alpha_pesos"]
        c_opt = opt4["mejores_params"]["C"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
        titulo_editorial(
            fig,
            "Ajuste en 5-Fold CV sobre Train — Alt. 4 (LogReg + SBERT 384D): balanceo α y regularización C",
            "Evidencia del Tema 4 (Dilema Sesgo-Varianza) en D = 384 dimensiones sin tocar Dev ni Test",
            y_title=1.01, y_sub=0.94,
        )

        ax1.plot(df_a["alpha"], df_a["cv_f1_macro"], color=THEME["accent"], marker="s", lw=2.3, label="CV F1-Macro (5 clases)")
        ax1.plot(df_a["alpha"], df_a["cv_f1_min"], color=THEME["ok"], marker="o", lw=2.5, label=f"CV F1-{self.clase_min}")
        ax1.plot(df_a["alpha"], df_a["cv_rec_min"], color="#60A5FA", marker="^", lw=1.8, ls="--", label=f"CV Recall-{self.clase_min}")
        ax1.axvline(a_opt, color=THEME["alert"], ls="--", lw=1.7)
        ax1.text(
            a_opt - 0.04, max(df_a["cv_rec_min"].max(), df_a["cv_f1_macro"].max()) + 0.04,
            f"α* = {a_opt}\n(óptimo CV)", ha="right", va="top",
            fontsize=8.5, fontweight="bold", color=THEME["alert"],
        )
        ax1.set_title(r"A) Intensidad del balanceo $w_c \propto (N / N_c)^\alpha$ (Alt. 4)", loc="left", fontweight="bold", pad=10)
        ax1.set_xlabel("Exponente α (0 = sin balanceo · 1 = balanced)")
        ax1.set_ylabel("F1 / Recall en 5-Fold CV [↑ Mejor]")
        ax1.set_ylim(-0.02, max(0.42, df_a["cv_rec_min"].max() + 0.12))
        ax1.legend(loc="upper left", fontsize=8.2)
        estilar_ejes(ax1, grid_axis="y")

        ax2.plot(df_bv["C"], df_bv["train_f1_macro"], color=THEME["alert"], marker="o", lw=2.2, label="Train F1-Macro (memorización)")
        ax2.plot(df_bv["C"], df_bv["cv_f1_macro"], color=THEME["accent"], marker="s", lw=2.5, label="5-Fold CV F1-Macro (generalización)")
        ax2.plot(df_bv["C"], df_bv["cv_f1_min"], color=THEME["ok"], marker="^", lw=1.9, ls="--", label=f"5-Fold CV F1-{self.clase_min}")
        ax2.fill_between(df_bv["C"], df_bv["cv_f1_macro"], df_bv["train_f1_macro"], color=THEME["alert_soft"], alpha=0.35)
        ax2.axvline(c_opt, color=THEME["ink_dark"], ls="--", lw=1.5)

        cv_pico = float(df_bv.loc[df_bv["C"] == c_opt, "cv_f1_macro"].values[0]) if (df_bv["C"] == c_opt).any() else float(df_bv["cv_f1_macro"].max())
        ax2.annotate(
            f"C* = {c_opt} (óptimo CV)",
            xy=(c_opt, cv_pico + 0.01), xytext=(min(c_opt * 1.45, 5.0), 0.44),
            arrowprops=dict(arrowstyle="->", color=THEME["ink_dark"], lw=1.4),
            fontsize=8.4, fontweight="bold", color=THEME["ink_dark"],
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor=THEME["ink_dark"], lw=0.9),
            zorder=6,
        )
        ax2.text(9.8, 0.62, "Brecha de\noverfitting", ha="center", va="center", fontsize=8.5, fontweight="bold", color=THEME["alert"])

        ax2.set_xscale("log")
        ax2.set_title("B) Dilema Sesgo-Varianza según regularización C (Alt. 4)", loc="left", fontweight="bold", pad=10)
        ax2.set_xlabel("C en escala log (← más regularización | más overfitting →)")
        ax2.set_ylabel("F1-Score [↑ Mejor en CV]")
        ax2.set_ylim(0.0, 1.06)
        ax2.legend(loc="upper left", fontsize=8.0)
        estilar_ejes(ax2, grid_axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.89])
        return self._guardar(fig, "poster_06_tuning_cv")

    # -------------------------------------------------------------------------
    # FIGURA 7: Interpretabilidad Clínica
    # -------------------------------------------------------------------------
    def fig7_interpretabilidad_clinica(self) -> Path:
        top_min = self.informe["top_ngramas_minoritaria"][:8]
        top_sup = self.informe["top_ngramas_supportive"][:8]

        extractor = ExtractorPatronesEstilometricos(limpiar_artefactos=True)
        df_kw, df_z = extractor.analizar_significancia_clases(self.textos_train, self.y_train, np.array(self.clases))
        top5_feats = df_kw["feature"].head(5).tolist()
        mat_z = df_z[top5_feats].T

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.4, 5.0), gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]})
        titulo_editorial(
            fig,
            "Transparencia clínica (Tema 4): pistas léxicas aprendidas y firma estilométrica por clase",
            "Pesos positivos θ de LogReg sobre TF-IDF caracteres (A, B) y perfiles Z-score de los 5 rasgos Kruskal-Wallis más significativos en Train (C)",
            y_title=1.005, y_sub=0.94,
        )

        for ax, datos, tit, col in [
            (ax1, top_min, f"A) Top n-gramas '{self.clase_min}'", THEME["alert"]),
            (ax2, top_sup, "B) Top n-gramas 'Supportive'", THEME["ok"]),
        ]:
            etqs = [x[0] for x in reversed(datos)]
            vals = [x[1] for x in reversed(datos)]
            y_p = np.arange(len(etqs))
            ax.hlines(y_p, 0, vals, color=col, lw=2.4, alpha=0.85)
            ax.scatter(vals, y_p, color=col, s=55, zorder=4)
            for yp, v in zip(y_p, vals):
                ax.text(v + max(vals) * 0.04, yp, f"+{v:.2f}", va="center", fontsize=8.3, fontweight="bold", color=col)
            ax.set_yticks(y_p)
            ax.set_yticklabels(etqs, fontsize=8.6, fontfamily="monospace", fontweight="bold")
            ax.set_xlim(0, max(vals) * 1.35)
            ax.set_xlabel("Peso θ en LogReg [↑]")
            ax.set_title(tit, loc="left", fontweight="bold", pad=10, color=col)
            estilar_ejes(ax, grid_axis="x")

        im = ax3.imshow(mat_z.values, cmap="RdBu_r", vmin=-0.95, vmax=0.95, aspect="auto")
        for r in range(mat_z.shape[0]):
            for c in range(mat_z.shape[1]):
                v = mat_z.values[r, c]
                c_t = "#FFFFFF" if abs(v) > 0.52 else THEME["ink_dark"]
                ax3.text(c, r, f"{v:+.2f}", ha="center", va="center", color=c_t, fontsize=8.4, fontweight="bold")

        etq_cortas = {
            "balance_yo_vs_tu": "Yo − Tú (p<.001)",
            "pct_1a_persona": "% 1ª pers. (p<.001)",
            "pct_2a_persona": "% 2ª pers. (p<.001)",
            "riqueza_lexica_guiraud": "Guiraud (p=.005)",
            "long_media_palabra": "Long. pal. (p=.016)",
        }
        ax3.set_yticks(range(len(top5_feats)))
        ax3.set_yticklabels([etq_cortas.get(f, f) for f in top5_feats], fontsize=8.5, fontweight="bold")
        ax3.set_xticks(range(len(mat_z.columns)))
        ax3.set_xticklabels(mat_z.columns, rotation=24, ha="right", fontsize=8.5, fontweight="bold")
        ax3.set_title("C) Perfil lingüístico (Z-score)", loc="left", fontweight="bold", pad=10)
        ax3.tick_params(length=0)
        for sp in ax3.spines.values():
            sp.set_color(THEME["grid"])

        plt.tight_layout(rect=[0, 0.08, 1, 0.89], w_pad=2.4)

        pos_c = ax3.get_position()
        cax = fig.add_axes([pos_c.x0 + 0.02, 0.025, pos_c.width - 0.04, 0.022])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks([-0.9, 0.0, 0.9])
        cbar.set_ticklabels(["Bajo (-0.9)", "Media (0)", "Alto (+0.9)"])
        cbar.ax.tick_params(labelsize=8.0, length=0)
        cbar.outline.set_edgecolor(THEME["grid"])

        return self._guardar(fig, "poster_07_interpretabilidad")

    def generar_todas(self):
        configurar_estilo_editorial()
        rutas = [
            self.fig1_diagnostico(),
            self.fig2_resultados_y_estabilidad(),
            self.fig3_roc_pr(),
            self.fig4_rq1_ruido(),
            self.fig5_matrices_historia(),
            self.fig6_tuning_cv(),
            self.fig7_interpretabilidad_clinica(),
        ]
        print(f"\n[OK] 7 figuras editoriales pulidas (PNG 300 DPI + SVG) en:\n     {self.dir_salida}\n")
        for i, r in enumerate(rutas, start=1):
            print(f"  {i}) {r.name}  (+ .svg)")


if __name__ == "__main__":
    gen = GeneradorPosterAesthetic(
        dir_datos=OUTPUT_DIR,
        dir_salida=FIGURES_DIR,
    )
    gen.generar_todas()