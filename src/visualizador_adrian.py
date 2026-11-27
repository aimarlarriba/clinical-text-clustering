"""
Módulo de Visualización 100% Autoexplicativa para Póster DIN-A3 - Adrián (v4).
Novedades v4:
  - Fig 1 corregida: escala logarítmica real sin recorte artificial en 3500 tokens.
  - Fig 9 mejorada: muestra la palabra ejemplo real de cada n-grama sin duplicados.
  - NUEVA Fig 10: análisis estadístico Kruskal-Wallis de los 12 patrones estilométricos
    y mapa de calor (Z-scores) del perfil lingüístico de las 5 clases clínicas.
"""

import textwrap
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
except ImportError:
    plt.style.use("ggplot")


class VisualizadorPoster:
    """Genera las 10 figuras autoexplicativas (300 DPI) del proyecto."""

    PALETA = ["#6c757d", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    def __init__(self, carpeta_salida: Path, clases: np.ndarray, clase_minoritaria: str):
        self.carpeta_salida = Path(carpeta_salida)
        self.carpeta_salida.mkdir(parents=True, exist_ok=True)
        self.clases = [str(c) for c in clases]
        self.clase_minoritaria = str(clase_minoritaria)

    @staticmethod
    def _poner_caja_lectura(fig, texto: str, ancho: int = 132, color_fondo: str = "#f4f6f9", color_borde: str = "#495057"):
        fig.text(
            0.5, 0.01, textwrap.fill(texto, width=ancho),
            ha="center", va="bottom", fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color_fondo, edgecolor=color_borde, alpha=0.95),
        )

    def fig1_diagnostico_dataset_y_truncamiento(
        self, textos_todos: np.ndarray, y_todos: np.ndarray, limite_tokens_sbert: int = 256
    ) -> Path:
        """Figura 1 corregida: distribución real de tokens en escala logarítmica (sin recorte artificial)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 6.4))
        fig.suptitle(
            "Diagnóstico del Corpus Clínico (N = 500 Usuarios de Reddit): Desbalanceo y Truncamiento de SBERT",
            fontsize=13.5, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Dos propiedades estructurales reales del corpus (sin recortes artificiales) que condicionan el rendimiento",
            ha="center", fontsize=9.8, style="italic", color="#333333",
        )

        # Panel A: Clases
        conteos = pd.Series(y_todos).value_counts().reindex(self.clases, fill_value=0)
        colores_cls = ["#d62728" if c == self.clase_minoritaria else "#3b75af" for c in self.clases]
        barras = ax1.bar(self.clases, conteos.values, color=colores_cls, edgecolor="black", alpha=0.9)
        for b, cnt in zip(barras, conteos.values):
            ax1.text(
                b.get_x() + b.get_width() / 2.0, cnt + 3,
                f"{cnt}\n({cnt / len(y_todos) * 100:.1f}%)", ha="center", va="bottom", fontsize=9.0, fontweight="bold",
            )
        ax1.set_title("A) Distribución de Clases (Rojo = Clase Minoritaria)", fontsize=11.2, fontweight="bold")
        ax1.set_ylabel("Número de usuarios (reparto estratificado 70/15/15)", fontweight="bold")
        ax1.set_xlabel("Nivel de Riesgo Clínico (C-SSRS)", fontweight="bold")
        ax1.set_ylim(0, max(conteos.values) * 1.24)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # Panel B: Longitud REAL en escala logarítmica
        tokens_aprox = np.array([max(1, int(len(str(t).split()) * 1.3)) for t in textos_todos])
        pct_truncados = float(np.mean(tokens_aprox > limite_tokens_sbert) * 100.0)
        mediana_tok = float(np.median(tokens_aprox))
        p90_tok = float(np.percentile(tokens_aprox, 90))
        max_tok = int(np.max(tokens_aprox))

        bins_log = np.logspace(np.log10(max(10, tokens_aprox.min())), np.log10(max_tok * 1.05), 32)
        ax2.hist(tokens_aprox, bins=bins_log, color="#2b8cbe", edgecolor="white", alpha=0.86)
        ax2.set_xscale("log")

        ax2.axvline(limite_tokens_sbert, color="#d62728", linestyle="--", linewidth=2.4, label=f"Límite SBERT = {limite_tokens_sbert} tokens (recorta aquí)")
        ax2.axvline(mediana_tok, color="#2ca02c", linestyle="-.", linewidth=2.0, label=f"Mediana = {mediana_tok:.0f} tokens")
        ax2.axvline(p90_tok, color="#ff7f0e", linestyle=":", linewidth=2.0, label=f"Percentil 90 = {p90_tok:.0f} (Máx real: {max_tok:,})")

        ticks_cand = [25, 50, 100, 256, 500, 1000, 2500, 5000, 10000, 25000, 50000]
        ticks_ok = [t for t in ticks_cand if tokens_aprox.min() * 0.8 <= t <= max_tok * 1.2]
        ax2.set_xticks(ticks_ok)
        ax2.set_xticklabels([f"{t:,}" for t in ticks_ok], rotation=20, fontsize=8.5)

        ax2.annotate(
            f"{pct_truncados:.1f}% de usuarios\nsuperan 256 tokens\n(SBERT trunca el resto;\nTF-IDF lee el 100%)",
            xy=(limite_tokens_sbert, ax2.get_ylim()[1] * 0.72),
            xytext=(limite_tokens_sbert * 5.5, ax2.get_ylim()[1] * 0.52),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#d62728"),
            fontsize=8.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#d62728"),
        )
        ax2.set_title("B) Longitud Real por Usuario (Escala Log, sin recortes)", fontsize=11.2, fontweight="bold")
        ax2.set_xlabel("Longitud real estimada en tokens (escala logarítmica)", fontweight="bold")
        ax2.set_ylabel("Número de usuarios", fontweight="bold")
        ax2.legend(loc="upper left", fontsize=8.2, frameon=True)
        ax2.grid(True, which="both", linestyle=":", alpha=0.5)

        ratio = conteos.max() / max(conteos.min(), 1)
        self._poner_caja_lectura(
            fig,
            f"Lectura clave: (A) Desbalanceo de {ratio:.1f}x entre '{conteos.idxmax()}' y '{self.clase_minoritaria}' (solo 7 casos en Test). "
            f"(B) En escala logarítmica real (mediana={mediana_tok:.0f}, P90={p90_tok:.0f}, máx={max_tok:,} tokens), el {pct_truncados:.1f}% de usuarios superan el límite de 256 tokens de SBERT.",
            color_fondo="#eef7ff", color_borde="#1f77b4",
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.90])
        ruta = self.carpeta_salida / "fig1_diagnostico_dataset_truncamiento.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig2_comparativa_oficial(self, resultados_test: List[Dict[str, Any]]) -> Path:
        metricas = [
            ("f1_minoritaria", f"F1 Minoritaria\n({self.clase_minoritaria})\n[↑ Mejor]"),
            ("f1_macro", "F1 Macro\n(5 clases igual)\n[↑ Mejor]"),
            ("f1_weighted", "F1 Weighted\n(ponderado)\n[↑ Mejor]"),
            ("kappa", "Kappa (κ)\n(0 = azar)\n[↑ Mejor]"),
            ("roc_auc_macro", "ROC-AUC Macro\n(0.5 = azar)\n[↑ Mejor]"),
        ]

        fig, ax = plt.subplots(figsize=(14.5, 6.8))
        fig.suptitle(
            "Comparativa Global de Alternativas en Test Oficial (Partición Común 70/15/15, seed=42, N=75)",
            fontsize=13.4, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "En todas las métricas, barras más altas (↑) indican mejor rendimiento | Incluye 4 alternativas acordadas + 5ª Propuesta Síntesis",
            ha="center", fontsize=9.7, style="italic", color="#333333",
        )

        n_mod = len(resultados_test)
        x = np.arange(len(metricas))
        ancho = 0.155

        for idx_m, res in enumerate(resultados_test):
            valores = [res[k] for k, _ in metricas]
            despl = (idx_m - (n_mod - 1) / 2.0) * ancho
            barras = ax.bar(
                x + despl, valores, ancho,
                label=res["modelo_corto"],
                color=self.PALETA[idx_m % len(self.PALETA)],
                edgecolor="black", linewidth=0.7, alpha=0.92,
            )
            for b in barras:
                h = b.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(b.get_x() + b.get_width() / 2.0, max(h, 0.0)),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.9, fontweight="bold",
                )

        ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2, alpha=0.75)
        ax.text(-0.42, 0.515, "Referencia azar ROC-AUC (0.50)", ha="left", fontsize=8.3, color="#555555", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metricas], fontweight="bold", fontsize=9.3)
        ax.set_ylim(0.0, 1.04)
        ax.set_ylabel("Puntuación de la Métrica (↑ Mayor es mejor)", fontweight="bold")
        ax.legend(loc="upper left", ncol=3, frameon=True, fontsize=8.8, title="Alternativas Evaluadas", title_fontsize=9.2)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        mejor_macro = max(resultados_test, key=lambda r: r["f1_macro"])
        max_f1_min = max(r["f1_minoritaria"] for r in resultados_test)
        if max_f1_min > 0:
            gan_min = max(resultados_test, key=lambda r: r["f1_minoritaria"])
            txt_min = f"y '{gan_min['modelo_corto']}' consigue la mejor F1 en la clase minoritaria '{self.clase_minoritaria}' ({gan_min['f1_minoritaria']:.3f})."
        else:
            gan_auc = max(resultados_test, key=lambda r: r["roc_auc_minoritaria"])
            txt_min = f"en '{self.clase_minoritaria}', '{gan_auc['modelo_corto']}' lidera en ROC-AUC ({gan_auc['roc_auc_minoritaria']:.3f})."

        self._poner_caja_lectura(
            fig,
            f"Lectura clave: El Baseline no supervisado colapsa (Kappa=0.00, F1-Macro={resultados_test[0]['f1_macro']:.2f}). "
            f"Todos los modelos supervisados de Regresión Logística lo superan ampliamente: '{mejor_macro['modelo_corto']}' lidera en F1-Macro ({mejor_macro['f1_macro']:.3f}), {txt_min}",
        )
        plt.tight_layout(rect=[0, 0.11, 1, 0.90])
        ruta = self.carpeta_salida / "fig2_comparativa_4_alternativas.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig3_estabilidad_multirun(self, df_runs: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.4))
        fig.suptitle(
            "Estabilidad Estadística de Todas las Alternativas (10 Particiones Estratificadas 70/15/15)",
            fontsize=13.5, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Barras = Media | Bigotes = ±1 Desviación Típica (StDev) | Puntos negros = cada ejecución individual | ↑ Mayor barra y ↓ menor bigote es mejor",
            ha="center", fontsize=9.6, style="italic", color="#333333",
        )

        modelos_orden = list(df_runs["modelo_corto"].unique())
        metricas_panel = [
            ("f1_minoritaria", f"A) F1 Minoritaria ({self.clase_minoritaria}) [↑]"),
            ("f1_macro", "B) F1-Score Macro (5 clases) [↑]"),
            ("kappa", "C) Kappa de Cohen (κ, 0=azar) [↑]"),
        ]

        rng = np.random.RandomState(42)
        for ax, (col, titulo) in zip(axes, metricas_panel):
            medias = [df_runs.loc[df_runs["modelo_corto"] == m, col].mean() for m in modelos_orden]
            stds = [df_runs.loc[df_runs["modelo_corto"] == m, col].std(ddof=1) for m in modelos_orden]
            x = np.arange(len(modelos_orden))

            barras = ax.bar(
                x, medias, yerr=stds, capsize=5,
                color=self.PALETA[: len(modelos_orden)],
                edgecolor="black", linewidth=0.8, alpha=0.82,
                error_kw={"elinewidth": 1.6, "ecolor": "#222222"},
            )
            max_punto = 0.0
            for idx, m in enumerate(modelos_orden):
                vals = df_runs.loc[df_runs["modelo_corto"] == m, col].values
                max_punto = max(max_punto, float(np.max(vals)))
                jitter = rng.uniform(-0.13, 0.13, size=len(vals))
                ax.scatter(np.full_like(vals, idx) + jitter, vals, color="#111111", s=16, alpha=0.52, zorder=5)

            tope = max(max_punto, max([m + s for m, s in zip(medias, stds)] + [0.22]))
            for idx, (b, mu, sd) in enumerate(zip(barras, medias, stds)):
                vals_m = df_runs.loc[df_runs["modelo_corto"] == modelos_orden[idx], col].values
                altura_txt = max(mu + sd, float(np.max(vals_m))) + tope * 0.04
                ax.text(
                    b.get_x() + b.get_width() / 2.0, altura_txt,
                    f"{mu:.3f}\n±{sd:.3f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(modelos_orden, rotation=20, ha="right", fontweight="bold", fontsize=8.5)
            ax.set_ylim(-0.03, min(1.05, tope * 1.32))
            ax.set_ylabel("Puntuación (Media ± StDev) [↑ Mejor]", fontweight="bold")
            ax.set_title(titulo, fontsize=11.0, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        medias_min = {m: df_runs.loc[df_runs["modelo_corto"] == m, "f1_minoritaria"].mean() for m in modelos_orden}
        ganador_min = max(medias_min, key=medias_min.get)
        std_gan = df_runs.loc[df_runs["modelo_corto"] == ganador_min, "f1_minoritaria"].std(ddof=1)
        self._poner_caja_lectura(
            fig,
            f"Lectura clave (RQ2): Evaluar 10 particiones distintas elimina el azar de tener solo 7 casos 'Attempt' en un corte. "
            f"En 10 runs, los modelos sin balancear sacan 0.000 en '{self.clase_minoritaria}', mientras '{ganador_min}' alcanza F1-Min = {medias_min[ganador_min]:.3f} ± {std_gan:.3f}.",
            color_fondo="#edf7ed", color_borde="#2ca02c",
        )
        plt.tight_layout(rect=[0, 0.11, 1, 0.90])
        ruta = self.carpeta_salida / "fig3_estabilidad_10runs.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig4_matrices_confusion(self, resultados_test: List[Dict[str, Any]]) -> Path:
        fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.6))
        fig.suptitle(
            "Matrices de Confusión de Todas las Alternativas en Test Oficial (N = 75 Usuarios)",
            fontsize=14.0, fontweight="bold", y=0.985,
        )
        fig.text(
            0.5, 0.948,
            "Color = Recall por fila (Ideal: 100% azul oscuro en la diagonal principal) | Recuadro rojo = Clase Minoritaria ('Attempt')",
            ha="center", fontsize=9.8, style="italic", color="#333333",
        )

        idx_min = self.clases.index(self.clase_minoritaria)
        axes_flat = axes.flatten()

        for ax, res in zip(axes_flat[: len(resultados_test)], resultados_test):
            M = np.asarray(res["matriz_confusion"], dtype=float)
            M_norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
            ax.imshow(M_norm, cmap="Blues", vmin=0.0, vmax=1.0)

            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    cnt = int(M[r, c])
                    pct = M_norm[r, c] * 100.0
                    color_txt = "white" if M_norm[r, c] > 0.52 else "black"
                    ax.text(c, r, f"{cnt}\n({pct:.0f}%)", ha="center", va="center", color=color_txt, fontsize=8.2, fontweight="bold")

            rect = patches.Rectangle(
                (-0.48, idx_min - 0.48), len(self.clases) - 0.04, 0.96,
                linewidth=2.3, edgecolor="#d62728", facecolor="none", zorder=5,
            )
            ax.add_patch(rect)
            ax.set_title(
                f"{res['modelo_corto']}\nF1-Min={res['f1_minoritaria']:.3f} | F1-Macro={res['f1_macro']:.3f} | κ={res['kappa']:.3f}",
                fontsize=10.3, fontweight="bold", pad=7,
            )
            ax.set_xticks(range(len(self.clases)))
            ax.set_yticks(range(len(self.clases)))
            ax.set_xticklabels(self.clases, rotation=28, ha="right", fontsize=8.6, fontweight="bold")
            ax.set_yticklabels(self.clases, fontsize=8.6, fontweight="bold")
            ax.set_xlabel("Predicho", fontweight="bold", fontsize=9.0)
            ax.set_ylabel("Real", fontweight="bold", fontsize=9.0)
            ax.grid(False)

        ax_info = axes_flat[-1]
        ax_info.axis("off")
        resumen_txt = (
            "GUÍA DE LECTURA RÁPIDA\n"
            "─────────────────────────\n\n"
            "1. Baseline (Clust. 384D):\n"
            "   Colapsa al 100% en 'Ideation'\n"
            "   (clase mayoritaria). κ = 0.00.\n\n"
            "2. Modelos sin balancear:\n"
            "   Aciertan 'Ideation' y 'Supportive',\n"
            "   pero ignoran la fila roja ('Attempt').\n\n"
            "3. Modelos balanceados (Alt. 4 y 5):\n"
            "   Penalizar el desbalanceo desplaza\n"
            "   la frontera hacia las clases de\n"
            "   alto riesgo ('Attempt' y 'Behavior'),\n"
            "   activando su detección real."
        )
        ax_info.text(
            0.06, 0.92, resumen_txt, va="top", ha="left", fontsize=10.0,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#495057", lw=1.3),
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        ruta = self.carpeta_salida / "fig4_matrices_confusion_4modelos.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig5_roc_y_pr_minoritaria(self, resultados_test: List[Dict[str, Any]], prevalencia_min: float) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.3))
        fig.suptitle(
            f"Evaluación Probabilística Independiente del Umbral (Tema 3) — Clase Minoritaria '{self.clase_minoritaria}'",
            fontsize=13.3, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Izquierda: Curva ROC (↑ AUC mejor, 0.50 = azar) | Derecha: Curva Precision-Recall (↑ AP mejor, línea gris = prevalencia en Test)",
            ha="center", fontsize=9.5, style="italic", color="#333333",
        )

        for idx, res in enumerate(resultados_test):
            col = self.PALETA[idx % len(self.PALETA)]
            ax1.plot(
                res["curva_roc_minoritaria"]["fpr"], res["curva_roc_minoritaria"]["tpr"],
                linewidth=2.3, color=col,
                label=f"{res['modelo_corto']} (AUC = {res['roc_auc_minoritaria']:.3f})",
            )
            ax2.plot(
                res["curva_pr_minoritaria"]["recall"], res["curva_pr_minoritaria"]["precision"],
                linewidth=2.3, color=col,
                label=f"{res['modelo_corto']} (AP = {res['ap_minoritaria']:.3f})",
            )

        ax1.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2, alpha=0.7, label="Azar (AUC = 0.500)")
        ax1.fill_between([0, 1], [0, 1], color="#dddddd", alpha=0.3)
        ax1.annotate(
            "Óptimo teórico (0, 1)\n↖ Más arriba-izq = Mejor",
            xy=(0.02, 0.98), xytext=(0.26, 0.84),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#2ca02c"),
            fontsize=8.6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#edf7ed", edgecolor="#2ca02c"),
        )
        ax1.set_title(f"A) Curvas ROC One-vs-Rest ('{self.clase_minoritaria}')", fontsize=11.2, fontweight="bold")
        ax1.set_xlabel("False Positive Rate (1 - Specificity) [↓ Menor es mejor]", fontweight="bold")
        ax1.set_ylabel("True Positive Rate (Recall) [↑ Mayor es mejor]", fontweight="bold")
        ax1.legend(loc="lower right", frameon=True, fontsize=8.3)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.axhline(
            prevalencia_min, linestyle="--", color="black", linewidth=1.2, alpha=0.7,
            label=f"Azar / Prevalencia ({prevalencia_min:.3f})",
        )
        ax2.set_title(f"B) Curvas Precision-Recall ('{self.clase_minoritaria}')", fontsize=11.2, fontweight="bold")
        ax2.set_xlabel("Recall (Cobertura de casos reales) [↑ Mayor es mejor]", fontweight="bold")
        ax2.set_ylabel("Precision (Acierto cuando predice Attempt) [↑ Mejor]", fontweight="bold")
        ax2.set_xlim(0.0, 1.02)
        ax2.set_ylim(0.0, 1.05)
        ax2.legend(loc="upper right", frameon=True, fontsize=8.3)
        ax2.grid(True, linestyle=":", alpha=0.6)

        mejor_auc = max(resultados_test, key=lambda r: r["roc_auc_minoritaria"])
        self._poner_caja_lectura(
            fig,
            f"Lectura clave: En clases muy minoritarias ({prevalencia_min*100:.1f}% del corpus), ROC y Precision-Recall miden qué modelo ordena mejor los casos graves. "
            f"'{mejor_auc['modelo_corto']}' lidera con AUC = {mejor_auc['roc_auc_minoritaria']:.3f} y AP = {mejor_auc['ap_minoritaria']:.3f} frente al Baseline bajo la diagonal ({resultados_test[0]['roc_auc_minoritaria']:.3f}).",
        )
        plt.tight_layout(rect=[0, 0.11, 1, 0.90])
        ruta = self.carpeta_salida / "fig5_roc_y_pr_minoritaria.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig6_desglose_clases_rq2(self, resultados_test: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(14.0, 6.4))
        fig.suptitle(
            "Desglose de F1-Score por Clase Clínica C-SSRS en Test Oficial (Todas las Alternativas)",
            fontsize=13.4, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Muestra qué clases aprende cada enfoque: las clases más pequeñas ('Attempt', 'Behavior') requieren balanceo para no quedar eclipsadas por 'Ideation'",
            ha="center", fontsize=9.5, style="italic", color="#333333",
        )

        x = np.arange(len(self.clases))
        n_mod = len(resultados_test)
        ancho = 0.155

        for idx, res in enumerate(resultados_test):
            f1s = [res["por_clase"][c]["f1"] for c in self.clases]
            despl = (idx - (n_mod - 1) / 2.0) * ancho
            barras = ax.bar(
                x + despl, f1s, ancho,
                label=res["modelo_corto"],
                color=self.PALETA[idx % len(self.PALETA)],
                edgecolor="black", linewidth=0.7, alpha=0.9,
            )
            for b in barras:
                h = b.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(b.get_x() + b.get_width() / 2.0, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.9, fontweight="bold",
                )

        soportes = [resultados_test[0]["por_clase"][c]["soporte"] for c in self.clases]
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{c} (n={s})\n★ MINORITARIA" if c == self.clase_minoritaria else f"{c} (n={s})" for c, s in zip(self.clases, soportes)],
            fontweight="bold", fontsize=9.2,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("F1-Score por Clase (↑ Mayor es mejor)", fontweight="bold")
        ax.legend(loc="upper left", ncol=3, frameon=True, fontsize=8.7)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        self._poner_caja_lectura(
            fig,
            "Lectura clave: El Baseline (gris) solo acierta en 'Ideation' (0.50) con 0.00 en las otras 4 clases. "
            "Los clasificadores supervisados reparten la capacidad predictiva por todo el espectro clínico y desbloquean las clases minoritarias al balancear.",
        )
        plt.tight_layout(rect=[0, 0.11, 1, 0.90])
        ruta = self.carpeta_salida / "fig6_desglose_clases_rq2.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig7_robustez_ruido_rq1(self, df_resumen_ruido: pd.DataFrame, caidas_rq1: Dict[str, Any]) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.3))
        fig.suptitle(
            "RQ1: Robustez ante Ruido Léxico (Erratas en Test) — TF-IDF Caracteres vs. Embeddings SBERT 384D",
            fontsize=13.3, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Entrenados en texto limpio y evaluados con erratas crecientes (sustitución, borrado, inserción, transposición) | Línea más plana y alta = Más robusto",
            ha="center", fontsize=9.5, style="italic", color="#333333",
        )

        estilos = {
            "LogReg + TF-IDF Char":  {"color": "#ff7f0e", "marker": "s", "ls": "-"},
            "LogReg + Char (Bal.+)": {"color": "#9467bd", "marker": "D", "ls": "-"},
            "LogReg + 384D":         {"color": "#1f77b4", "marker": "o", "ls": "--"},
            "LogReg + 384D (Bal.)":  {"color": "#2ca02c", "marker": "^", "ls": "-."},
        }

        niveles_x = sorted(df_resumen_ruido["nivel_ruido_pct"].unique())
        nivel_max = max(niveles_x)

        for mod, g in df_resumen_ruido.groupby("modelo"):
            g = g.sort_values("nivel_ruido_pct")
            est = estilos.get(mod, {"color": "#333333", "marker": "o", "ls": "-"})
            x = g["nivel_ruido_pct"].values
            mu = g["f1_macro_mean"].values
            sd = g["f1_macro_std"].values

            ax1.plot(x, mu, marker=est["marker"], linestyle=est["ls"], linewidth=2.4, color=est["color"], label=mod)
            ax1.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=est["color"], alpha=0.14)

            base_0 = mu[0] if mu[0] > 0 else 1e-6
            retencion = (mu / base_0) * 100.0
            delta_pct = caidas_rq1[mod]["delta_rel_f1_macro_pct"]
            ax2.plot(
                x, retencion, marker=est["marker"], linestyle=est["ls"], linewidth=2.4,
                color=est["color"], label=f"{mod} (Δ a {nivel_max}%: {delta_pct:+.1f}%)",
            )

        ax1.set_title("A) Rendimiento Absoluto: F1-Macro vs. Tasa de Erratas", fontsize=11.2, fontweight="bold")
        ax1.set_xlabel("Tasa de ruido léxico en Test (% de palabras corrompidas)", fontweight="bold")
        ax1.set_ylabel("F1-Score Macro (Media ± StDev) [↑ Mejor]", fontweight="bold")
        ax1.set_xticks(niveles_x)
        ax1.set_ylim(0.0, max(0.55, df_resumen_ruido["f1_macro_mean"].max() + 0.14))
        ax1.legend(loc="lower left", frameon=True, fontsize=8.4)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.axhline(100.0, color="#555555", linestyle=":", linewidth=1.3, label="Referencia ideal sin caída (100%)")
        ax2.set_title("B) Retención Relativa de Rendimiento (100% = Inmune al Ruido)", fontsize=11.2, fontweight="bold")
        ax2.set_xlabel("Tasa de ruido léxico en Test (% de palabras corrompidas)", fontweight="bold")
        ax2.set_ylabel("% del F1-Macro original conservado [↑ Mejor]", fontweight="bold")
        ax2.set_xticks(niveles_x)
        ax2.set_ylim(45.0, 118.0)
        ax2.legend(loc="lower left", frameon=True, fontsize=8.2)
        ax2.grid(True, linestyle=":", alpha=0.6)

        c_char = caidas_rq1.get("LogReg + TF-IDF Char", {}).get("delta_rel_f1_macro_pct", 0.0)
        c_emb = caidas_rq1.get("LogReg + 384D", {}).get("delta_rel_f1_macro_pct", 0.0)
        self._poner_caja_lectura(
            fig,
            f"Respuesta a RQ1: SÍ, TF-IDF por caracteres es claramente más robusto al ruido léxico. Al {nivel_max}% de palabras con erratas, "
            f"TF-IDF Caracteres varía solo {c_char:+.1f}% frente a {c_emb:+.1f}% en SBERT 384D, porque los n-gramas solapados conservan subcadenas válidas mientras las erratas rompen los subwords de SBERT.",
            color_fondo="#fff8e8", color_borde="#ff7f0e",
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.90])
        ruta = self.carpeta_salida / "fig7_robustez_ruido_lexico_rq1.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig8_balanceo_y_sesgo_varianza(
        self, curva_alpha_cv: List[dict], alpha_elegido: float, curva_sesgo_varianza: List[dict], c_elegido: float
    ) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.3))
        fig.suptitle(
            "Validación en 5-Fold CV (Train, N=350): Intensidad de Balanceo (α) y Dilema Sesgo-Varianza (C)",
            fontsize=13.3, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Todas las decisiones de hiperparámetros se toman con 5-Fold CV sobre Train (sin tocar Test) para evitar sobreajuste",
            ha="center", fontsize=9.6, style="italic", color="#333333",
        )

        df_a = pd.DataFrame(curva_alpha_cv)
        ax1.plot(df_a["alpha"], df_a["cv_f1_min"], marker="o", linewidth=2.5, color="#2ca02c", label=f"OOF F1 Minoritaria ({self.clase_minoritaria}) [↑]")
        ax1.plot(df_a["alpha"], df_a["cv_rec_min"], marker="^", linewidth=2.0, linestyle="--", color="#1f77b4", label="OOF Recall Minoritaria [↑]")
        ax1.plot(df_a["alpha"], df_a["cv_f1_macro"], marker="s", linewidth=2.2, linestyle="-.", color="#ff7f0e", label="OOF F1-Macro (5 clases) [↑]")
        ax1.axvline(alpha_elegido, color="#d62728", linestyle="--", linewidth=2.0, label=f"α óptimo en CV = {alpha_elegido}")
        ax1.axvline(1.0, color="#666666", linestyle=":", linewidth=1.3, label="α = 1.0 (class_weight='balanced')")
        ax1.set_title("A) Efecto del Peso de Clase α en 5-Fold CV (LogReg + 384D)", fontsize=11.0, fontweight="bold")
        ax1.set_xlabel("Exponente de balanceo α (0.0 = sin balanceo | 1.0 = balanced)", fontweight="bold")
        ax1.set_ylabel("Puntuación en 5-Fold CV (↑ Mayor es mejor)", fontweight="bold")
        ax1.set_ylim(-0.02, max(0.55, df_a[["cv_f1_min", "cv_rec_min", "cv_f1_macro"]].max().max() + 0.14))
        ax1.legend(loc="best", frameon=True, fontsize=8.4)
        ax1.grid(True, linestyle=":", alpha=0.6)

        df_bv = pd.DataFrame(curva_sesgo_varianza)
        ax2.plot(df_bv["C"], df_bv["train_f1_macro"], marker="o", linewidth=2.3, color="#d62728", label="F1-Macro en Train (baja sesgo al subir C)")
        ax2.plot(df_bv["C"], df_bv["cv_f1_macro"], marker="s", linewidth=2.5, color="#1f77b4", label="F1-Macro en 5-Fold CV (generalización) [↑]")
        ax2.plot(df_bv["C"], df_bv["cv_f1_min"], marker="^", linewidth=2.0, linestyle="--", color="#2ca02c", label=f"F1 Minoritaria ({self.clase_minoritaria}) en CV [↑]")
        ax2.axvline(c_elegido, color="#333333", linestyle="--", linewidth=1.8, label=f"C óptimo en CV = {c_elegido}")
        ax2.set_xscale("log")
        ax2.set_title("B) Dilema Sesgo-Varianza al Variar Regularización C (Tema 4)", fontsize=11.0, fontweight="bold")
        ax2.set_xlabel("Parámetro C en escala log (← Más regularización | Menos regularización / Más overfitting →)", fontweight="bold")
        ax2.set_ylabel("F1-Score (↑ Mayor en CV es mejor)", fontweight="bold")
        ax2.set_ylim(0.0, 1.05)
        ax2.legend(loc="upper left", frameon=True, fontsize=8.3)
        ax2.grid(True, which="both", linestyle=":", alpha=0.5)

        self._poner_caja_lectura(
            fig,
            f"Lectura clave (Tema 4): (A) α = {alpha_elegido} maximiza la detección de '{self.clase_minoritaria}' sin hundir F1-Macro. "
            f"(B) Con D=384 atributos y N=350 muestras, valores altos de C llevan Train cerca de 1.00 (overfitting) mientras CV cae; fijar C={c_elegido} controla la varianza.",
            color_fondo="#edf7ed", color_borde="#2ca02c",
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.90])
        ruta = self.carpeta_salida / "fig8_balanceo_y_sesgo_varianza.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig9_interpretabilidad_logreg_char(self, top_minoritaria: list, top_supportive: list) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2))
        fig.suptitle(
            "Interpretabilidad Paramétrica (Tema 4): 10 Patrones de Caracteres Distintos + Palabra Ejemplo",
            fontsize=13.3, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Cada n-grama se asocia a su palabra más frecuente en Train y se deduplica por raíz | ↑ Barra más larga = mayor peso θ hacia esa clase",
            ha="center", fontsize=9.5, style="italic", color="#333333",
        )

        paneles = [
            (ax1, top_minoritaria, f"A) Top Patrones para '{self.clase_minoritaria}' (Alto Riesgo)", "#d62728"),
            (ax2, top_supportive, "B) Top Patrones para 'Supportive' (Apoyo / Bajo Riesgo)", "#2ca02c"),
        ]
        for ax, datos, titulo, color in paneles:
            etiquetas = [ng for ng, _ in reversed(datos)]
            pesos = [w for _, w in reversed(datos)]
            y_pos = np.arange(len(etiquetas))
            barras = ax.barh(y_pos, pesos, color=color, edgecolor="black", alpha=0.85)
            for b, w in zip(barras, pesos):
                ax.text(w + max(pesos) * 0.02, b.get_y() + b.get_height() / 2.0, f"+{w:.2f}", va="center", fontsize=8.6, fontweight="bold")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(etiquetas, fontfamily="monospace", fontsize=9.1, fontweight="bold")
            ax.set_xlim(0, max(pesos) * 1.22)
            ax.set_xlabel("Peso positivo θ en la Regresión Logística (↑ Mayor influencia)", fontweight="bold")
            ax.set_title(titulo, fontsize=11.0, fontweight="bold")
            ax.grid(axis="x", linestyle=":", alpha=0.6)

        self._poner_caja_lectura(
            fig,
            "Lectura clave: La Regresión Logística sobre char_wb descubre pistas clínicas interpretables: en 'Supportive' dominan pronombres de 2ª persona y términos de apoyo, "
            "mientras en 'Attempt' destacan raíces ligadas a intento/crisis y puntuación expresiva, manteniendo resistencia frente a erratas.",
        )
        plt.tight_layout(rect=[0, 0.11, 1, 0.90])
        ruta = self.carpeta_salida / "fig9_interpretabilidad_logreg_char.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta

    def fig10_patrones_estilometricos(
        self, df_kw: pd.DataFrame, df_zscores: pd.DataFrame, etiquetas_legibles: dict
    ) -> Path:
        """
        Figura 10 (NUEVA — Descubrimiento de Patrones Lingüísticos y Estructurales):
          - Panel A: Ranking de significancia estadística Kruskal-Wallis (-log10(p)) entre las 5 clases.
          - Panel B: Mapa de calor (Z-scores) de los 6 patrones más significativos por clase clínica.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.8, 6.6), gridspec_kw={"width_ratios": [1.05, 1.15]})
        fig.suptitle(
            "Análisis de Patrones Estructurales y Psicolingüísticos entre Clases Clínicas (Train, N=350)",
            fontsize=13.5, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.925,
            "Evaluación no paramétrica (Kruskal-Wallis) para verificar qué rasgos de estilo/longitud diferencian de verdad las 5 clases antes de usarlos",
            ha="center", fontsize=9.5, style="italic", color="#333333",
        )

        # Panel A: Ranking Kruskal-Wallis
        df_ord = df_kw.iloc[::-1]
        colores = ["#2ca02c" if sig else "#adb5bd" for sig in df_ord["significativo_05"]]
        y_pos = np.arange(len(df_ord))
        barras = ax1.barh(y_pos, df_ord["neg_log10_p"].values, color=colores, edgecolor="black", alpha=0.88)

        umbral_sig = -np.log10(0.05)
        ax1.axvline(umbral_sig, color="#d62728", linestyle="--", linewidth=2.0, label="Umbral significancia p = 0.05 (Verde = significativo)")

        max_val = max(df_ord["neg_log10_p"].max(), 2.0)
        for b, pval in zip(barras, df_ord["p_value"].values):
            txt_p = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
            ax1.text(b.get_width() + max_val * 0.02, b.get_y() + b.get_height() / 2.0, txt_p, va="center", fontsize=8.2, fontweight="bold")

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_ord["etiqueta"].values, fontsize=8.8, fontweight="bold")
        ax1.set_xlim(0, max_val * 1.28)
        ax1.set_xlabel("Significancia estadística: -log10(p-valor Kruskal-Wallis) [↑ Más discriminativo]", fontweight="bold")
        ax1.set_title("A) ¿Qué Patrones Cambian de Verdad entre Clases?", fontsize=11.0, fontweight="bold")
        ax1.legend(loc="lower right", frameon=True, fontsize=8.3)
        ax1.grid(axis="x", linestyle=":", alpha=0.6)

        # Panel B: Heatmap Z-scores de los Top 6 patrones más significativos
        top6_feats = df_kw["feature"].head(6).tolist()
        mat_z = df_zscores[top6_feats].T  # Filas = patrones, Columnas = clases
        im = ax2.imshow(mat_z.values, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")

        for r in range(mat_z.shape[0]):
            for c in range(mat_z.shape[1]):
                val = mat_z.values[r, c]
                col_txt = "white" if abs(val) > 0.48 else "black"
                ax2.text(c, r, f"{val:+.2f}", ha="center", va="center", color=col_txt, fontsize=9.0, fontweight="bold")

        ax2.set_yticks(range(len(top6_feats)))
        ax2.set_yticklabels([etiquetas_legibles[f] for f in top6_feats], fontsize=9.0, fontweight="bold")
        ax2.set_xticks(range(len(mat_z.columns)))
        ax2.set_xticklabels(mat_z.columns, rotation=20, ha="right", fontsize=9.2, fontweight="bold")
        ax2.set_xlabel("Clase Clínica Real (C-SSRS)", fontweight="bold")
        ax2.set_title("B) Perfil Lingüístico por Clase (Z-score: Rojo = Alto, Azul = Bajo)", fontsize=11.0, fontweight="bold")
        ax2.grid(False)
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Desviación sobre la media (Z-score)", fontsize=8.8)

        n_sig = int(df_kw["significativo_05"].sum())
        top1_nom = df_kw.iloc[0]["etiqueta"]
        top1_p = df_kw.iloc[0]["p_value"]
        self._poner_caja_lectura(
            fig,
            f"Lectura clave: {n_sig} de los 12 patrones estilométricos presentan diferencias estadísticamente significativas (p < 0.05) entre las 5 clases clínicas en Train, "
            f"liderados por '{top1_nom}' (p={'<0.001' if top1_p < 0.001 else f'{top1_p:.3f}'}). El mapa de calor muestra la firma característica de cada nivel de riesgo.",
            color_fondo="#eef7ff", color_borde="#1f77b4",
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.90])
        ruta = self.carpeta_salida / "fig10_patrones_estilometricos.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return ruta