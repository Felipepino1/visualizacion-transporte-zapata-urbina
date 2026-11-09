"""Estilo gráfico común para todas las figuras del proyecto.

Paleta categórica validada para daltonismo (ΔE CVD >= 8 entre colores adyacentes):
se asigna SIEMPRE en el mismo orden y cada color sigue a la misma entidad en todas
las figuras (doméstico = azul, internacional = naranjo), nunca al ranking.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8984"
GRILLA = "#e4e3df"

AZUL = "#2a78d6"      # slot 1
NARANJO = "#eb6834"   # slot 2
AQUA = "#1baf7a"      # slot 3
AMARILLO = "#eda100"  # slot 4

COLOR_TRAFICO = {"Doméstico": AZUL, "Internacional": NARANJO}
COLOR_CATEGORIA_PESO = {
    "Ligera (<5,7 t)": AQUA,
    "Mediana (5,7–50 t)": AZUL,
    "Pesada (>50 t)": NARANJO,
}
# Rampa secuencial de un solo tono (azul) para magnitudes (heatmaps)
RAMPA_AZUL = ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def aplicar_estilo() -> None:
    """Configura matplotlib con ejes recesivos, grilla tenue y tipografía legible."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 24,
        "axes.axisbelow": True,
        "axes.labelsize": 10.5,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": GRILLA,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "text.color": TINTA,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=[AZUL, NARANJO, AQUA, AMARILLO]),
    })


def subtitulo(ax, texto: str) -> None:
    """Agrega un subtítulo en tinta secundaria bajo el título del eje."""
    ax.text(0, 1.01, texto, transform=ax.transAxes, fontsize=9.5,
            color=TINTA_2, va="bottom", ha="left")


def nota_fuente(fig, texto: str = "Fuente: JAC/DGAC, «Operaciones de Aeronaves», datos.gob.cl (CC0).") -> None:
    fig.text(0.01, -0.02, texto, fontsize=8, color=TINTA_3, ha="left", va="top")


def miles(x, _pos=None) -> str:
    """Formateador de eje con separador de miles chileno."""
    return f"{x:,.0f}".replace(",", ".")


def guardar(fig, nombre: str, carpeta) -> None:
    ruta = carpeta / f"{nombre}.png"
    fig.savefig(ruta)
    print(f"Figura guardada: {ruta.relative_to(carpeta.parents[1])}")
    plt.show()
