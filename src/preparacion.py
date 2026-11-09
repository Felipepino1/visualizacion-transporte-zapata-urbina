"""Funciones de carga, limpieza y derivación de variables.

Todas las transformaciones se aplican sobre copias en memoria: los archivos de
data/raw/ nunca se modifican.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Zonas horarias locales por aeropuerto. La documentación JAC indica que mes_id
# se construye en la hora local de cada aeropuerto (p. ej. America/Punta_Arenas).
# Magallanes mantiene un horario propio (zona IANA America/Punta_Arenas) y la
# Isla de Pascua tiene su propio huso; el resto del país usa America/Santiago.
# Esta asignación se valida contra el CSV publicado (ver notebook, sección 7).
ZONA_HORARIA = {
    "SCIP": "Pacific/Easter",          # Isla de Pascua (Mataveri)
    "SCCI": "America/Punta_Arenas",    # Punta Arenas
    "SCNT": "America/Punta_Arenas",    # Puerto Natales
    "SCGZ": "America/Punta_Arenas",    # Puerto Williams
    "SCFM": "America/Punta_Arenas",    # Porvenir
}
ZONA_POR_DEFECTO = "America/Santiago"

# Umbral OACI de aeronave ligera: peso máximo de despegue de 5.700 kg.
UMBRAL_LIGERA_T = 5.7
UMBRAL_PESADA_T = 50.0
PMD_MAX_PLAUSIBLE_T = 600.0   # sobre esto sólo es real el An-225 (A225, 640 t)

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def cargar_bitacora(ruta: Path) -> pd.DataFrame:
    """Lee el parquet con pyarrow y convierte cadenas a category.

    - pmd viene como decimal128 (pandas lo dejaría como objetos Decimal, lentos):
      se convierte a float64 en pyarrow antes de pasar a pandas.
    - Las columnas de texto tienen baja cardinalidad relativa a 11 M de filas,
      por lo que `category` reduce la memoria de ~2,6 GB a ~0,36 GB.
    """
    tabla = pq.read_table(ruta)
    i = tabla.schema.get_field_index("pmd")
    tabla = tabla.set_column(i, "pmd", pc.cast(tabla["pmd"], pa.float64()))
    return tabla.to_pandas(strings_to_categorical=True)


def vacios_a_nulos(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Homologa cadenas vacías o de sólo espacios como valores ausentes (NaN)."""
    df = df.copy()
    for c in columnas:
        cats = df[c].cat.categories
        vacias = [x for x in cats if str(x).strip() == ""]
        if vacias:
            df[c] = df[c].cat.remove_categories(vacias)
    return df


def agregar_hora_local(df: pd.DataFrame) -> pd.DataFrame:
    """Crea dt_local (hora local del aeropuerto, sin zona) a partir del instante UTC."""
    df = df.copy()
    utc = df["dt_operacion"].dt.tz_convert("UTC")
    zona = df["aeropuerto_oaci"].astype(str).map(ZONA_HORARIA).fillna(ZONA_POR_DEFECTO)
    local = pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")
    for z in zona.unique():
        m = zona == z
        local[m] = utc[m].dt.tz_convert(z).dt.tz_localize(None)
    df["dt_local"] = local
    return df


def categoria_peso(pmd: pd.Series) -> pd.Series:
    cat = pd.cut(pmd, bins=[-np.inf, UMBRAL_LIGERA_T, UMBRAL_PESADA_T, np.inf],
                 labels=["Ligera (<5,7 t)", "Mediana (5,7–50 t)", "Pesada (>50 t)"],
                 right=False)
    return cat


def preparar_mensual(csv: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y variables derivadas del resumen mensual por aeropuerto."""
    df = csv.rename(columns={"cnt_operaciones": "n_operaciones"}).copy()
    df["aeropuerto_oaci"] = df["aeropuerto_oaci"].str.strip().str.upper()
    df["internacional_domestico"] = df["internacional_domestico"].str.strip().str.upper()
    df["fecha"] = pd.to_datetime(df["mes_id"].astype(str), format="%Y%m")
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["tipo_trafico"] = df["internacional_domestico"].map({"D": "Doméstico", "I": "Internacional"})
    return df
