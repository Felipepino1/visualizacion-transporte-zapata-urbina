# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluación 2 — Ingeniería y Visualización de Datos de Transporte
#
# **Autor:** Felipe Ignacio Zapata Lagos · Ingeniería en Informática · INACAP
# **Asignatura:** Visualización de Datos · **Unidad:** Diseño e implementación de gráficos para datos
#
# **Fuente:** Junta de Aeronáutica Civil (JAC), *Operaciones de Aeronaves*, Portal de Datos Abiertos
# (https://datos.gob.cl/dataset/operaciones-aeronaves), licencia CC0. Detalle de la investigación de
# la fuente (Parte I) y de la formulación del problema (Parte II) en `docs/`.
#
# **Pregunta principal.** ¿Cómo se distribuye y cómo ha evolucionado la actividad aeroportuaria
# registrada en Chile entre 2000 y 2026 según aeropuerto, tipo de tráfico (doméstico/internacional),
# tamaño de aeronave y momento del día, y qué patrones temporales la caracterizan?
#
# | | Pregunta secundaria |
# |---|---|
# | **P1** | ¿Qué aeropuertos concentran las operaciones y qué peso tiene el tráfico internacional en cada uno? |
# | **P2** | ¿Cómo evolucionaron las operaciones domésticas e internacionales entre 2000 y 2025 y cómo se recuperó cada una tras la caída de 2020? |
# | **P3** | ¿Cómo se distribuyen las operaciones según tamaño de aeronave y hora del día, y difieren esos patrones entre segmentos? |
#
# **H1 (exploratoria, no es conclusión):** tras la caída de 2020, el tráfico internacional se
# recuperó más lentamente que el doméstico respecto de su nivel de 2019.
#
# > Regla de trabajo: **dato → evidencia → interpretación → posible acción**. No se afirma
# > causalidad a partir de patrones conjuntos.

# %% [markdown]
# ## 1. Importaciones

# %%
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

# %% [markdown]
# ## 2. Configuración
#
# Las rutas se resuelven desde la raíz del proyecto para que el notebook funcione igual si se
# ejecuta desde `notebooks/` o desde la raíz.

# %%
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))

import estilo  # noqa: E402
import preparacion as prep  # noqa: E402

RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "output" / "figures"
for carpeta in (PROCESSED, FIGURAS):
    carpeta.mkdir(parents=True, exist_ok=True)

RUTA_CSV = RAW / "operaciones-aeropuertos.csv"
RUTA_PARQUET = RAW / "bitacora-vuelos.parquet"
if not RUTA_PARQUET.exists():
    raise FileNotFoundError(
        "Falta data/raw/bitacora-vuelos.parquet (~138 MB, no se versiona en GitHub). "
        "Ejecute desde la raíz:  python src/descargar_datos.py"
    )

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
estilo.aplicar_estilo()
print("Raíz del proyecto:", RAIZ)
print("pandas", pd.__version__, "| seaborn", sns.__version__)

# %% [markdown]
# ## 3. Carga de datos
#
# Se utilizan **ambos recursos** del conjunto:
#
# * `operaciones-aeropuertos.csv` — resumen mensual por aeropuerto y tipo de tráfico (sin sobrevuelos).
# * `bitacora-vuelos.parquet` — un registro por operación (aterrizaje, despegue o sobrevuelo).
#
# El parquet se lee con `pyarrow` convirtiendo los textos a `category` y el `pmd` (decimal) a
# `float64`; leerlo directamente con `pd.read_parquet` ocupa ~2,6 GB de RAM, lo que puede
# hacer inviable el análisis en un computador personal.

# %%
csv_raw = pd.read_csv(RUTA_CSV)
t0 = time.perf_counter()
bit_raw = prep.cargar_bitacora(RUTA_PARQUET)
print(f"Bitácora cargada en {time.perf_counter() - t0:.1f} s | "
      f"{bit_raw.memory_usage(deep=True).sum() / 1e6:,.0f} MB en memoria")
print("CSV:", csv_raw.shape, "| Bitácora:", bit_raw.shape)

# %% [markdown]
# ## 4. Comparación de formatos (Parte VI)
#
# **Los dos archivos publicados NO son equivalentes:** el CSV es un *resumen agregado*
# (aeropuerto × mes × tipo de tráfico, 4 columnas) y el parquet es el *registro individual*
# (11 millones de operaciones, 12 columnas). Comparar directamente su tiempo de carga o
# su tamaño sería una comparación artificial: no contienen la misma información.
#
# Por eso se hace (a) una descripción de cada archivo por separado y (b) un experimento
# controlado: **el mismo subconjunto de datos** (bitácora 2025) se escribe en CSV y en Parquet
# y se comparan tamaño, tiempo de carga, memoria y tipos inferidos.

# %%
meta = pq.ParquetFile(RUTA_PARQUET).metadata
print("Parquet publicado:", f"{RUTA_PARQUET.stat().st_size / 1e6:,.1f} MB,",
      f"{meta.num_rows:,} filas, {meta.num_row_groups} row groups, creado con {meta.created_by}")
print("CSV publicado:    ", f"{RUTA_CSV.stat().st_size / 1e6:,.2f} MB,", f"{len(csv_raw):,} filas")
print("\nTipos declarados en el esquema parquet (se conservan al leer):")
print(pq.read_schema(RUTA_PARQUET))
print("\nTipos inferidos por pandas al leer el CSV (no hay esquema; se infieren del texto):")
print(csv_raw.dtypes)

# %%
experimento = RAIZ / "data" / "tmp_formatos"
experimento.mkdir(exist_ok=True)
sub = bit_raw[bit_raw["dt_operacion"].dt.year == 2025].copy()
sub["pmd"] = sub["pmd"].astype("float64")
for c in sub.select_dtypes("category"):
    sub[c] = sub[c].astype("string")
sub.to_csv(experimento / "bitacora_2025.csv", index=False)
sub.to_parquet(experimento / "bitacora_2025.parquet", index=False, compression="zstd")

filas = []
for fmt, lector in [("csv", pd.read_csv), ("parquet", pd.read_parquet)]:
    ruta = experimento / f"bitacora_2025.{fmt}"
    tiempos = []
    for _ in range(3):
        t0 = time.perf_counter()
        d = lector(ruta)
        tiempos.append(time.perf_counter() - t0)
    filas.append({
        "formato": fmt,
        "tamaño_MB": ruta.stat().st_size / 1e6,
        "carga_s (mediana de 3)": float(np.median(tiempos)),
        "memoria_MB": d.memory_usage(deep=True).sum() / 1e6,
        "dt_operacion": str(d["dt_operacion"].dtype),
        "es_internacional": str(d["es_internacional"].dtype),
        "pmd": str(d["pmd"].dtype),
    })
comparacion_formatos = pd.DataFrame(filas).set_index("formato")
comparacion_formatos

# %% [markdown]
# **Lectura.** Con los mismos datos, Parquet ocupa una fracción del tamaño del CSV, se carga más
# rápido y **conserva los tipos** (fecha con zona horaria y booleano), mientras que el CSV
# entrega la fecha como texto y obliga a reconvertirla. Para la bitácora (11 M de filas) Parquet
# es claramente el formato conveniente; para el resumen mensual (16 mil filas, 285 KB) el CSV es
# suficiente y tiene la ventaja de abrirse en cualquier herramienta. *La carpeta temporal del
# experimento no se versiona.*

# %%
import shutil  # noqa: E402

shutil.rmtree(experimento)
del sub, d

# %% [markdown]
# ## 5. Perfilamiento (Partes VII y VIII)
#
# ### 5.1 Resumen mensual por aeropuerto (CSV)

# %%
csv_raw.head()

# %%
csv_raw.tail()

# %%
csv_raw.sample(5, random_state=42)

# %%
print("shape:", csv_raw.shape)
print("columnas:", list(csv_raw.columns))
csv_raw.info()

# %%
csv_raw.describe(include="all").T

# %%
csv_raw.nunique()

# %% [markdown]
# ### 5.2 Bitácora de vuelos (Parquet)

# %%
bit_raw.head()

# %%
bit_raw.tail()

# %%
bit_raw.sample(5, random_state=42)

# %%
print("shape:", bit_raw.shape)
bit_raw.info(memory_usage="deep")

# %%
bit_raw.describe(include="all").T

# %%
bit_raw.nunique().sort_values()

# %% [markdown]
# ### 5.3 Perfil consolidado

# %%
def perfil(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "cardinalidad": df.nunique(),
        "nulos": df.isna().sum(),
        "%_nulos": (df.isna().mean() * 100).round(2),
    })
    vacios = {}
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object or str(df[c].dtype) == "str":
            s = df[c].astype("string")
            vacios[c] = int((s.str.strip() == "").sum())
    out["cadenas_vacías"] = pd.Series(vacios)
    return out


print("CSV — registros:", len(csv_raw), "| columnas:", csv_raw.shape[1],
      "| duplicados exactos:", csv_raw.duplicated().sum())
display(perfil(csv_raw))
print("Bitácora — registros:", f"{len(bit_raw):,}", "| columnas:", bit_raw.shape[1],
      "| duplicados exactos:", f"{bit_raw.duplicated().sum():,}")
display(perfil(bit_raw))

# %% [markdown]
# **Clasificación inicial de variables**
#
# * **Numéricas:** `cnt_operaciones` (discreta, CSV), `pmd` (continua, toneladas).
# * **Categóricas:** `aeropuerto_oaci`, `internacional_domestico`, `tipo_operacion`, `aerolinea_dgac`,
#   `actividad_cod`, `modelo_avion`, `aeropuerto_dgac_orig_dest`; `es_internacional` (booleana).
# * **Temporales:** `mes_id` (entero AAAAMM, CSV), `dt_operacion` (fecha-hora con zona, bitácora).
# * **Identificadores / alta cardinalidad:** `matricula` (13.835), `numero_vuelo`, `modelo_avion_desc`.
# * **Espaciales:** no hay coordenadas; sólo códigos OACI de aeropuerto.

# %% [markdown]
# ## 6. Auditoría de calidad (Partes IX y X)
#
# Se investiga mediante código cada problema potencial. El objetivo es **detectar lo que
# realmente existe**, no inventar problemas.

# %%
print("Nulos CSV:\n", csv_raw.isna().sum(), sep="")
print("\nDuplicados exactos CSV:", csv_raw.duplicated().sum())
print("Duplicados de clave (mes, aeropuerto, tipo):",
      csv_raw.duplicated(["mes_id", "aeropuerto_oaci", "internacional_domestico"]).sum())
print("Valores de internacional_domestico:", csv_raw["internacional_domestico"].unique())
print("cnt_operaciones <= 0:", (csv_raw["cnt_operaciones"] <= 0).sum())
meses = csv_raw["mes_id"] % 100
print("Meses fuera de 1-12:", (~meses.between(1, 12)).sum())
print("Rango mes_id:", csv_raw["mes_id"].min(), "→", csv_raw["mes_id"].max(),
      "| meses distintos:", csv_raw["mes_id"].nunique(),
      "| meses esperados ene-2000 a sep-2026:", 26 * 12 + 9)

# %%
cobertura = (csv_raw.groupby("aeropuerto_oaci")
             .agg(meses_con_datos=("mes_id", "nunique"), primer_mes=("mes_id", "min"),
                  ultimo_mes=("mes_id", "max"), total_operaciones=("cnt_operaciones", "sum"))
             .sort_values("total_operaciones", ascending=False))
print("Aeropuertos con serie completa (321 meses):", (cobertura["meses_con_datos"] == 321).sum(),
      "de", len(cobertura))
cobertura.head(12)

# %%
# Aeropuertos que dejan de informar o aparecen tarde: cambios reales de infraestructura
cobertura.loc[["SCTI", "SCTC", "SCQP", "SCHA", "SCAT", "SCLC", "SCPF", "SCRG"]]

# %%
# Concentración y asimetría de la medida
print("Asimetría (skew) de cnt_operaciones:", round(csv_raw["cnt_operaciones"].skew(), 2))
print("Registros con 1 sola operación:", (csv_raw["cnt_operaciones"] == 1).sum())
print("Participación de los 2 mayores aeropuertos en el total:",
      f"{cobertura['total_operaciones'].head(2).sum() / cobertura['total_operaciones'].sum():.1%}")

# %% [markdown]
# ### 6.2 Bitácora

# %%
print("Nulos:\n", bit_raw.isna().sum(), sep="")
print("\nValores de tipo_operacion:", bit_raw["tipo_operacion"].value_counts().to_dict())
bit_raw[bit_raw["tipo_operacion"] == "t"]

# %%
for c in ["numero_vuelo", "modelo_avion", "modelo_avion_desc"]:
    s = bit_raw[c].astype("string")
    print(f"{c:18s} nulos={s.isna().sum():>9,}  cadenas vacías={(s.str.strip() == '').sum():>9,}")

# %%
dups = bit_raw[bit_raw.duplicated(keep=False)]
print(f"Filas duplicadas exactas (sobran): {bit_raw.duplicated().sum():,}")
print("Por aeropuerto:", dups["aeropuerto_oaci"].value_counts().head(4).to_dict())
print("Por año:", dups["dt_operacion"].dt.year.value_counts().head(4).to_dict())
dups.sort_values(["matricula", "dt_operacion"]).head(6)

# %%
pmd = bit_raw["pmd"]
print(pmd.describe().round(2))
print("\npmd == 0:", (pmd == 0).sum(), "| pmd nulo:", pmd.isna().sum(),
      f"| pmd > {prep.PMD_MAX_PLAUSIBLE_T:.0f} t:", (pmd > prep.PMD_MAX_PLAUSIBLE_T).sum())
(bit_raw[pmd > prep.PMD_MAX_PLAUSIBLE_T]
 .groupby("modelo_avion", observed=True)["pmd"].agg(["count", "min", "max"])
 .sort_values("count", ascending=False))

# %%
# Referencia: el PMD de un mismo modelo es estable (p. ej. A320 ~ 73-79 t, C172 ~ 1,1 t)
for m in ["A320", "C172", "T34T"]:
    print(m, bit_raw.loc[bit_raw["modelo_avion"] == m, "pmd"].value_counts().head(3).to_dict())

# %%
print("Rango dt_operacion:", bit_raw["dt_operacion"].min(), "→", bit_raw["dt_operacion"].max())
print("Zona horaria almacenada:", bit_raw["dt_operacion"].dt.tz)
print("Registros con fecha local en 1999:", (bit_raw["dt_operacion"].dt.year == 1999).sum())
print("Código de aerolínea numérico (aviación general / privados):",
      bit_raw["aerolinea_dgac"].astype(str).str.match(r"^\d").mean().round(3))

# %% [markdown]
# ### 6.3 Informe de calidad
#
# | Variable | Problema detectado | Evidencia | Decisión adoptada | Justificación |
# |---|---|---|---|---|
# | `cnt_operaciones` (CSV) | Nombre distinto al documentado (`n_operaciones`) | Cabecera del archivo vs. documentación JAC | Renombrar a `n_operaciones` | Homologar con la documentación oficial; no altera valores |
# | `mes_id` (CSV) | Fecha almacenada como entero AAAAMM | `dtype int64`; 321 valores, todos con mes 1-12 | Convertir a `fecha` (datetime) y derivar `anio`, `mes` | Permite ordenar, filtrar y graficar como eje temporal real |
# | `mes_id` (CSV) | Año 2026 incompleto | Último mes 202609 | Conservar, marcar `anio_completo`; comparaciones anuales usan 2000-2025 | Sumar 9 meses contra 12 sesgaría la tendencia a la baja |
# | `aeropuerto_oaci` (CSV) | Series con cobertura desigual | Sólo 17 de 69 aeropuertos tienen los 321 meses; SCTI cierra en 2006, SCTC→SCQP en 2014, SCHA→SCAT en 2005 | Conservar sin imputar; comparaciones entre aeropuertos en un año fijo (2025) | Son cambios reales de infraestructura; rellenar con ceros o promedios fabricaría datos |
# | `cnt_operaciones` (CSV) | Fuerte asimetría (skew 3,49) y concentración (2 aeropuertos = 41 %) | `describe()`, `skew()` | Conservar; usar escala logarítmica o agregación cuando corresponda | No es error: refleja la estructura real del sistema |
# | Filas (bitácora) | 1.122 duplicados exactos | `duplicated()`: misma matrícula, aeropuerto, tipo y minuto; 1.285 filas involucradas en SCVM 2004-2007 | Eliminar | Una aeronave no puede aterrizar dos veces en el mismo minuto: es duplicación del registro |
# | `tipo_operacion` | Valor `t` fuera del dominio A/D/W | 1 registro (SCSN, 16-01-2017) | Excluir | Ambiguo (¿despegue?); 1 de 11 M filas no altera resultados y asignarlo sería suponer |
# | `numero_vuelo`, `modelo_avion`, `modelo_avion_desc` | Ausentes ocultos como cadena vacía | 2.865.981 / 6 / 36.763 cadenas vacías además de 482.145 nulos | Homologar `""` → NaN | Un vacío es un ausente; tratarlo como categoría distorsiona conteos y cardinalidad |
# | `pmd` | Valores imposibles = 0 | 25.823 registros | Convertir a NaN (sin imputar) | Ninguna aeronave pesa 0 t |
# | `pmd` | Unidades mezcladas (kg en vez de t) | T34T = 1.340, C210 = 1.500, R44 = 1.000 (aeronaves de ~1-2 t); B38M = 710 | > 600 t → NaN excepto A225 (An-225, 640 t real) | Evita que 1.078 errores inflen la distribución; no se divide por 1.000 porque no hay certeza de la unidad en todos los casos |
# | `pmd` | Tipo `decimal128` | Esquema parquet; pandas lo leería como objetos `Decimal` | Convertir a `float64` | Permite operar numéricamente y reduce memoria |
# | `dt_operacion` | Documentación dice UTC, archivo trae zona `America/Santiago`; meses del CSV en hora local de cada aeropuerto | 41 registros caen en 1999 en hora local; diferencias con el CSV concentradas en SCIP y SCCI | Crear `dt_local` con la zona de cada aeropuerto | Es el mismo criterio de la JAC; se valida contra el CSV (sección 7.3) |
# | `dt_operacion` | Último mes incompleto en la bitácora | Último registro 30-09-2026 20:58 (= 23:58 UTC); el CSV de 202609 es algo mayor | No usar septiembre 2026 de la bitácora para conclusiones | La documentación advierte que t-1 es provisional |
# | `es_internacional` | Sin problemas | 0 nulos, booleano | Ninguna | — |

# %% [markdown]
# ## 7. Limpieza y preparación (Parte VIII)
#
# ### 7.1 Resumen mensual

# %%
mensual = prep.preparar_mensual(csv_raw)
mensual["anio_completo"] = mensual["anio"] <= 2025
cambios_codigo = (mensual["aeropuerto_oaci"] != csv_raw["aeropuerto_oaci"]).sum()
print("Códigos de aeropuerto modificados por normalización de cadenas:", cambios_codigo)
print("Total de operaciones antes/después:", csv_raw["cnt_operaciones"].sum(), mensual["n_operaciones"].sum())
mensual.head()

# %% [markdown]
# ### 7.2 Bitácora
#
# Cada paso informa cuántas filas o valores afecta, para que la limpieza sea auditable.

# %%
n0 = len(bit_raw)
bit = bit_raw.drop_duplicates().copy()
print(f"Duplicados exactos eliminados: {n0 - len(bit):,}")

n1 = len(bit)
bit = bit[bit["tipo_operacion"].isin(["A", "D", "W"])].copy()
bit["tipo_operacion"] = bit["tipo_operacion"].cat.remove_unused_categories()
print(f"Registros con tipo_operacion fuera de dominio excluidos: {n1 - len(bit):,}")

bit = prep.vacios_a_nulos(bit, ["numero_vuelo", "modelo_avion", "modelo_avion_desc"])
print("Nulos tras homologar vacíos:",
      bit[["numero_vuelo", "modelo_avion", "modelo_avion_desc"]].isna().sum().to_dict())

pmd_cero = bit["pmd"] == 0
pmd_implausible = (bit["pmd"] > prep.PMD_MAX_PLAUSIBLE_T) & (bit["modelo_avion"] != "A225")
bit.loc[pmd_cero | pmd_implausible, "pmd"] = np.nan
print(f"pmd = 0 → NaN: {pmd_cero.sum():,} | pmd implausible → NaN: {pmd_implausible.sum():,} | "
      f"pmd nulo final: {bit['pmd'].isna().sum():,} ({bit['pmd'].isna().mean():.1%})")

bit = prep.agregar_hora_local(bit)
n2 = len(bit)
bit = bit[bit["dt_local"] >= "2000-01-01"].copy()
print(f"Registros previos a 2000 en hora local excluidos: {n2 - len(bit):,}")
print(f"Filas finales: {len(bit):,} ({len(bit) / n0:.3%} del original)")

# %% [markdown]
# ### 7.3 Validación cruzada bitácora ↔ CSV
#
# Si la limpieza y la hora local están bien construidas, al agregar la bitácora (sin
# sobrevuelos) por aeropuerto × mes local × tipo de tráfico debería reproducirse el CSV publicado.

# %%
def agregar_como_csv(df):
    d = df[df["tipo_operacion"].isin(["A", "D"])]
    return (d.assign(mes_id=d["dt_local"].dt.year * 100 + d["dt_local"].dt.month,
                     internacional_domestico=np.where(d["es_internacional"], "I", "D"),
                     aeropuerto_oaci=d["aeropuerto_oaci"].astype(str))
            .groupby(["mes_id", "aeropuerto_oaci", "internacional_domestico"])
            .size().rename("n_bitacora").reset_index())


cruce = mensual.merge(agregar_como_csv(bit), how="left",
                      on=["mes_id", "aeropuerto_oaci", "internacional_domestico"])
coincide = cruce["n_operaciones"] == cruce["n_bitacora"]
print(f"Celdas aeropuerto-mes-tráfico idénticas: {coincide.mean():.2%} de {len(cruce):,}")
dif = cruce[~coincide].assign(diferencia=lambda x: x["n_bitacora"] - x["n_operaciones"])
print("Celdas distintas por mes:", dif["mes_id"].value_counts().head(5).to_dict())
print("Diferencia absoluta total (operaciones):", int(dif["diferencia"].abs().sum()),
      "de", int(mensual["n_operaciones"].sum()))

# Contraprueba: ¿la JAC deduplica? Se repite el cruce con la bitácora CON duplicados
con_dups = prep.agregar_hora_local(bit_raw[bit_raw["tipo_operacion"].isin(["A", "D"])])
cruce_dups = mensual.merge(agregar_como_csv(con_dups), how="left",
                           on=["mes_id", "aeropuerto_oaci", "internacional_domestico"])
print(f"Coincidencia usando la bitácora SIN eliminar duplicados: "
      f"{(cruce_dups['n_operaciones'] == cruce_dups['n_bitacora']).mean():.2%}")
del con_dups, cruce_dups

# %% [markdown]
# **Resultado.** La bitácora limpia reproduce el CSV en ~99 % de las celdas; las diferencias se
# concentran en septiembre 2026 (mes provisional y cortado en UTC) y en los meses con duplicados.
# La contraprueba muestra que el CSV oficial **sí incluye** los duplicados (coincide mejor sin
# eliminarlos). Por eso: (1) el análisis mensual usa el CSV tal como fue publicado, declarando que
# puede contener ~0,01 % de registros duplicados, y (2) el análisis a nivel de vuelo usa la
# bitácora deduplicada. Ver sección de ética.

# %% [markdown]
# ### 7.4 Justificación de transformaciones principales
#
# **Eliminación de duplicados (bitácora)**
# * *Variable:* fila completa. *Problema:* 1.122 registros idénticos en las 12 columnas.
# * *Evidencia:* `duplicated()`; casos como CCCYK aterrizando dos veces en SCFA el 01-01-2000 21:31.
# * *Alternativas:* conservarlos (como hace el CSV oficial) / eliminarlos / marcarlos.
# * *Decisión:* eliminar. *Justificación:* es físicamente imposible; son errores de captura.
# * *Efecto potencial:* −0,01 % de filas; afecta sobre todo a SCVM 2004-2007, sin impacto en tendencias nacionales.
#
# **`pmd` = 0 o > 600 t → NaN**
# * *Problema:* valores imposibles o en otra unidad. *Evidencia:* 25.823 ceros; T34T = 1.340 "t".
# * *Alternativas:* eliminar filas / dividir por 1.000 / imputar la mediana del modelo / NaN.
# * *Decisión:* NaN sin imputar. *Justificación:* eliminar filas borraría operaciones reales (que sí
#   cuentan para volúmenes); imputar o reescalar supone información que no tenemos.
# * *Efecto potencial:* los análisis de tamaño de aeronave excluyen ~2,7 % de operaciones sin PMD válido.
#
# **Hora local por aeropuerto (`dt_local`)**
# * *Problema:* el instante viene en una sola zona; Isla de Pascua y Magallanes tienen otra hora local.
# * *Alternativas:* usar `America/Santiago` para todo / usar UTC / zona por aeropuerto.
# * *Decisión:* zona por aeropuerto. *Justificación:* es el criterio documentado por la JAC y la
#   validación cruzada mejora al aplicarlo (sin él, las diferencias se concentran en SCIP y SCCI).
# * *Efecto potencial:* cambia la hora del día y el mes asignado a operaciones cercanas a medianoche.
#
# **Homologación de cadenas vacías → NaN**
# * *Problema:* 2,87 M de `numero_vuelo` vacíos que no aparecen como nulos.
# * *Decisión:* convertir a NaN. *Justificación:* un vacío no es un número de vuelo; el perfil de
#   nulos subestimaba el problema (4 % declarado vs. 30 % real). *Efecto:* ninguno en conteos de operaciones.

# %% [markdown]
# ### 7.5 Datasets procesados
#
# Se guardan versiones nuevas en `data/processed/`; los archivos de `data/raw/` no se tocan.

# %%
mensual.to_csv(PROCESSED / "operaciones_mensuales_procesado.csv", index=False)
print("Guardado: data/processed/operaciones_mensuales_procesado.csv", mensual.shape)
