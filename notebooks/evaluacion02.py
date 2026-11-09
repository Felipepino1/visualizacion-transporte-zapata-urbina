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
# **Autores:**
# * Felipe Ignacio Zapata Lagos
# * Agustín Alejandro Urbina López
#
# Ingeniería en Informática · INACAP  
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

# %% [markdown]
# ## 8. Ingeniería de variables (Parte IX)
#
# | Variable derivada | Origen | Transformación | Significado | Utilidad analítica |
# |---|---|---|---|---|
# | `fecha`, `anio`, `mes` | `mes_id` (CSV) | `to_datetime(format="%Y%m")` → `.dt.year`, `.dt.month` | Periodo de la operación | Jerarquía temporal Año → Mes; series y estacionalidad (P2) |
# | `tipo_trafico` | `internacional_domestico` / `es_internacional` | Mapeo D/I → "Doméstico"/"Internacional" | Ámbito del vuelo | Segmentación legible en leyendas (P1, P2) |
# | `indice_2019` | `n_operaciones` anual por tráfico | `valor / valor_2019 × 100` | Nivel relativo al último año pre-pandemia | Compara recuperación de series de escala muy distinta en **un solo eje** (P2, H1) |
# | `participacion_internacional` | `n_operaciones` por aeropuerto | `internacional / total` | Proporción internacional | Caracteriza el rol de cada aeropuerto (P1) |
# | `dt_local`, `hora`, `dia_semana`, `fin_de_semana` | `dt_operacion` + zona por aeropuerto | Conversión de zona; `.dt.hour`, `.dt.dayofweek` | Momento local de la operación | Patrón diario y semanal (P3) |
# | `categoria_peso` | `pmd` | `pd.cut` en 5,7 t (umbral OACI de aeronave ligera) y 50 t | Tamaño de aeronave | Separa aviación general/instrucción de la comercial (P3) |

# %%
# Variables a nivel de vuelo: se trabaja con 2025, último año completo
b25 = bit[bit["dt_local"].dt.year == 2025].copy()
b25["hora"] = b25["dt_local"].dt.hour
b25["dia_semana"] = pd.Categorical(b25["dt_local"].dt.dayofweek.map(dict(enumerate(prep.DIAS))),
                                   categories=prep.DIAS, ordered=True)
b25["fin_de_semana"] = b25["dt_local"].dt.dayofweek >= 5
b25["categoria_peso"] = prep.categoria_peso(b25["pmd"])
b25["tipo_trafico"] = np.where(b25["es_internacional"], "Internacional", "Doméstico")
for c in b25.select_dtypes("category"):
    b25[c] = b25[c].cat.remove_unused_categories()
print(f"Operaciones 2025: {len(b25):,}")
b25[["dt_local", "hora", "dia_semana", "fin_de_semana", "pmd", "categoria_peso", "tipo_trafico"]].head()

# %%
b25["categoria_peso"].value_counts(dropna=False)

# %%
cols_guardar = ["dt_operacion", "dt_local", "aeropuerto_oaci", "tipo_operacion", "aeropuerto_dgac_orig_dest",
                "aerolinea_dgac", "actividad_cod", "modelo_avion", "es_internacional", "pmd",
                "hora", "dia_semana", "fin_de_semana", "categoria_peso", "tipo_trafico"]
b25[cols_guardar].to_parquet(PROCESSED / "bitacora_2025_procesada.parquet", index=False, compression="zstd")
print("Guardado: data/processed/bitacora_2025_procesada.parquet",
      f"({(PROCESSED / 'bitacora_2025_procesada.parquet').stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 9. Dimensionalidad (Parte X)
#
# | Variable | Tipo | Dimensión / Medida / ID | Cardinalidad | Función en el análisis |
# |---|---|---|---|---|
# | `mes_id` / `fecha` | Temporal | Dimensión | 321 meses | Eje temporal de P2 |
# | `anio` | Temporal (numérica discreta) | Dimensión | 27 | Nivel superior de la jerarquía temporal |
# | `mes` | Categórica ordinal | Dimensión | 12 | Estacionalidad (visualización multidimensional) |
# | `aeropuerto_oaci` | Categórica nominal | Dimensión | 69 | Comparación entre aeropuertos (P1) |
# | `internacional_domestico` / `tipo_trafico` | Categórica nominal (binaria) | Dimensión | 2 | Segmentación (P1, P2) |
# | `n_operaciones` | Numérica discreta | **Medida** (aditiva) | 2.837 | Volumen de actividad |
# | `indice_2019` | Numérica continua | Medida derivada (no aditiva) | — | Recuperación relativa (P2) |
# | `participacion_internacional` | Numérica continua (0-1) | Medida derivada (no aditiva) | — | Rol del aeropuerto (P1) |
# | `dt_operacion` / `dt_local` | Temporal | Dimensión (granularidad minuto) | 6,5 M | Base de hora y día |
# | `hora` | Numérica discreta / ordinal | Dimensión | 24 | Patrón diario (P3) |
# | `dia_semana` | Categórica ordinal | Dimensión | 7 | Patrón semanal (P3) |
# | `tipo_operacion` | Categórica nominal | Dimensión | 3 (A/D/W) | Filtro (sobrevuelos fuera) |
# | `pmd` | Numérica continua (t) | Medida (no aditiva) | 785 | Distribución de tamaño (P3) |
# | `categoria_peso` | Categórica ordinal | Dimensión derivada | 3 | Segmentación por tamaño (P3) |
# | `actividad_cod` | Categórica nominal | Dimensión | 26 | Contexto (motivo del vuelo); sin diccionario público de códigos |
# | `aerolinea_dgac` | Categórica nominal | Dimensión (alta cardinalidad) | 2.568 | No se grafica directamente: códigos internos DGAC |
# | `modelo_avion` | Categórica nominal | Dimensión (alta cardinalidad) | 1.286 | Contexto para validar `pmd` |
# | `aeropuerto_dgac_orig_dest` | Categórica nominal / espacial (código) | Dimensión | 1.671 | Origen/destino (no usado: sin coordenadas) |
# | `matricula` | Textual | **Identificador** de aeronave | 13.835 | Detección de duplicados |
# | `numero_vuelo` | Textual | Identificador (parcial, 30 % ausente) | 7.442 | No usado |
# | `es_internacional` | Booleana | Dimensión | 2 | Origen de `tipo_trafico` |
#
# La medida principal es **el conteo de operaciones** (cada fila de la bitácora = 1 operación;
# en el CSV ya viene sumada en `n_operaciones`).

# %% [markdown]
# ### 9.1 Jerarquías
#
# * **Temporal (real):** Año → Mes → Día → Hora. Año y mes existen en ambos archivos; día y hora
#   sólo en la bitácora.
# * **Territorial: no existe en los datos.** Hay códigos OACI, pero ninguna columna de región,
#   ciudad ni coordenadas. Agrupar por región exigiría una tabla externa; no se inventa ese nivel.
# * **Aeronave (parcial):** `modelo_avion` → `categoria_peso` es una agrupación derivada, no una
#   jerarquía publicada por la fuente.

# %% [markdown]
# ## 10. Transformaciones analíticas y agregación (Parte XI)

# %%
# Agregación SIMPLE: 1 dimensión (aeropuerto) + 1 medida (operaciones), año 2025
ag_simple = (mensual.query("anio == 2025")
             .groupby("aeropuerto_oaci")["n_operaciones"].sum()
             .sort_values(ascending=False))
print(f"Aeropuertos con operaciones en 2025: {len(ag_simple)} | total: {ag_simple.sum():,}")
ag_simple.head(10)

# %% [markdown]
# * **Representa:** total de despegues + aterrizajes de cada aeropuerto en 2025.
# * **Granularidad:** aeropuerto-año (se suman 12 meses × 2 tipos de tráfico).
# * **Se pierde:** la variación mensual y la distinción doméstico/internacional.
# * **Visualización apropiada:** barras horizontales ordenadas (comparación de magnitudes entre categorías nominales).

# %%
# Agregación MULTIDIMENSIONAL: 2 dimensiones (año, tipo de tráfico) + 1 medida
ag_multi = mensual.pivot_table(index="anio", columns="tipo_trafico", values="n_operaciones",
                               aggfunc="sum")
indice = (ag_multi / ag_multi.loc[2019] * 100).round(1)
ag_multi.join(indice, rsuffix=" (índice 2019=100)").loc[[2000, 2010, 2019, 2020, 2021, 2022, 2023, 2024, 2025]]

# %% [markdown]
# * **Representa:** operaciones anuales por tipo de tráfico y su nivel relativo a 2019.
# * **Granularidad:** año × tipo de tráfico.
# * **Se pierde:** el detalle por aeropuerto y por mes (la estacionalidad y la caída abrupta de abril 2020 quedan promediadas en el año).
# * **Visualización apropiada:** líneas (evolución temporal de dos series) sobre el índice, para compartir un eje.

# %%
# Agregación MULTIDIMENSIONAL (bitácora): hora × categoría de peso + conteo, sólo aterrizajes y despegues
ag_hora = (b25.query("tipo_operacion in ['A', 'D'] and categoria_peso.notna()", engine="python")
           .groupby(["categoria_peso", "hora"], observed=True).size()
           .rename("operaciones").reset_index())
ag_hora["participacion_en_categoria"] = (ag_hora["operaciones"]
                                         / ag_hora.groupby("categoria_peso", observed=True)["operaciones"].transform("sum"))
ag_hora.pivot_table(index="hora", columns="categoria_peso", values="participacion_en_categoria",
                    observed=True).style.format("{:.1%}").background_gradient(cmap="Blues")

# %% [markdown]
# * **Granularidad:** categoría de peso × hora local (2025). **Se pierde:** aeropuerto, día y tipo de tráfico.
# * Se normaliza dentro de cada categoría (participación) porque los volúmenes absolutos difieren mucho;
#   así se compara la **forma** del día, no el tamaño.

# %% [markdown]
# ## 11. Visualizaciones
#
# ### 11.1 Matriz de diseño (Parte XII)
#
# | Pregunta | Dimensiones | Medidas | Transformación | Representación elegida | Justificación |
# |---|---|---|---|---|---|
# | **P1** ¿Qué aeropuertos concentran las operaciones y qué peso tiene el tráfico internacional? | Aeropuerto (nominal), tipo de tráfico (binaria) | Operaciones 2025 | `query` 2025 → `pivot_table` aeropuerto × tráfico → `sort_values` → top 15 | **Barras horizontales apiladas** (Matplotlib) | Comparar magnitudes entre categorías nominales: la longitud sobre una base común es el canal más preciso; horizontales para leer códigos; apilado muestra total y composición |
# | **P2** ¿Cómo evolucionaron doméstico e internacional y cómo se recuperó cada uno tras 2020? | Año (temporal), tipo de tráfico | Operaciones anuales → índice 2019 = 100 | `groupby` año × tráfico → división por 2019 | **Líneas** (Matplotlib) + benchmark con heatmap | Evolución temporal continua; el índice pone dos series de escala 5:1 en un solo eje sin usar doble eje |
# | **P3a** ¿Cómo se distribuye el tamaño de las aeronaves que operan? | — | PMD (t) por operación 2025 | Filtro A/D con PMD válido | **Histograma en escala log** (Seaborn) | Distribución de una variable continua muy asimétrica (0,5 t a 640 t) |
# | **P3b** ¿Difiere el patrón horario según tamaño de aeronave? | Hora local, categoría de peso | % de operaciones de la categoría en cada hora | `groupby` categoría × hora → participación | **Líneas por segmento** (Seaborn) | Relación hora-actividad segmentada; normalizar compara la forma del día |
# | **Multidim.** ¿Cambió la estacionalidad mensual entre años y tipos de tráfico? | Mes, año, tipo de tráfico | Operaciones mensuales | `groupby` año × mes × tráfico | **Líneas con color = año y faceta = tráfico** (Seaborn `relplot`) | 4 variables: x = mes, y = operaciones, color = año, faceta = tráfico |

# %% [markdown]
# ### 11.2 Matplotlib — Visualización 1: comparación entre aeropuertos (P1)

# %%
top_n = 15
p1 = (mensual.query("anio == 2025")
      .pivot_table(index="aeropuerto_oaci", columns="tipo_trafico", values="n_operaciones",
                   aggfunc="sum", fill_value=0))
p1["Total"] = p1.sum(axis=1)
p1["participacion_internacional"] = p1["Internacional"] / p1["Total"]
p1 = p1.sort_values("Total", ascending=False)
otros = p1.iloc[top_n:]
p1_top = p1.head(top_n).iloc[::-1]          # invertido: el mayor queda arriba
print(f"Top {top_n} = {p1.head(top_n)['Total'].sum() / p1['Total'].sum():.1%} del total 2025; "
      f"otros {len(otros)} aeropuertos = {otros['Total'].sum():,} operaciones")
p1.head(top_n)

# %%
fig, ax = plt.subplots(figsize=(9, 6.2))
y = np.arange(len(p1_top))
alto = 0.68
ax.barh(y, p1_top["Doméstico"], height=alto, color=estilo.AZUL, label="Doméstico",
        edgecolor=estilo.SUPERFICIE, linewidth=1)
ax.barh(y, p1_top["Internacional"], left=p1_top["Doméstico"], height=alto, color=estilo.NARANJO,
        label="Internacional", edgecolor=estilo.SUPERFICIE, linewidth=1)
for yi, (tot, part) in enumerate(zip(p1_top["Total"], p1_top["participacion_internacional"])):
    etiqueta = estilo.miles(tot) + (f"  ({part:.0%} intl.)" if part >= 0.01 else "")
    ax.text(tot + p1_top["Total"].max() * 0.01, yi, etiqueta, va="center", fontsize=9, color=estilo.TINTA_2)
ax.set_yticks(y, p1_top.index)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(estilo.miles))
ax.set_xlim(0, p1_top["Total"].max() * 1.22)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Operaciones en 2025 (aterrizajes + despegues)")
ax.set_title("SCEL concentra un tercio de las operaciones del país y casi todo el tráfico internacional")
estilo.subtitulo(ax, f"Top {top_n} de {len(p1)} aeropuertos con operaciones en 2025 · código OACI")
ax.legend(loc="lower right", ncols=2)
estilo.nota_fuente(fig, "Fuente: JAC/DGAC, «Operaciones en Aeropuertos», datos.gob.cl (CC0). "
                   "Excluye operaciones militares/policiales y aeródromos no administrados por la DGAC.")
estilo.guardar(fig, "v1_matplotlib_comparacion_aeropuertos_2025", FIGURAS)

# %% [markdown]
# **Decisiones de diseño.** Escala lineal desde cero (las barras codifican magnitud por longitud;
# truncar o usar log distorsionaría la comparación). Orden descendente para que el ranking se lea
# sin esfuerzo. Color sólo para el tipo de tráfico (azul = doméstico, naranjo = internacional, igual
# en todas las figuras). Total y % internacional como etiqueta directa, porque los segmentos
# naranjos de los aeropuertos regionales son demasiado pequeños para leerse por longitud.

# %% [markdown]
# ### 11.3 Matplotlib — Visualización 2: evolución y recuperación (P2)

# %%
p2 = (mensual.query("anio_completo")
      .groupby(["anio", "tipo_trafico"])["n_operaciones"].sum()
      .unstack("tipo_trafico"))
p2_indice = p2 / p2.loc[2019] * 100
anio_recupera = {c: int(p2_indice.loc[2020:, c][p2_indice.loc[2020:, c] >= 100].index.min())
                 for c in p2_indice.columns}
print("Primer año en volver al nivel 2019:", anio_recupera)
p2.join(p2_indice.round(1), rsuffix=" índice").tail(8)

# %%
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.axvspan(2019.5, 2021.5, color=estilo.GRILLA, alpha=0.6, lw=0)
ax.text(2020.5, 128, "Pandemia\n2020-2021", ha="center", va="top", fontsize=9, color=estilo.TINTA_2)
ax.axhline(100, color=estilo.TINTA_3, lw=1, ls="--")
ax.text(2000, 101.5, "Nivel 2019 = 100", fontsize=8.5, color=estilo.TINTA_3, va="bottom")
for serie, color in estilo.COLOR_TRAFICO.items():
    ax.plot(p2_indice.index, p2_indice[serie], color=color, lw=2.2, marker="o", markersize=4.5,
            markeredgecolor=estilo.SUPERFICIE, markeredgewidth=1.2)
    ultimo = p2_indice[serie].iloc[-1]
    ax.text(2025.35, ultimo, f"{serie}\n{ultimo:.0f}", color=estilo.TINTA, fontsize=9.5,
            va="center", fontweight="bold")
    minimo = p2_indice.loc[2020:2021, serie]
    ax.annotate(f"{minimo.min():.0f}", (minimo.idxmin(), minimo.min()), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9, color=estilo.TINTA_2)
ax.set_xlim(1999.5, 2027)
ax.set_ylim(30, 130)
ax.set_xticks(range(2000, 2026, 5))
ax.set_ylabel("Índice de operaciones (2019 = 100)")
ax.set_title("El tráfico internacional cayó más y tardó un año más que el doméstico en recuperarse")
estilo.subtitulo(ax, "Operaciones anuales (aterrizajes + despegues), índice base 2019 = 100 · 2000-2025 (años completos)")
estilo.nota_fuente(fig, f"Fuente: JAC/DGAC, datos.gob.cl. Volumen 2019: doméstico {estilo.miles(p2.loc[2019, 'Doméstico'])}, "
                   f"internacional {estilo.miles(p2.loc[2019, 'Internacional'])} operaciones.")
estilo.guardar(fig, "v2_matplotlib_evolucion_indice_2019", FIGURAS)

# %% [markdown]
# **Por qué un índice y no los valores absolutos.** El tráfico doméstico es ~5 veces el internacional;
# en un mismo eje absoluto la línea internacional quedaría aplastada contra el piso y su caída sería
# ilegible. Usar dos ejes Y (uno por serie) haría que el cruce de las líneas dependiera de cómo se
# eligen las escalas. El índice base 2019 responde exactamente la pregunta (¿cuánto se alejó cada
# serie de su nivel pre-pandemia?) en un solo eje. Los volúmenes absolutos se declaran en la nota.

# %% [markdown]
# ### 11.4 Seaborn — Visualización 3: distribución del tamaño de aeronave (P3a)

# %%
ops25 = b25[b25["tipo_operacion"].isin(["A", "D"])]
pmd25 = ops25["pmd"].dropna()
print(f"Operaciones 2025 (A/D): {len(ops25):,} | con PMD válido: {len(pmd25):,} ({len(pmd25) / len(ops25):.1%})")
print(pmd25.describe(percentiles=[.1, .25, .5, .75, .9]).round(2))
print("Asimetría:", round(pmd25.skew(), 2), "| asimetría de log10(PMD):", round(np.log10(pmd25).skew(), 2))
print(ops25["categoria_peso"].value_counts(normalize=True, dropna=False).round(3))

# %%
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(x=pmd25, log_scale=True, bins=60, stat="percent", color=estilo.AZUL,
             edgecolor=estilo.SUPERFICIE, linewidth=0.6, ax=ax)
for umbral, texto in [(prep.UMBRAL_LIGERA_T, "5,7 t\nlímite OACI\naeronave ligera"), (prep.UMBRAL_PESADA_T, "50 t")]:
    ax.axvline(umbral, color=estilo.TINTA_2, lw=1, ls="--")
    ax.text(umbral * (1.08 if umbral < 10 else 0.93), ax.get_ylim()[1] * 0.97, texto, fontsize=8.5,
            color=estilo.TINTA_2, va="top", ha="left" if umbral < 10 else "right")
ax.axvline(pmd25.median(), color=estilo.NARANJO, lw=1.5)
ax.text(pmd25.median() * 1.25, ax.get_ylim()[1] * 0.80, f"mediana = {pmd25.median():.0f} t\n(no representa a\nninguna de las dos\npoblaciones)",
        fontsize=8.5, color=estilo.TINTA, ha="left", va="top")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.set_xlabel("Peso máximo de despegue, PMD (toneladas, escala logarítmica)")
ax.set_ylabel("% de operaciones")
ax.grid(axis="x", visible=False)
ax.set_title("Dos poblaciones: avionetas de ~1 t y aviones comerciales de 70-90 t")
estilo.subtitulo(ax, f"Distribución del PMD por operación, aterrizajes y despegues 2025 · n = {estilo.miles(len(pmd25))}")
estilo.nota_fuente(fig, "Fuente: JAC/DGAC, «Bitácora de Vuelos», datos.gob.cl. Excluye operaciones sin PMD válido "
                   f"({estilo.miles(len(ops25) - len(pmd25))} registros con PMD nulo, 0 o implausible). Línea naranja: mediana.")
estilo.guardar(fig, "v3_seaborn_distribucion_pmd", FIGURAS)

# %% [markdown]
# **Análisis de distribución**
#
# * **¿Dónde se concentra?** En dos modos: aeronaves livianas de 0,7-2 t (instrucción, aviación general)
#   y aviones de fuselaje angosto de 70-90 t (A320/A321/B737), que dominan el tráfico comercial.
# * **¿Asimetría?** Fuerte asimetría positiva en escala lineal; en escala logarítmica la distribución es
#   **bimodal**, por lo que media y mediana no representan a ninguna de las dos poblaciones.
# * **¿Dispersión?** Muy alta: más de tres órdenes de magnitud (≈0,5 t a 640 t).
# * **¿Valores extremos?** Sí: una cola derecha legítima de aviones de fuselaje ancho (≈200-450 t en 2025)
#   y, en la serie completa, los valores ya tratados como errores (0 t y > 600 t en avionetas).
# * **Implicación:** cualquier análisis de "tamaño promedio" debe segmentarse; por eso se crea
#   `categoria_peso` y P3b compara segmentos en vez de promedios.

# %% [markdown]
# ### 11.5 Seaborn — Visualización 4: patrón horario segmentado por tamaño (P3b)

# %%
p3 = ag_hora.copy()
p3["porcentaje"] = p3["participacion_en_categoria"] * 100
pico = p3.loc[p3.groupby("categoria_peso", observed=True)["porcentaje"].idxmax(), ["categoria_peso", "hora", "porcentaje"]]
noche = (p3.query("hora <= 5").groupby("categoria_peso", observed=True)["porcentaje"].sum().round(1))
print("Hora de mayor actividad por categoría:\n", pico.to_string(index=False))
print("\n% de operaciones entre 00:00 y 05:59:\n", noche.to_string())

# %%
paleta_peso = dict(zip(p3["categoria_peso"].cat.categories, ["#86b6ef", "#2a78d6", "#104281"]))
fig, ax = plt.subplots(figsize=(10, 5.2))
sns.lineplot(data=p3, x="hora", y="porcentaje", hue="categoria_peso", palette=paleta_peso,
             lw=2.2, marker="o", markersize=5, markeredgecolor=estilo.SUPERFICIE, ax=ax, legend=True)
ax.axvspan(-0.5, 5.5, color=estilo.GRILLA, alpha=0.5, lw=0)
ax.text(2.5, ax.get_ylim()[1] * 0.95, "madrugada", ha="center", fontsize=9, color=estilo.TINTA_2, va="top")
ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlim(-0.5, 23.5)
ax.set_xlabel("Hora local del aeropuerto")
ax.set_ylabel("% de las operaciones de la categoría")
ax.legend(title="Categoría por PMD", loc="upper left", bbox_to_anchor=(0.02, 0.88))
ax.set_title("Las avionetas vuelan de día; la aviación pesada opera las 24 horas")
estilo.subtitulo(ax, "Distribución horaria de aterrizajes y despegues 2025 dentro de cada categoría de tamaño (cada línea suma 100 %)")
estilo.nota_fuente(fig)
estilo.guardar(fig, "v4_seaborn_patron_horario_por_tamano", FIGURAS)

# %% [markdown]
# **Lectura.** Se normaliza dentro de cada categoría porque las tres tienen volúmenes distintos; así
# se compara la *forma* del día. La categoría ligera concentra su actividad entre media mañana y la
# tarde (vuelo visual diurno, instrucción), mientras que la pesada mantiene actividad relevante de
# madrugada (vuelos internacionales y de carga). La rampa de un solo tono (claro → oscuro) refleja
# que la categoría es **ordinal**.

# %% [markdown]
# ### 11.6 Visualización multidimensional (Parte XV) — estacionalidad por año y tipo de tráfico

# %%
anios_sel = [2019, 2020, 2022, 2025]
p5 = (mensual.query("anio in @anios_sel")
      .groupby(["tipo_trafico", "anio", "mes"])["n_operaciones"].sum().reset_index())
p5["miles_operaciones"] = p5["n_operaciones"] / 1000
p5["anio"] = p5["anio"].astype(str)
p5.pivot_table(index="mes", columns=["tipo_trafico", "anio"], values="n_operaciones").head(4)

# %%
colores_anio = dict(zip(map(str, anios_sel), [estilo.AZUL, estilo.NARANJO, estilo.AQUA, estilo.AMARILLO]))
g = sns.relplot(data=p5, x="mes", y="miles_operaciones", hue="anio", col="tipo_trafico",
                kind="line", palette=colores_anio, lw=2.2, marker="o", markersize=5,
                markeredgecolor=estilo.SUPERFICIE, height=4.4, aspect=1.2,
                facet_kws={"sharey": False}, legend=True)
for ax, (trafico, d) in zip(g.axes.flat, p5.groupby("tipo_trafico")):
    # etiquetas directas al final de cada línea, separadas para que no se superpongan
    finales = (d[d["mes"] == 12].set_index("anio")["miles_operaciones"].sort_values())
    sep = finales.max() * 0.07
    posiciones, previo = {}, -np.inf
    for anio, y_fin in finales.items():
        previo = max(y_fin, previo + sep)
        posiciones[anio] = previo
    for anio, y_pos in posiciones.items():
        ax.text(12.3, y_pos, anio, color=estilo.TINTA, fontsize=9, va="center", fontweight="bold")
    ax.set_title(trafico, loc="left", fontsize=11.5)
    ax.set_xticks(range(1, 13), [m[0] for m in prep.MESES])
    ax.set_xlim(0.6, 13.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("Mes")
g.axes.flat[0].set_ylabel("Operaciones mensuales (miles)")
g.legend.set_title("Año")
g.figure.suptitle("Antes y después de la pandemia: la estacionalidad doméstica se mantuvo; "
                  "la internacional se desplazó hacia arriba", x=0.01, ha="left", fontweight="bold", y=1.14)
g.figure.text(0.01, 1.055, "Operaciones mensuales por tipo de tráfico · ejes Y independientes (el doméstico es ~5 veces mayor)",
              fontsize=9.5, color=estilo.TINTA_2)
estilo.nota_fuente(g.figure)
estilo.guardar(g.figure, "v5_multidimensional_estacionalidad", FIGURAS)

# %% [markdown]
# **Justificación de canales visuales**
#
# * `mes` → **posición X** → variable ordinal/cíclica; la posición es el canal más preciso y conserva el orden.
# * `operaciones` (miles) → **posición Y** → la medida cuantitativa debe ir en el canal más preciso.
# * `año` → **color (tono)** → 4 categorías distinguibles; paleta validada para daltonismo y etiquetas
#   directas al final de cada línea para no depender sólo del color.
# * `tipo de tráfico` → **faceta (columna)** → separa dos series de escala 5:1; con ejes Y independientes
#   cada panel muestra su propia forma (se advierte explícitamente en el subtítulo).
#
# **¿Qué ocurriría con una dimensión adicional?** Por ejemplo, agregar el aeropuerto como forma del
# marcador o tamaño: 4 años × 2 tráficos × N aeropuertos produciría decenas de líneas superpuestas;
# la forma del marcador es poco discriminable con más de 4-5 valores y el tamaño no se lee con precisión.
# La carga cognitiva aumentaría (el lector debería cruzar color + forma + panel) y la legibilidad caería.
# La alternativa correcta sería **más facetas** (una fila por aeropuerto principal) o un gráfico distinto,
# no más canales sobre el mismo panel.

# %% [markdown]
# ## 12. Benchmark visual (Parte XVI)
#
# **Pregunta P2:** ¿cómo evolucionó la actividad y cómo se recuperó tras 2020?
#
# * **Visualización A:** líneas con índice 2019 = 100 por tipo de tráfico (Visualización 2).
# * **Visualización B:** mapa de calor año × mes con el total mensual de operaciones (valores absolutos).

# %%
p_b = mensual.query("anio_completo").pivot_table(index="anio", columns="mes", values="n_operaciones", aggfunc="sum")
p_b.columns = prep.MESES
fig, ax = plt.subplots(figsize=(9.5, 8))
sns.heatmap(p_b / 1000, cmap=sns.blend_palette(estilo.RAMPA_AZUL, as_cmap=True), linewidths=1,
            linecolor=estilo.SUPERFICIE, cbar_kws={"label": "Operaciones mensuales (miles)", "shrink": 0.6}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(length=0)
ax.set_title("Benchmark B — Operaciones mensuales totales, 2000-2025")
estilo.subtitulo(ax, "Doméstico + internacional · color = volumen (miles de operaciones)")
estilo.nota_fuente(fig)
estilo.guardar(fig, "v6_benchmark_heatmap_anio_mes", FIGURAS)

# %% [markdown]
# | Criterio | Visualización A (líneas, índice 2019) | Visualización B (heatmap año × mes) |
# |---|---|---|
# | **Claridad** | Alta: dos líneas, un eje, mensaje directo | Media: el patrón general se ve, pero hay que recorrer 312 celdas |
# | **Precisión** | Alta: posición sobre eje común (el canal más preciso) | Baja: el color permite comparar "más o menos", no estimar valores |
# | **Comparabilidad** | Directa entre doméstico e internacional (misma base) | No distingue tipos de tráfico (los suma); compara bien meses entre sí |
# | **Densidad informativa** | Baja-media: 52 puntos anuales | Alta: 312 valores mensuales, muestra estacionalidad y el hueco de abril 2020 |
# | **Carga cognitiva** | Baja | Media-alta: requiere leer la escala de color |
# | **Utilidad para P2** | Responde la pregunta y permite evaluar H1 | Útil como exploración complementaria; no permite evaluar H1 |
#
# **¿Cuál debería utilizarse y por qué?** La **visualización A**. P2 y la hipótesis H1 piden comparar
# *dos series* respecto de *su propio nivel previo*, y eso requiere posición sobre un eje común,
# que es justamente lo que el heatmap no ofrece (codifica la magnitud en color y suma ambos tráficos).
# El heatmap aporta algo que A pierde —la estacionalidad mensual y el momento exacto del desplome
# (abril 2020)— por lo que es un buen complemento exploratorio, pero no la visualización principal.

# %% [markdown]
# ## 13. Validación de las visualizaciones definitivas (Parte XVII)
#
# Se auditan las dos visualizaciones más importantes: **V1 (comparación entre aeropuertos)** y
# **V2 (evolución e índice 2019)**. Primero se verifican por código las cifras que muestran.

# %%
# V1: los totales graficados coinciden con la suma directa del CSV original
check_v1 = csv_raw[csv_raw["mes_id"] // 100 == 2025].groupby("aeropuerto_oaci")["cnt_operaciones"].sum()
assert (check_v1.loc[p1_top.index] == p1_top["Total"]).all()
assert p1["Total"].sum() == check_v1.sum()
# V2: el índice 2019 vale exactamente 100 y los niveles anuales suman el total del CSV para 2000-2025
assert np.allclose(p2_indice.loc[2019], 100)
assert p2.to_numpy().sum() == csv_raw.loc[csv_raw["mes_id"] < 202601, "cnt_operaciones"].sum()
print("Validación numérica OK: V1 y V2 reproducen exactamente los totales del archivo original.")
print(f"SCEL: {p1.loc['SCEL', 'Total'] / p1['Total'].sum():.1%} del total 2025 y "
      f"{p1.loc['SCEL', 'Internacional'] / p1['Internacional'].sum():.1%} del tráfico internacional; "
      f"top 15 = {p1.head(15)['Total'].sum() / p1['Total'].sum():.1%}")

# %% [markdown]
# | Aspecto | V1 — Barras apiladas por aeropuerto 2025 | V2 — Líneas, índice 2019 = 100 |
# |---|---|---|
# | **Propósito** | Mostrar concentración y rol internacional de cada aeropuerto | Comparar caída y recuperación de dos tipos de tráfico |
# | **Audiencia** | Planificación aeroportuaria / autoridad (JAC, DGAC), público general informado | Igual; especialmente quien evalúa recuperación post-pandemia |
# | **Selección de datos** | 2025 (último año completo); top 15 de 54 (83,7 % del total), el resto se declara en el subtítulo | 2000-2025; se excluye 2026 por estar incompleto (9 meses) |
# | **Agregación** | Suma anual por aeropuerto × tráfico | Suma anual por tráfico; se pierde la estacionalidad (cubierta por V5 y el benchmark) |
# | **Escalas** | Lineal desde 0 (obligatorio en barras) | Índice; eje Y desde 30, *no* desde 0: es un gráfico de líneas sobre un índice y el 0 no es una referencia relevante; la línea base es 100 y está marcada |
# | **Color** | 2 colores con significado (tráfico), consistentes en todo el proyecto, validados para daltonismo | Mismos 2 colores; etiquetas directas, por lo que no depende sólo del color |
# | **Dimensiones** | Aeropuerto (Y), operaciones (X), tráfico (color) | Año (X), índice (Y), tráfico (color) |
# | **Legibilidad** | Etiquetas de total y % internacional; códigos OACI legibles en horizontal | Mínimos 2020-2021 anotados; banda de pandemia sombreada |
# | **Integridad** | Totales verificados contra el CSV (assert) | Índice y sumas verificados (assert); volúmenes 2019 declarados en la nota |
# | **Posibles sesgos** | El top 15 oculta 39 aeródromos pequeños; los códigos OACI no dicen nada a un público general | El índice hace ver al internacional "creciendo más" aunque en volumen absoluto es ~5 veces menor |
# | **Interpretación errónea posible** | Leer SCEL como "aeropuerto de Santiago" completo: SCTB y otros también están en la RM | Atribuir la diferencia de recuperación a una causa concreta (cierre de fronteras, aerolíneas) que los datos no contienen |
# | **Limitaciones** | Excluye vuelos militares/policiales y aeródromos no DGAC; no hay pasajeros ni carga, sólo operaciones | Las operaciones no miden pasajeros: un mismo número de vuelos puede transportar distinto volumen |

# %% [markdown]
# ## 14. Interpretación (Parte XVIII)
#
# **HALLAZGO 1 — La actividad está muy concentrada y el tráfico internacional casi no se descentraliza.**
# * *Datos utilizados:* CSV procesado, año 2025, 54 aeropuertos.
# * *Evidencia:* SCEL registra 176.740 operaciones (33,7 % del total nacional) y 80.362 internacionales,
#   el 97,3 % de todo el tráfico internacional del país. Ningún otro aeropuerto supera el 4 % internacional.
# * *Visualización:* V1.
# * *Interpretación:* el sistema funciona con un único punto de entrada internacional relevante.
# * *Posible utilidad:* priorización de capacidad, planes de contingencia (una interrupción en SCEL afecta casi todo el tráfico internacional).
# * *Limitación:* se cuentan operaciones, no pasajeros ni carga; no se observan vuelos militares ni aeródromos sin DGAC.
#
# **HALLAZGO 2 — La recuperación post-2020 fue asimétrica: el internacional cayó más, tocó fondo un año después y luego superó su nivel previo.**
# * *Datos utilizados:* CSV procesado, 2000-2025, agregado anual por tipo de tráfico.
# * *Evidencia:* índice 2019 = 100 → doméstico mínimo 57,8 (2020) y vuelve a ≥ 100 en 2023; internacional
#   mínimo 37,5 (2021, un año después) y vuelve a ≥ 100 en 2024. En 2025: doméstico 102, internacional 112.
# * *Visualización:* V2 (y V5 para el detalle mensual: abril 2020 fue el mes más bajo en ambos).
# * *Interpretación:* **H1 es consistente con los datos** (descriptivamente): el internacional tardó un año
#   más en recuperarse. Los datos no permiten afirmar *por qué* (restricciones sanitarias, demanda, oferta de
#   aerolíneas); sólo describen el patrón.
# * *Posible utilidad:* monitorear si el crecimiento internacional sobre 2019 se sostiene en 2026 para planificar capacidad en SCEL.
# * *Limitación:* nivel anual; 2026 incompleto; no hay variables explicativas en el dataset.
#
# **HALLAZGO 3 — Conviven dos sistemas de aviación con tamaños y horarios distintos.**
# * *Datos utilizados:* bitácora 2025 procesada (524.301 aterrizajes y despegues; 99,98 % con PMD válido).
# * *Evidencia:* la distribución del PMD es bimodal (≈1 t y 70-90 t); 54 % de las operaciones son de
#   aeronaves pesadas (> 50 t) y 40 % ligeras (< 5,7 t). Entre 00:00 y 05:59 ocurre el 11,4 % de las
#   operaciones pesadas, frente al 0,8 % de las ligeras, cuyo pico es a las 12:00.
# * *Visualización:* V3 y V4.
# * *Interpretación:* "la operación promedio" no existe: mezclar segmentos produce promedios engañosos
#   (la mediana de 77 t no describe a las avionetas). La aviación ligera opera casi exclusivamente de día.
# * *Posible utilidad:* asignar personal y servicios de torre/plataforma por franja horaria según el tipo de aeropuerto.
# * *Limitación:* un solo año; la categoría depende de un PMD con ~2,7 % de valores inválidos en la serie
#   completa; la asociación hora-tamaño no implica causalidad.

# %% [markdown]
# ## 15. Ética profesional (Parte XIX)
#
# **Decisión analizada:** usar un **índice base 2019 = 100** en la Visualización 2 en lugar de valores absolutos.
#
# * **¿Qué decisión tomé?** Dividir las operaciones anuales de cada tipo de tráfico por su valor de 2019.
# * **¿Qué evidencia la sustentó?** El doméstico (431.929 en 2019) es 5,8 veces el internacional (73.857);
#   en un eje absoluto la caída internacional se vería casi plana, y la pregunta es sobre recuperación relativa.
# * **¿Qué otra decisión pude adoptar?** (a) valores absolutos en un eje; (b) dos ejes Y; (c) base en otro
#   año (2018, 2000); (d) facetas con ejes independientes.
# * **¿Cómo afecta al resultado?** El índice hace que el internacional *parezca* el gran ganador (112 vs. 102),
#   cuando en volumen absoluto el doméstico sumó más operaciones sobre 2019 (+9.785) que el internacional (+8.730).
#   Además, la elección del año base cambia el relato: 2019 fue un año alto para el internacional
#   (base 2018 = 103,4); con otra base las cifras de "recuperación" cambian.
# * **¿Puede modificar la percepción del lector?** Sí: puede inducir a creer que el internacional "importa más"
#   o crece más en términos de actividad total.
# * **¿Qué debería declarar al presentar el gráfico?** Que es un **índice** (está en el título del eje y el
#   subtítulo), cuál es el año base y por qué, y los **volúmenes absolutos** del año base (incluidos en la nota
#   al pie). También debo declarar que las operaciones no equivalen a pasajeros.
#
# **Decisión complementaria declarada:** el CSV oficial incluye ~1.100 operaciones duplicadas que eliminé de la
# bitácora pero no puedo eliminar del resumen mensual (ya viene agregado). Lo declaro en lugar de "corregir"
# el CSV a mano, lo que violaría la regla de preservación de los datos originales.

# %% [markdown]
# ## 16. Conclusiones
#
# 1. El proceso completo es **reproducible**: los datos se descargan con `src/descargar_datos.py`, los
#    originales no se modifican y todas las transformaciones están en este notebook y en `src/`.
# 2. La calidad general de la fuente es buena (sin nulos en variables clave, coincidencia del 99,3 % entre
#    bitácora y resumen), pero hay problemas reales que afectan visualizaciones: ausentes ocultos como
#    cadenas vacías, PMD en unidades mezcladas, duplicados, una zona horaria distinta a la documentada y un
#    nombre de columna inconsistente.
# 3. La actividad aérea registrada está **fuertemente concentrada** en SCEL, que canaliza casi todo el
#    tráfico internacional (Hallazgo 1).
# 4. **H1 es consistente con los datos:** el tráfico internacional tardó un año más que el doméstico en volver
#    al nivel 2019 y hoy lo supera más en términos relativos (Hallazgo 2), sin que el dataset permita explicar la causa.
# 5. Los promedios esconden **dos poblaciones de aeronaves** con horarios distintos; segmentar es indispensable (Hallazgo 3).
# 6. Como diseñador, la principal lección es que la elección de la representación (índice vs. absoluto,
#    líneas vs. heatmap, escala log) es una decisión con consecuencias que debe declararse.
#
# **Limitaciones generales:** no hay pasajeros, carga ni coordenadas; excluye aviación militar/policial y
# aeródromos no DGAC; el último mes es provisional; los códigos de actividad y aerolíneas son internos DGAC
# sin diccionario público en la fuente.
