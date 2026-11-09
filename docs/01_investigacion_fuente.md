# Parte I. Investigación de la fuente

## Actividad 1. Localización

| Campo | Detalle |
|---|---|
| **Nombre** | Operaciones de Aeronaves |
| **Institución responsable** | Junta de Aeronáutica Civil (JAC), Ministerio de Transportes y Telecomunicaciones. Contacto declarado: "Unidad de Datos JAC" (data@jac.gob.cl). Los datos de origen son recolectados por la Dirección General de Aeronáutica Civil (DGAC). |
| **Enlace** | https://datos.gob.cl/dataset/operaciones-aeronaves (grupo *Transporte*: https://datos.gob.cl/group/transporte) |
| **Documentación técnica** | https://jac-mtt.github.io/jac-docs/docs/series/operaciones_aeroportuarias/ |
| **Descripción** | Dos series obtenidas a partir de las bitácoras de los aeropuertos de Chile: (1) un subconjunto de la bitácora de vuelos (registro por operación) y (2) un resumen mensual de operaciones por aeropuerto. |
| **Licencia** | Creative Commons CCZero (CC0) — dominio público, `isopen: true`. |
| **Frecuencia declarada** | Mensual (`frequency: 1m`). Se actualiza cerca del día 10 de cada mes con el mes anterior. |
| **Versión / metadatos** | Versión 0.1.0; dataset creado el 07-03-2025; metadatos modificados el 13-10-2026. Campo `period` declara "marzo 2025" (desactualizado respecto de los datos). |

### Archivos disponibles

| Recurso | Archivo | Formato | Tamaño declarado | Granularidad | Contenido |
|---|---|---|---|---|---|
| Bitácora de Vuelos | `bitacora-vuelos.parquet` | Parquet (compresión zstd) | 138.247.225 bytes (~138 MB) | 1 fila = 1 operación (aterrizaje, despegue o sobrevuelo) | 12 columnas: fecha-hora UTC, aeropuerto, tipo de operación, origen/destino, aerolínea, n.º de vuelo, actividad, matrícula, modelo, internacional, PMD |
| Operaciones en Aeropuertos | `operaciones-aeropuertos.csv` | CSV, UTF-8, separado por comas, con cabecera | 285.004 bytes | 1 fila = aeropuerto × mes × tipo de tráfico (I/D) | 4 columnas: `mes_id`, `aeropuerto_oaci`, `internacional_domestico`, `cnt_operaciones` |

URLs directas de descarga:

- CSV: `https://datos.gob.cl/dataset/d5fac3ed-01c7-43f0-9dc4-8970566bc059/resource/2b7f3fbd-8be8-45e6-8ef5-64eea097528b/download/operaciones-aeropuertos.csv`
- Parquet: `https://datos.gob.cl/dataset/d5fac3ed-01c7-43f0-9dc4-8970566bc059/resource/5bc8842e-d95a-4f8f-8ec6-acfbe700258a/download/bitacora-vuelos.parquet`

### Metadatos de variables (según documentación JAC)

**Operaciones en Aeropuertos (CSV)**

| Columna | Tipo declarado | Descripción |
|---|---|---|
| `mes_id` | int | Año/mes de las operaciones (`200001` = enero 2000). Construido desde `dt_operacion` convertido a hora local. |
| `aeropuerto_oaci` | string | Código OACI del aeropuerto/aeródromo que informa. |
| `internacional_domestico` | string | `I` internacional / `D` doméstico. |
| `n_operaciones` | int | Número de despegues + aterrizajes informados. **En el archivo real la columna se llama `cnt_operaciones`** (inconsistencia entre documentación y datos). |

**Bitácora de Vuelos (Parquet)**: `dt_operacion` (datetime UTC), `aeropuerto_oaci`, `tipo_operacion` (A/D/W), `aeropuerto_dgac_orig_dest`, `aerolinea_dgac`, `numero_vuelo`*, `actividad_cod`, `matricula`, `modelo_avion`*, `modelo_avion_desc`*, `es_internacional` (bool), `pmd`* (peso máximo de despegue). (*) columnas que la documentación declara con nulos.

### Periodo cubierto

No está declarado explícitamente en los metadatos (el campo `period` dice "marzo 2025", que corresponde a la fecha de publicación, no a la cobertura). La cobertura real se determina en el perfilamiento (ver notebook).

### Condiciones y limitaciones identificadas

1. **Cobertura incompleta por diseño:** se excluyen operaciones militares/policiales y las de aeródromos que no administra la DGAC. Los conteos *no* son el total de la actividad aérea del país.
2. **El CSV excluye sobrevuelos (W)**; la bitácora sí los incluye. Ambos recursos no son directamente equivalentes.
3. **Mes t-1 provisional:** los registros del último día de t-2 pueden corregirse; el último mes publicado puede estar incompleto.
4. **Zona horaria:** `dt_operacion` está en UTC; `mes_id` se construye en hora local. Una agregación ingenua por mes en UTC no coincidirá exactamente con el CSV.
5. **Códigos no estándar:** aeródromos y aerolíneas sin código OACI reciben códigos internos DGAC.
6. **Sin dimensión geográfica explícita:** no hay región, ciudad ni coordenadas; sólo códigos OACI.
7. **Inconsistencia documental:** `n_operaciones` (doc) vs `cnt_operaciones` (archivo).
8. **Tamaño:** el parquet (~138 MB) supera el límite de 100 MB por archivo de GitHub, por lo que no se versiona; se documenta su descarga reproducible.
