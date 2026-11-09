# Parte I (Actividad 2) y Parte II — Análisis preliminar y formulación del problema

## Actividad 2. Análisis preliminar (antes de programar)

**¿Qué fenómeno representa el dataset?**
Las operaciones de aeronaves (aterrizajes, despegues y sobrevuelos) registradas en las bitácoras de los aeropuertos y aeródromos chilenos administrados por la DGAC, desde enero de 2000 hasta el último mes publicado.

**¿Quién produce los datos?**
Los registra el personal aeronáutico de cada unidad de la Dirección General de Aeronáutica Civil (DGAC) mediante formularios de bitácora; la Junta de Aeronáutica Civil (JAC) los procesa y publica en datos.gob.cl.

**¿Cuál es la unidad de observación?**
- Bitácora (parquet): **una operación** (un aterrizaje, despegue o sobrevuelo de una aeronave en un aeropuerto en un minuto dado).
- Resumen (CSV): **un aeropuerto en un mes para un tipo de tráfico** (doméstico o internacional), con el conteo de operaciones.

**¿Qué información permite analizar?**
Volumen de actividad por aeropuerto y en el tiempo (2000-2026), la división doméstico/internacional, la hora y el día de las operaciones, el tamaño de las aeronaves (peso máximo de despegue), el modelo de avión, el operador (código DGAC) y el origen/destino de cada vuelo.

**¿Qué información no proporciona?**
Pasajeros, carga, tarifas, retrasos o cancelaciones; ubicación geográfica (región, coordenadas) de los aeropuertos; nombres de aeropuertos o aerolíneas (sólo códigos); diccionario de los códigos de actividad; vuelos militares/policiales y de aeródromos no administrados por la DGAC.

**¿Qué dimensiones identifico preliminarmente?**
Tiempo (año → mes → día → hora), aeropuerto, tipo de tráfico (doméstico/internacional), tipo de operación (A/D/W), tamaño/modelo de aeronave, operador y actividad. La medida principal es el **conteo de operaciones**; el PMD es una medida no aditiva.

**¿Qué potencial utilidad podría tener para una organización?**
- Para la DGAC / concesionarias: planificación de capacidad, dotación de torres y plataformas por franja horaria.
- Para la JAC / Ministerio de Transportes: seguimiento de la recuperación del sector y de la concentración del tráfico internacional.
- Para gobiernos regionales y turismo: evolución de la conectividad aérea de cada aeropuerto.

## Parte II. Formulación del problema

**Pregunta principal**
¿Cómo se distribuye y cómo ha evolucionado la actividad aeroportuaria registrada en Chile entre 2000 y 2026 según aeropuerto, tipo de tráfico (doméstico/internacional), tamaño de aeronave y momento del día, y qué patrones temporales la caracterizan?

**Preguntas secundarias**
- **P1:** ¿Qué aeropuertos concentran las operaciones y qué peso tiene el tráfico internacional en cada uno?
- **P2:** ¿Cómo evolucionaron las operaciones domésticas e internacionales entre 2000 y 2025 y cómo se recuperó cada una tras la caída de 2020?
- **P3:** ¿Cómo se distribuyen las operaciones según tamaño de aeronave y hora del día, y difieren esos patrones entre segmentos?

**Hipótesis exploratoria**
- **H1:** Tras la caída de 2020, el tráfico internacional se recuperó más lentamente que el doméstico respecto de su nivel de 2019.

> La hipótesis no es una conclusión: se contrasta descriptivamente en el notebook (sección 14, Hallazgo 2) y no permite atribuir causas.
