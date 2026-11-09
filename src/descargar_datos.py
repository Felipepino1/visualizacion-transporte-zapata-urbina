"""Descarga los recursos originales del dataset "Operaciones de Aeronaves" (datos.gob.cl).

Uso (desde la raíz del proyecto):
    python src/descargar_datos.py

Los archivos se guardan en data/raw/ y NO deben editarse manualmente.
Si un archivo ya existe, no se vuelve a descargar (usar --forzar para reemplazarlo).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"

BASE = "https://datos.gob.cl/dataset/d5fac3ed-01c7-43f0-9dc4-8970566bc059/resource"
RECURSOS = {
    "operaciones-aeropuertos.csv": f"{BASE}/2b7f3fbd-8be8-45e6-8ef5-64eea097528b/download/operaciones-aeropuertos.csv",
    "bitacora-vuelos.parquet": f"{BASE}/5bc8842e-d95a-4f8f-8ec6-acfbe700258a/download/bitacora-vuelos.parquet",
}


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main(forzar: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for nombre, url in RECURSOS.items():
        destino = RAW / nombre
        if destino.exists() and not forzar:
            print(f"[omitido] {nombre} ya existe ({destino.stat().st_size:,} bytes)")
            continue
        print(f"[descargando] {url}")
        urllib.request.urlretrieve(url, destino)
        print(f"[ok] {nombre}: {destino.stat().st_size:,} bytes, md5={md5(destino)}, "
              f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")


if __name__ == "__main__":
    main(forzar="--forzar" in sys.argv)
