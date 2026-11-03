#!/usr/bin/env python3
"""
Photo Manager CLI v1.6.4
Herramienta completa de gestión de fotos.

Funcionalidades:
  1. Deduplicar fotos (pHash + BK-tree + complete-link clustering)
  2. Comparar dos carpetas en busca de duplicados cruzados
  3. Mejorar fotos con IA (FSRCNN/ESPCN super-resolución) + OpenCV
  4. Organizar por fecha (Año/Mes desde EXIF)
  5. Detectar fotos borrosas (Laplacian variance)
  6. Comprimir / Convertir (HEIF, WebP, JPEG)
  7. Estadísticas de carpeta

Dependencias:
  pip install Pillow imagehash pillow-heif opencv-contrib-python==4.10.0.84

Autor: Josmar Angulo / Claude
"""

import csv
import hashlib
import math
import os
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# numpy: carga diferida — no se necesita si solo se usa dedup
try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# ══════════════════════════════════════════════════
# DEPENDENCIAS
# ══════════════════════════════════════════════════

try:
    from PIL import Image, ImageFilter
    import imagehash
except ImportError:
    print("Faltan dependencias base. Instala con:")
    print("  pip install Pillow imagehash")
    sys.exit(1)

# HEIF (opcional)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIF = True
except ImportError:
    _HEIF = False

# OpenCV (opcional — para mejora y detección de blur)
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

# OpenCV dnn_superres (opcional — super-resolución IA)
_SUPERRES = False
if _CV2:
    try:
        _sr_test = cv2.dnn_superres.DnnSuperResImpl_create()
        _SUPERRES = True
    except AttributeError:
        pass

# ══════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════

VERSION = "1.6.4"

# [FIX #12] Separar formatos soportados por Pillow vs RAW
PILLOW_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".gif",
}
RAW_EXTENSIONS = {".raw", ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2"}
IMAGE_EXTENSIONS = PILLOW_EXTENSIONS | RAW_EXTENSIONS

MODELS_DIR = Path.home() / ".photo_manager" / "models"

MODELS = {
    "fsrcnn_x2": {
        "file": "FSRCNN_x2.pb",
        "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
        "algo": "fsrcnn",
        "scale": 2,
        "desc": "FSRCNN ×2 — super-resolución ligera (40 KB)",
    },
    "espcn_x2": {
        "file": "ESPCN_x2.pb",
        "url": "https://github.com/Saafke/ESPCN_Tensorflow/raw/master/models/ESPCN_x2.pb",
        "algo": "espcn",
        "scale": 2,
        "desc": "ESPCN ×2 — super-resolución rápida (100 KB)",
    },
}

MESES = {
    1: "01-Enero", 2: "02-Febrero", 3: "03-Marzo", 4: "04-Abril",
    5: "05-Mayo", 6: "06-Junio", 7: "07-Julio", 8: "08-Agosto",
    9: "09-Septiembre", 10: "10-Octubre", 11: "11-Noviembre", 12: "12-Diciembre",
}

# ══════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════


def cls():
    os.system("clear" if os.name != "nt" else "cls")


def banner(title: str):
    w = 50
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}\n")


def ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    val = input(f"  {prompt}{hint}: ").strip()
    return val or default


def ask_folder(prompt: str, must_exist: bool = True) -> Path | None:
    raw = ask(prompt)
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if must_exist and not p.is_dir():
        print(f"  ✗ No existe: {p}")
        return None
    return p


# [FIX #9] ask_int con rango min/max
def ask_int(prompt: str, default: int, min_val: int | None = None,
            max_val: int | None = None) -> int:
    raw = ask(prompt, str(default))
    try:
        val = int(raw)
    except ValueError:
        return default
    if min_val is not None and val < min_val:
        print(f"    ⚠ Valor mínimo: {min_val}")
        val = min_val
    if max_val is not None and val > max_val:
        print(f"    ⚠ Valor máximo: {max_val}")
        val = max_val
    return val


def ask_confirm(msg: str = "¿Continuar?", default_yes: bool = False) -> bool:
    hint = "(S/n)" if default_yes else "(s/N)"
    resp = input(f"  {msg} {hint}: ").strip().lower()
    if not resp:
        return default_yes
    return resp in ("s", "si", "sí", "y", "yes")


# [FIX #15] find_images retorna lista ordenada
def _is_relative_to(path: Path, parent: Path) -> bool:
    """Comprueba relación real entre rutas, sin comparar prefijos de texto."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_source_output(source: Path, output: Path) -> bool:
    """Permite output dentro de source, pero nunca que output contenga source."""
    source, output = source.resolve(), output.resolve()
    if source == output or _is_relative_to(source, output):
        print("  ✗ La salida no puede ser el origen ni contenerlo.")
        return False
    return True


def find_images(root: Path, exclude: Path | None = None,
                warn_raw: bool = True) -> list[Path]:
    """Encuentra imágenes recursivamente y excluye una ruta y sus descendientes."""
    root = root.expanduser().resolve()
    excluded = exclude.expanduser().resolve() if exclude else None
    images, raw_count = [], 0

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # os.walk(followlinks=False) no sigue symlinks: dirpath ya está
        # bajo root resuelto. Evitamos resolve() por directorio (FS call).
        dp = Path(dirpath)

        # Comprobar igualdad/descendencia sin volver a consultar el filesystem.
        if excluded and (dp == excluded or excluded in dp.parents):
            dirnames[:] = []
            continue

        if excluded:
            kept = []
            for name in dirnames:
                child = dp / name
                if child != excluded and excluded not in child.parents:
                    kept.append(name)
            dirnames[:] = kept

        for fname in filenames:
            suffix = Path(fname).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                if suffix in RAW_EXTENSIONS:
                    raw_count += 1
                else:
                    images.append(dp / fname)

    if raw_count and warn_raw:
        print(f"  ⚠ {raw_count} archivos RAW ignorados (requiere rawpy)")
    return sorted(images)


def get_photo_date(filepath: Path) -> datetime | None:
    """Extrae fecha EXIF o de modificación."""
    try:
        with Image.open(filepath) as img:
            # [FIX #19] Usar getexif() en vez de _getexif()
            exif = img.getexif()
            if exif:
                for tag_id in (36867, 36868, 306):
                    val = exif.get(tag_id)
                    if val and isinstance(val, str):
                        try:
                            return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            continue
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(filepath.stat().st_mtime)
    except Exception:
        return None


# [FIX #10] Hash SHA-256 streaming en chunks de 1 MB en vez de leer todo en memoria
def get_image_info(filepath: Path, hash_size: int = 12,
                   skip_counts: dict | None = None) -> dict | None:
    """Hash perceptual + metadata de una imagen.

    skip_counts, si se proporciona, acumula motivos de omisión para resumirlos.
    """
    try:
        with Image.open(filepath) as img:
            if getattr(img, "n_frames", 1) > 1:
                if skip_counts is not None:
                    skip_counts["multipage"] = skip_counts.get("multipage", 0) + 1
                return None
            phash = imagehash.phash(img.convert("RGB"), hash_size=hash_size)
            w, h = img.size
            exif_date = None
            try:
                exif = img.getexif()
                if exif:
                    for tag_id in (36867, 306):
                        val = exif.get(tag_id)
                        if val and isinstance(val, str):
                            try:
                                exif_date = datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                                break
                            except ValueError:
                                continue
            except Exception:
                pass
            mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)

            # Streaming SHA-256
            file_digest = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    file_digest.update(chunk)

            return {
                "path": filepath,
                "phash": phash,
                "resolution": w * h,
                "width": w,
                "height": h,
                "date": exif_date or mod_time,
                "file_size": filepath.stat().st_size,
                "file_hash": file_digest.hexdigest(),
            }
    except Exception as e:
        if skip_counts is not None:
            skip_counts["error"] = skip_counts.get("error", 0) + 1
            skip_counts.setdefault("error_samples", []).append(f"{filepath.name}: {e}")
        else:
            print(f"  ⚠ Error: {filepath.name} — {e}")
        return None


# [FIX #16] download_model con timeout y archivo temporal
def _validate_model_file(path: Path, info: dict) -> bool:
    """Valida tamaño y que OpenCV pueda cargar/configurar la red."""
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            return False
        if _SUPERRES:
            test_sr = cv2.dnn_superres.DnnSuperResImpl_create()
            test_sr.readModel(str(path))
            test_sr.setModel(info["algo"], info["scale"])
        return True
    except Exception as exc:
        print(f"  ⚠ Modelo inválido: {exc}")
        return False


def download_model(name: str) -> Path | None:
    """Descarga un modelo IA si no existe localmente."""
    info = MODELS.get(name)
    if not info:
        print(f"  ✗ Modelo desconocido: {name}")
        return None
    dest = MODELS_DIR / info["file"]
    if dest.exists():
        if _validate_model_file(dest, info):
            return dest
        print(f"  ⚠ Modelo local inválido; se descargará nuevamente: {dest}")
        dest.unlink(missing_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Descargando {info['desc']}...")
    tmp = f"{dest}.{os.getpid()}.tmp"
    try:
        req = urllib.request.Request(info["url"])
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if not _validate_model_file(Path(tmp), info):
            raise ValueError("el archivo descargado no es un modelo válido")
        os.replace(tmp, str(dest))
        print(f"  ✓ Modelo descargado y validado por OpenCV: {dest}")
        return dest
    except Exception as e:
        print(f"  ✗ Error descargando: {e}")
        return None
    finally:
        # Limpiar .tmp huérfano (cubre KeyboardInterrupt que es BaseException)
        try:
            if os.path.exists(tmp) and not dest.exists():
                os.unlink(tmp)
        except OSError:
            pass


# [FIX #17] format_size con TB
def format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / (1024 ** 2):.1f} MB"
    elif n_bytes < 1024 ** 4:
        return f"{n_bytes / (1024 ** 3):.2f} GB"
    else:
        return f"{n_bytes / (1024 ** 4):.2f} TB"


# ══════════════════════════════════════════════════
# 1. DEDUPLICACIÓN
# ══════════════════════════════════════════════════


# [v1.5] BK-tree: búsqueda por radio sin descartar buckets
def group_duplicates(infos: list[dict], threshold: int) -> list[list[dict]]:
    """Agrupa hashes exactos y vecinos pHash sin encadenamiento transitivo."""
    if threshold < 0:
        raise ValueError("threshold no puede ser negativo")
    if not infos:
        return []
    values = [int(str(info["phash"]), 16) for info in infos]

    def hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    # Cada archivo comienza en su propio grupo; bytes idénticos se fusionan.
    groups = [[i] for i in range(len(infos))]
    exact = defaultdict(list)
    for i, info in enumerate(infos):
        exact[info["file_hash"]].append(i)
    for indexes in exact.values():
        if len(indexes) > 1:
            root_idx = indexes[0]
            for idx in indexes[1:]:
                groups[root_idx].append(idx)
                groups[idx] = []

    # BK-tree para generar todos los pares candidatos dentro del umbral.
    root, candidate_pairs = None, set()
    for i, value in enumerate(values):
        if root is not None:
            stack = [root]
            while stack:
                node = stack.pop()
                dist = hamming(value, node[0])
                j = node[1]
                if dist <= threshold and infos[i]["file_hash"] != infos[j]["file_hash"]:
                    candidate_pairs.add((j, i))
                stack.extend(child for edge, child in node[2].items()
                             if dist - threshold <= edge <= dist + threshold)
        if root is None:
            root = [value, i, {}]
        else:
            node = root
            while True:
                dist = hamming(value, node[0])
                if dist not in node[2]:
                    node[2][dist] = [value, i, {}]
                    break
                node = node[2][dist]

    # Mapear TODOS los índices, incluidos los absorbidos por grupos exactos.
    # Los índices absorbidos tienen groups[idx] == [], pero siguen apareciendo
    # como extremos en candidate_pairs.
    owner = {}
    for gid, group in enumerate(groups):
        if group:
            for idx in group:
                owner[idx] = gid
    for indexes in exact.values():
        if len(indexes) > 1:
            root_gid = owner[indexes[0]]
            for idx in indexes[1:]:
                owner[idx] = root_gid

    # Complete-link: toda pareja entre grupos debe cumplir el umbral.
    for i, j in sorted(candidate_pairs):
        gi, gj = owner[i], owner[j]
        if gi == gj or not groups[gi] or not groups[gj]:
            continue
        left, right = groups[gi], groups[gj]
        if all(infos[a]["file_hash"] == infos[b]["file_hash"]
               or hamming(values[a], values[b]) <= threshold
               for a in left for b in right):
            left.extend(right)
            groups[gj] = []
            for idx in right:
                owner[idx] = gi

    return [[infos[i] for i in group] for group in groups if group]


def pick_best(group: list[dict]) -> dict:
    return max(group, key=lambda x: (x["resolution"], x["file_size"],
                                     -x["date"].timestamp()))


# [FIX C2] _convert_heif retorna (path escrito, motivo_fallback | None)
def _convert_heif(src: Path, dst: Path, quality: int = 85
                  ) -> tuple[Path | None, str | None]:
    """Convierte imagen a HEIF.

    Retorna (path_escrito, motivo). motivo es None si se convirtió a HEIF,
    o "heic" / "alpha" / "exception" si hubo fallback al formato original.
    path_escrito es None solo si todo falló.
    """
    # [v1.3 C1] HEIC/HEIF ya es HEIF — copiar sin recomprimir
    if src.suffix.lower() in (".heic", ".heif"):
        try:
            shutil.copy2(str(src), str(dst))
            return (dst, "heic") if dst.exists() else (None, None)
        except Exception:
            return (None, None)
    try:
        with Image.open(src) as img:
            exif = img.info.get("exif")
            # [v1.3 C2] Detectar canal alfa — HEIF no soporta alfa
            has_alpha = (
                img.mode in ("RGBA", "LA", "PA", "La")
                or (img.mode == "P" and "transparency" in img.info)
            )
            if has_alpha:
                # No convertir a HEIF: perdería canal alfa → copiar en formato original
                fallback = dst.with_suffix(src.suffix)
                # Anti-colisión en fallback
                if fallback.exists():
                    stem, sfx, c = fallback.stem, fallback.suffix, 1
                    while fallback.exists():
                        fallback = fallback.parent / f"{stem}_{c}{sfx}"
                        c += 1
                shutil.copy2(str(src), str(fallback))
                return (fallback, "alpha") if fallback.exists() else (None, None)
            if img.mode not in ("RGB",):
                img = img.convert("RGB")
            kw = {"format": "HEIF", "quality": quality}
            if exif:
                kw["exif"] = exif
            img.save(str(dst), **kw)
            if dst.exists() and dst.stat().st_size > 0:
                return (dst, None)
            return (None, None)
    except Exception:
        # Fallback: copiar en formato original (path distinto a dst)
        try:
            fallback = dst.with_suffix(src.suffix)
            # [v1.3 M11] Anti-colisión en fallback
            if fallback.exists():
                stem, sfx, c = fallback.stem, fallback.suffix, 1
                while fallback.exists():
                    fallback = fallback.parent / f"{stem}_{c}{sfx}"
                    c += 1
            shutil.copy2(str(src), str(fallback))
            return (fallback, "exception") if fallback.exists() else (None, None)
        except Exception:
            return (None, None)


def run_dedup(source: Path, output: Path, compare: Path | None = None,
              hash_size: int = 12, threshold: int = 6,
              no_convert: bool = False, quality: int = 85,
              keep_originals: bool = True, dry_run: bool = False):
    """Ejecuta deduplicación completa.

    Por seguridad, keep_originals=True por defecto: nunca elimina originales
    salvo que el usuario lo solicite explícitamente y confirme la operación.
    """
    # [v1.3 M7] Validar hash_size
    if hash_size < 4 or hash_size > 32:
        print(f"  ✗ hash_size debe estar entre 4 y 32 (recibido: {hash_size})")
        return
    if threshold < 0 or threshold > hash_size * hash_size:
        print(f"  ✗ threshold debe estar entre 0 y {hash_size * hash_size}.")
        return

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    compare = compare.expanduser().resolve() if compare else None
    compare_mode = compare is not None
    if not _validate_source_output(source, output):
        return
    if compare_mode and compare and not _validate_source_output(compare, output):
        return

    print(f"\n  Origen:    {source}")
    if compare_mode:
        print(f"  Comparar:  {compare}")
    print(f"  Salida:    {output}")
    print(f"  Threshold: {threshold} | Hash size: {hash_size}")
    fmt = "original" if no_convert else f"HEIF (q={quality})"
    print(f"  Formato:   {fmt}")
    print(f"  Originales: {'conservar' if keep_originals else 'BORRAR'}")
    print(f"  Modo:      {'DRY RUN' if dry_run else 'EJECUTAR'}\n")

    # [FIX H3] Detectar overlap entre source y compare
    if compare_mode and compare:
        try:
            src_resolved = source.resolve()
            cmp_resolved = compare.resolve()
            if (_is_relative_to(cmp_resolved, src_resolved)
                    or _is_relative_to(src_resolved, cmp_resolved)
                    or src_resolved == cmp_resolved):
                print("  ✗ Las carpetas origen y comparación se solapan.")
                print(f"    A: {source}")
                print(f"    B: {compare}")
                return
        except Exception:
            pass

    if not no_convert and not _HEIF:
        print("  ✗ HEIF no disponible. Instala pillow-heif o usa sin conversión.")
        return

    # Escanear
    print("  [1/5] Buscando imágenes...")
    images_a = find_images(source, exclude=output)
    source_a_set = set(str(p) for p in images_a)
    if compare_mode:
        images_b = find_images(compare, exclude=output)
        print(f"    Carpeta A: {len(images_a)} | Carpeta B: {len(images_b)}")
        images = images_a + images_b
    else:
        images = images_a
    print(f"    Total: {len(images)} imágenes")

    if not images:
        print("  No se encontraron imágenes.")
        return

    # Analizar
    print("  [2/5] Analizando (hash + metadata)...")
    infos = []
    skip_counts = {"multipage": 0, "error": 0, "error_samples": []}
    for i, img in enumerate(images, 1):
        if i % 50 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}]", end="\r")
        info = get_image_info(img, hash_size, skip_counts=skip_counts)
        if info:
            infos.append(info)
    print(f"    {len(infos)} procesadas OK")
    if skip_counts["multipage"]:
        print(f"    ℹ {skip_counts['multipage']} archivo(s) animado(s)/multipágina omitido(s)")
    if skip_counts["error"]:
        print(f"    ⚠ {skip_counts['error']} archivo(s) ilegible(s) omitido(s)")
        for sample in skip_counts["error_samples"][:3]:
            print(f"      - {sample}")
        if skip_counts["error"] > 3:
            print(f"      ... y {skip_counts['error'] - 3} más")

    # Agrupar
    print("  [3/5] Detectando duplicados...")
    t0 = time.time()
    groups = group_duplicates(infos, threshold)
    dup_groups = [g for g in groups if len(g) > 1]
    discarded = sum(len(g) - 1 for g in dup_groups)

    print(f"    Tiempo: {time.time() - t0:.1f}s")
    print(f"    Únicas: {sum(1 for g in groups if len(g) == 1)}")
    print(f"    Grupos duplicados: {len(dup_groups)} ({discarded} a descartar)")
    print(f"    A conservar: {len(groups)} imágenes finales")

    if compare_mode:
        cross = sum(1 for g in dup_groups
                    if any(str(i["path"]) in source_a_set for i in g)
                    and any(str(i["path"]) not in source_a_set for i in g))
        print(f"    Duplicados cruzados A↔B: {cross}")

    # Precalcular pick_best por índice (evita id() frágil)
    winners = [pick_best(group) for group in groups]
    report_rows = []

    for idx, group in enumerate(groups):
        best = winners[idx]
        for item in group:
            origin = ""
            if compare_mode:
                origin = "A" if str(item["path"]) in source_a_set else "B"
            row = {
                "archivo": str(item["path"]),
                "resolucion": f"{item['width']}x{item['height']}",
                "tamano_mb": round(item["file_size"] / (1024 ** 2), 2),
                "fecha": item["date"].strftime("%Y-%m-%d %H:%M"),
                "phash": str(item["phash"]),
                "grupo": str(best["phash"]),
                "decision": "CONSERVAR" if item is best else "DESCARTAR",
            }
            if compare_mode:
                row["carpeta"] = origin
            report_rows.append(row)

    winner_paths_set = set(w["path"] for w in winners)
    space_saved = sum(i["file_size"] for g in dup_groups
                      for i in g if i["path"] not in winner_paths_set)
    print(f"    Espacio a recuperar: {format_size(space_saved)}\n")

    will_delete = not keep_originals and not dry_run
    if will_delete:
        print("  ⚠ Se borrarán únicamente los duplicados descartados; los ganadores y únicos se conservan.")
        if not ask_confirm():
            print("  Cancelado.")
            return
        print()

    # Convertir/copiar
    converted_ok = set()  # [FIX #1, #2] Tracking de conversiones exitosas

    if dry_run:
        print("  [4/5] DRY RUN — sin cambios")
    else:
        converting = not no_convert
        action = "Convirtiendo a HEIF" if converting else "Copiando"
        print(f"  [4/5] {action} {len(winners)} imágenes...")
        output.mkdir(parents=True, exist_ok=True)

        orig_total, conv_total = 0, 0
        heic_copied = alpha_copied = exception_copied = 0
        for i, w in enumerate(winners, 1):
            src = w["path"]
            # [FIX #11] Paths relativos correctos para compare
            try:
                rel = src.relative_to(source)
            except ValueError:
                if compare_mode and compare:
                    try:
                        rel = Path("B") / src.relative_to(compare)
                    except ValueError:
                        rel = Path(src.name)
                else:
                    rel = Path(src.name)

            if converting:
                dst = output / rel.with_suffix(".heif")
            else:
                dst = output / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Colisiones
            if dst.exists():
                stem, sfx, c = dst.stem, dst.suffix, 1
                while dst.exists():
                    dst = dst.parent / f"{stem}_{c}{sfx}"
                    c += 1

            orig_total += w["file_size"]
            actual_dst = None
            if converting:
                actual_dst, reason = _convert_heif(src, dst, quality)
                if reason == "heic":
                    heic_copied += 1
                elif reason == "alpha":
                    alpha_copied += 1
                elif reason == "exception":
                    exception_copied += 1
            else:
                try:
                    shutil.copy2(str(src), str(dst))
                    actual_dst = dst if dst.exists() else None
                except Exception:
                    actual_dst = None

            if actual_dst:
                converted_ok.add(src)
                try:
                    conv_total += actual_dst.stat().st_size
                except Exception:
                    pass
            else:
                print(f"  ⚠ Falló conversión: {src.name}")

            if i % 20 == 0 or i == len(winners):
                print(f"    [{i}/{len(winners)}] {i * 100 // len(winners)}%", end="\r")
        print()
        if converting and orig_total:
            print(f"    Original: {format_size(orig_total)} → HEIF: {format_size(conv_total)}"
                  f" ({(1 - conv_total / orig_total) * 100:.0f}% reducción)")
        if heic_copied:
            print(f"    ℹ {heic_copied} archivo(s) HEIC copiado(s) sin recomprimir (ya son HEIF)")
        if alpha_copied:
            print(f"    ℹ {alpha_copied} archivo(s) con alfa copiado(s) sin convertir (HEIF perdería transparencia)")
        if exception_copied:
            print(f"    ⚠ {exception_copied} archivo(s) copiado(s) en formato original (falló conversión HEIF)")

    # [FIX C1] Solo borrar si el ganador del grupo fue convertido OK
    if will_delete:
        safe_to_delete = set()
        # Solo se eliminan duplicados descartados; nunca el ganador ni archivos únicos.
        for group in dup_groups:
            winner_path = pick_best(group)["path"]
            if winner_path not in converted_ok:
                print(f"  ⚠ Grupo protegido (ganador no convertido): {winner_path.name}")
                continue
            safe_to_delete.update(item["path"] for item in group
                                  if item["path"] != winner_path)

        print(f"  [5/5] Borrando {len(safe_to_delete)} originales verificados...")
        deleted = 0
        for fpath in sorted(safe_to_delete):
            try:
                fpath.unlink()
                deleted += 1
            except Exception:
                pass
        # Limpiar directorios vacíos
        for root_dir in ([source, compare] if compare_mode else [source]):
            if root_dir:
                for dp, _, _ in os.walk(root_dir, topdown=False):
                    p = Path(dp)
                    if p != root_dir and p != output:
                        try:
                            if not any(p.iterdir()):
                                p.rmdir()
                        except Exception:
                            pass
        print(f"    {deleted} archivos eliminados")
    elif not dry_run:
        print("  [5/5] Originales conservados.")

    # [FIX #20] Reporte CSV con try/except
    if dry_run:
        print("\n  ℹ DRY RUN: no se creó ningún archivo ni se modificó el origen.")
    else:
        report_path = output / "dedup_reporte.csv"
        fields = ["archivo", "resolucion", "tamano_mb", "fecha", "phash", "grupo", "decision"]
        if compare_mode:
            fields.insert(1, "carpeta")
        try:
            with open(report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"\n  ✓ Reporte: {report_path}")
        except Exception as e:
            print(f"\n  ⚠ No se pudo escribir el reporte: {e}")

    print(f"  ✓ Conservadas: {len(winners)} | Descartadas: {discarded}")


# ══════════════════════════════════════════════════
# 3. MEJORA DE FOTOS (IA + OpenCV)
# ══════════════════════════════════════════════════


def _ensure_cv2():
    if not _CV2:
        print("  ✗ OpenCV no disponible. Instala:")
        print("    pip install opencv-contrib-python==4.10.0.84")
        return False
    return True


# [FIX #7] Leer imagen con fallback a Pillow para HEIC/formatos no-OpenCV
def _read_image_cv2(img_path: Path):
    """Lee imagen para OpenCV. Intenta cv2, luego Pillow como fallback."""
    img = cv2.imread(str(img_path))
    if img is not None:
        return img
    # Fallback: Pillow (soporta HEIC, GIF, etc.)
    try:
        with Image.open(img_path) as pil_img:
            rgb = pil_img.convert("RGB")
            arr = np.array(rgb)[:, :, ::-1].copy()  # RGB → BGR
            return arr
    except Exception:
        return None


# [FIX H2] Retorna Path escrito o None (caller necesita saber el path real)
def _write_image_cv2(img, dst: Path, quality: int = 95) -> Path | None:
    """Escribe imagen. Usa Pillow para HEIF, cv2 para el resto.
    Retorna el path realmente escrito, o None si falló."""
    ext = dst.suffix.lower()
    try:
        if ext in (".heic", ".heif") and _HEIF:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_out = Image.fromarray(rgb)
            pil_out.save(str(dst), format="HEIF", quality=quality)
            return dst if dst.exists() else None
        else:
            params = []
            if ext in (".jpg", ".jpeg"):
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == ".png":
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            elif ext == ".webp":
                params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            ok = cv2.imwrite(str(dst), img, params if params else None)
            if ok:
                return dst
            # Fallback a JPEG con anti-colisión
            fallback = dst.with_suffix(".jpg")
            base = fallback
            counter = 1
            while fallback.exists():
                fallback = base.with_name(f"{base.stem}_{counter}{base.suffix}")
                counter += 1
            ok = cv2.imwrite(str(fallback), img,
                             [cv2.IMWRITE_JPEG_QUALITY, quality])
            return fallback if ok else None
    except Exception:
        return None


def enhance_clahe(img):
    """CLAHE en canal L (espacio LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def enhance_white_balance(img):
    """Balance de blancos — gray world."""
    result = img.copy().astype("float64")
    avg_b, avg_g, avg_r = [result[:, :, i].mean() for i in range(3)]
    avg_all = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] *= avg_all / max(avg_b, 1)
    result[:, :, 1] *= avg_all / max(avg_g, 1)
    result[:, :, 2] *= avg_all / max(avg_r, 1)
    return result.clip(0, 255).astype("uint8")


def enhance_gamma(img):
    """Corrección de gamma adaptativa."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum = gray.mean()
    if mean_lum < 80:
        gamma = 0.6
    elif mean_lum < 120:
        gamma = 0.8
    elif mean_lum > 200:
        gamma = 1.4
    elif mean_lum > 170:
        gamma = 1.2
    else:
        return img
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, lut)


def enhance_denoise(img, strength: int = 6):
    """Reducción de ruido preservando detalle."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def enhance_sharpen(img, amount: float = 0.5):
    """Unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


# [FIX #4] Verificar retorno de cv2.imwrite en superres
def enhance_superres(img_path: Path, output_path: Path,
                     model_name: str = "fsrcnn_x2",
                     model_path: Path | None = None) -> Path | None:
    """Super-resolución con red neuronal (OpenCV dnn_superres)."""
    if not _SUPERRES:
        print("  ✗ dnn_superres no disponible. Necesitas opencv-contrib-python==4.10.0.84")
        return None

    # [v1.4] Usar model_path pre-descargado si se pasa; si no, descargar
    if model_path is None:
        model_path = download_model(model_name)
    if not model_path:
        return None

    info = MODELS[model_name]
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(info["algo"], info["scale"])

    img = _read_image_cv2(img_path)
    if img is None:
        print(f"  ✗ No se pudo leer: {img_path.name}")
        return None

    # Validar tamaño mínimo para superres
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        print(f"  ⚠ Imagen demasiado pequeña para superres ({w}x{h}): {img_path.name}")
        return None

    result = sr.upsample(img)
    written = _write_image_cv2(result, output_path)
    if not written:
        print(f"  ✗ No se pudo escribir: {output_path}")
        return None
    return written


def enhance_auto(img):
    """Pipeline completo: denoise → WB → CLAHE → gamma → sharpen."""
    img = enhance_denoise(img, strength=4)
    img = enhance_white_balance(img)
    img = enhance_clahe(img)
    img = enhance_gamma(img)
    img = enhance_sharpen(img, amount=0.4)
    return img


def _restore_metadata(source: Path, destination: Path) -> bool:
    """Restaura EXIF, ICC y DPI. Devuelve False si no pudo restaurarlos."""
    try:
        with Image.open(source) as original:
            if getattr(original, "n_frames", 1) > 1:
                return True
            exif = original.getexif()
            # OpenCV aplica normalmente la orientación EXIF al leer.
            if exif and 274 in exif:
                exif[274] = 1
            exif_bytes = exif.tobytes() if exif else None
            icc = original.info.get("icc_profile")
            dpi = original.info.get("dpi")

        with Image.open(destination) as processed:
            save_kwargs = {}
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            if icc:
                save_kwargs["icc_profile"] = icc
            if dpi:
                save_kwargs["dpi"] = dpi
            if save_kwargs:
                temp = destination.with_name(
                    destination.stem + "_metadata_tmp" + destination.suffix
                )
                try:
                    processed.save(temp, format=processed.format, **save_kwargs)
                    os.replace(temp, destination)
                finally:
                    if temp.exists():
                        temp.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def run_enhance(source: Path, output: Path, mode: str):
    """Ejecuta mejora de fotos."""
    if not _ensure_cv2():
        return

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not _validate_source_output(source, output):
        return
    images = find_images(source, exclude=output)
    if not images:
        print("  No se encontraron imágenes.")
        return

    # [v1.3 H6] Pre-descargar modelo UNA vez antes del loop
    if mode.startswith("superres"):
        model_name = "espcn_x2" if mode == "superres_espcn" else "fsrcnn_x2"
        model_path = download_model(model_name)
        if not model_path:
            print("  ✗ No se pudo obtener el modelo. Abortando.")
            return

    output.mkdir(parents=True, exist_ok=True)
    print(f"  {len(images)} imágenes a procesar\n")

    ok, errors, metadata_warnings = 0, 0, 0
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        try:
            rel = img_path.relative_to(source)
        except ValueError:
            rel = Path(img_path.name)
        dst = output / rel
        # La salida debe ser distinta del origen; nunca sobrescribir un archivo.
        dst.parent.mkdir(parents=True, exist_ok=True)
        if str(dst) == str(img_path) or dst.resolve() == img_path.resolve():
            print(f"  ✗ Ruta de salida coincide con el original: {img_path}")
            errors += 1
            continue
        if not mode.startswith("superres") and dst.exists():
            stem, suffix, c = dst.stem, dst.suffix, 1
            while dst.exists():
                dst = dst.with_name(f"{stem}_{c}{suffix}")
                c += 1

        try:
            if mode.startswith("superres"):
                model = "espcn_x2" if mode == "superres_espcn" else "fsrcnn_x2"
                # [FIX C3] Anti-colisión con base fija fuera del loop
                base = dst.with_stem(dst.stem + "_x2")
                dst = base
                c = 1
                while dst.exists():
                    dst = base.parent / f"{base.stem}_{c}{base.suffix}"
                    c += 1
                enhanced_path = enhance_superres(
                    img_path, dst, model_name=model, model_path=model_path
                )
                if not enhanced_path:
                    errors += 1
                    continue
                if not _restore_metadata(img_path, enhanced_path):
                    metadata_warnings += 1
            else:
                # Evitar convertir silenciosamente GIF animados o TIFF multipágina.
                with Image.open(img_path) as check_img:
                    if getattr(check_img, "n_frames", 1) > 1:
                        raise ValueError("imagen animada/multipágina: omitida")
                # [FIX #7] Leer con fallback a Pillow
                img = _read_image_cv2(img_path)
                if img is None:
                    errors += 1
                    continue

                if mode == "auto":
                    img = enhance_auto(img)
                elif mode == "clahe":
                    img = enhance_clahe(img)
                elif mode == "denoise":
                    img = enhance_denoise(img, strength=8)
                elif mode == "gamma":
                    img = enhance_gamma(img)
                elif mode == "sharpen":
                    img = enhance_sharpen(img, amount=0.6)

                written = _write_image_cv2(img, dst)
                if not written:
                    print(f"  ⚠ No se pudo escribir: {dst.name}")
                    errors += 1
                    continue
                if not _restore_metadata(img_path, written):
                    metadata_warnings += 1

            ok += 1
        except Exception as e:
            print(f"  ⚠ {img_path.name}: {e}")
            errors += 1

        if i % 10 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}] {i * 100 // len(images)}%", end="\r")

    elapsed = time.time() - t0
    print(f"\n\n  ✓ {ok} mejoradas | {errors} errores | {elapsed:.1f}s")
    if metadata_warnings:
        print(f"  ⚠ {metadata_warnings} archivo(s) sin metadatos restaurados (EXIF/ICC/DPI)")
    print(f"  ✓ Resultado en: {output}")


# ══════════════════════════════════════════════════
# 4. ORGANIZAR POR FECHA
# ══════════════════════════════════════════════════


def run_organize(source: Path, output: Path, move: bool = False):
    """Organiza fotos en carpetas Año/Mes."""
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not _validate_source_output(source, output):
        return
    images = find_images(source, exclude=output)
    if not images:
        print("  No se encontraron imágenes.")
        return

    output.mkdir(parents=True, exist_ok=True)
    organized = 0
    no_date = 0

    for i, img_path in enumerate(images, 1):
        date = get_photo_date(img_path)
        if date:
            # [FIX #22] Validar año razonable
            year = date.year
            if year < 1900 or year > 2100:
                folder = output / "Sin_fecha"
                no_date += 1
            else:
                folder = output / str(year) / MESES.get(date.month, f"{date.month:02d}")
        else:
            folder = output / "Sin_fecha"
            no_date += 1

        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / img_path.name

        # [FIX #14] Skip si dst == src (ya organizado)
        if str(dst) == str(img_path) or dst.resolve() == img_path.resolve():
            organized += 1
            continue

        # Colisión
        if dst.exists():
            stem, sfx, c = dst.stem, dst.suffix, 1
            while dst.exists():
                dst = dst.parent / f"{stem}_{c}{sfx}"
                c += 1

        try:
            if move:
                shutil.move(str(img_path), str(dst))
            else:
                shutil.copy2(str(img_path), str(dst))
            organized += 1
        except Exception as e:
            print(f"  ⚠ {img_path.name}: {e}")

        if i % 50 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}]", end="\r")

    action = "movidas" if move else "copiadas"
    print(f"\n\n  ✓ {organized} fotos {action}")
    if no_date:
        print(f"  ℹ {no_date} sin fecha EXIF → Sin_fecha/")
    print(f"  ✓ Resultado en: {output}")

    # Limpiar directorios vacíos si movimos
    if move:
        for dp, _, _ in os.walk(source, topdown=False):
            p = Path(dp)
            if p != source and p != output:
                try:
                    if not any(p.iterdir()):
                        p.rmdir()
                except Exception:
                    pass


# ══════════════════════════════════════════════════
# 5. DETECTAR FOTOS BORROSAS
# ══════════════════════════════════════════════════


def blur_score(filepath: Path) -> float | None:
    """Varianza del Laplaciano — menor = más borrosa."""
    if not _CV2:
        # Fallback con Pillow
        # [v1.3 H5] Verificar numpy antes de usarlo
        if not _NUMPY:
            # Sin numpy ni cv2: varianza streaming (sin materializar lista)
            try:
                with Image.open(filepath) as img:
                    gray = img.convert("L")
                    edges = gray.filter(ImageFilter.FIND_EDGES)
                    n = s = s2 = 0
                    for p in edges.getdata():
                        n += 1
                        s += p
                        s2 += p * p
                    if n == 0:
                        return None
                    mean = s / n
                    return float(s2 / n - mean * mean)
            except Exception:
                return None
        try:
            with Image.open(filepath) as img:
                gray = img.convert("L")
                edges = gray.filter(ImageFilter.FIND_EDGES)
                arr = np.array(edges, dtype="float64")
                return float(arr.var())
        except Exception:
            return None
    try:
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback Pillow para HEIC
            try:
                with Image.open(filepath) as pil:
                    img = np.array(pil.convert("L"))
            except Exception:
                return None
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception:
        return None


# [FIX #5] Anti-colisión con while loop en move_blurry
def run_blur_detect(source: Path, threshold: float = 100.0,
                    move_blurry: bool = False):
    """Detecta y opcionalmente separa fotos borrosas."""
    source = source.expanduser().resolve()
    images = find_images(source, exclude=source / "_BORROSAS")
    if not images:
        print("  No se encontraron imágenes.")
        return

    print(f"  Threshold: {threshold} (menor = más borrosa)")
    print(f"  Analizando {len(images)} imágenes...\n")

    blurry = []
    sharp = []
    # [v1.3 H4] Rastrear archivos no analizables (score=None)
    skipped = []

    for i, img_path in enumerate(images, 1):
        score = blur_score(img_path)
        if score is None:
            skipped.append(img_path)
        elif score < threshold:
            blurry.append((img_path, score))
        else:
            sharp.append((img_path, score))

        if i % 30 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}]", end="\r")

    blurry.sort(key=lambda x: x[1])
    print(f"\n\n  Nítidas: {len(sharp)} | Borrosas: {len(blurry)}", end="")
    if skipped:
        print(f" | No analizadas: {len(skipped)}")
    else:
        print()

    # [v1.3 M10] Mapeo de paths movidos para actualizar el CSV
    moved_paths = {}  # original_path → new_path

    if blurry:
        print(f"\n  Top 10 más borrosas:")
        for path, score in blurry[:10]:
            print(f"    {score:8.1f}  {path.name}")

        if move_blurry:
            blur_dir = source / "_BORROSAS"
            # [v1.3 M9] Verificar que blur_dir no sea un archivo
            if blur_dir.exists() and not blur_dir.is_dir():
                print(f"  ✗ {blur_dir} existe pero no es un directorio.")
            else:
                blur_dir.mkdir(exist_ok=True)
                moved = 0
                for path, _ in blurry:
                    try:
                        dst = blur_dir / path.name
                        c = 0
                        while dst.exists():
                            dst = blur_dir / f"{path.stem}_{c}{path.suffix}"
                            c += 1
                        shutil.move(str(path), str(dst))
                        moved_paths[path] = dst
                        moved += 1
                    except Exception:
                        pass
                print(f"\n  ✓ {moved} borrosas movidas a {blur_dir}")

    # [FIX v1.5+] Reporte CSV en _BORROSAS (si existe) o junto a source, nunca dentro de source
    if move_blurry and blurry:
        report = (source / "_BORROSAS") / "blur_reporte.csv"
    else:
        report = source.parent / f"blur_reporte_{source.name}.csv"
    try:
        with open(report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "score", "estado"])
            for path, score in blurry:
                # Usar path post-move si fue movido
                actual = moved_paths.get(path, path) if move_blurry else path
                w.writerow([str(actual), f"{score:.1f}", "BORROSA"])
            for path, score in sharp:
                w.writerow([str(path), f"{score:.1f}", "NITIDA"])
            # [v1.3 H4] Incluir archivos no analizados en el reporte
            for path in skipped:
                w.writerow([str(path), "", "NO_ANALIZADA"])
        print(f"  ✓ Reporte: {report}")
    except Exception as e:
        print(f"  ⚠ No se pudo escribir reporte: {e}")


# ══════════════════════════════════════════════════
# 6. COMPRIMIR / CONVERTIR
# ══════════════════════════════════════════════════


def run_compress(source: Path, output: Path, fmt: str = "heif",
                 quality: int = 85, keep_structure: bool = True):
    """Comprime/convierte imágenes a otro formato."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not _validate_source_output(source, output):
        return
    images = find_images(source, exclude=output)
    if not images:
        print("  No se encontraron imágenes.")
        return

    if fmt == "heif" and not _HEIF:
        print("  ✗ HEIF no disponible. Instala: pip install pillow-heif")
        return

    output.mkdir(parents=True, exist_ok=True)
    print(f"  {len(images)} imágenes → {fmt.upper()} (q={quality})\n")

    orig_total, conv_total = 0, 0
    ok, errors, heic_copied = 0, 0, 0

    ext_map = {"heif": ".heif", "webp": ".webp", "jpeg": ".jpg"}
    target_ext = ext_map.get(fmt, ".jpg")

    for i, img_path in enumerate(images, 1):
        if keep_structure:
            try:
                rel = img_path.relative_to(source)
            except ValueError:
                rel = Path(img_path.name)
            dst = output / rel.with_suffix(target_ext)
        else:
            dst = output / (img_path.stem + target_ext)

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Colisión
        if dst.exists():
            stem, sfx, c = dst.stem, dst.suffix, 1
            while dst.exists():
                dst = dst.parent / f"{stem}_{c}{sfx}"
                c += 1

        orig_size = img_path.stat().st_size
        orig_total += orig_size

        try:
            if fmt == "heif":
                actual, reason = _convert_heif(img_path, dst, quality)
                if reason == "heic":
                    heic_copied += 1
                if not actual:
                    errors += 1
                    continue
                dst = actual  # usar path real (puede diferir si hubo fallback)
            else:
                with Image.open(img_path) as opened:
                    if getattr(opened, "n_frames", 1) > 1:
                        raise ValueError("imagen animada/multipágina: omitida para no perder fotogramas")
                    img = opened.copy()
                    exif = opened.info.get("exif")
                    icc = opened.info.get("icc_profile")
                    dpi = opened.info.get("dpi")
                    kw = {"quality": quality}
                    if exif:
                        kw["exif"] = exif
                    if icc:
                        kw["icc_profile"] = icc
                    if dpi:
                        kw["dpi"] = dpi
                    if fmt == "webp":
                        # WebP admite transparencia: conservar RGBA/LA.
                        if img.mode not in ("RGB", "RGBA"):
                            img = img.convert("RGBA" if "transparency" in opened.info else "RGB")
                        kw.update({"format": "WEBP", "method": 4})
                    else:
                        # JPEG no admite alfa; componer explícitamente sobre blanco.
                        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in opened.info):
                            rgba = img.convert("RGBA")
                            background = Image.new("RGB", rgba.size, (255, 255, 255))
                            background.paste(rgba, mask=rgba.getchannel("A"))
                            img = background
                        elif img.mode != "RGB":
                            img = img.convert("RGB")
                        kw.update({"format": "JPEG", "optimize": True})
                    img.save(str(dst), **kw)

            conv_total += dst.stat().st_size
            ok += 1
        except Exception as e:
            print(f"  ⚠ {img_path.name}: {e}")
            errors += 1

        if i % 20 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}] {i * 100 // len(images)}%", end="\r")

    print(f"\n\n  Original:  {format_size(orig_total)}")
    print(f"  {fmt.upper()}:     {format_size(conv_total)}")
    if orig_total:
        print(f"  Reducción: {(1 - conv_total / orig_total) * 100:.0f}%")
    print(f"  ✓ {ok} convertidas | {errors} errores")
    if heic_copied:
        print(f"  ℹ {heic_copied} archivo(s) HEIC copiado(s) sin recomprimir (ya son HEIF)")
    print(f"  ✓ Resultado en: {output}")


# ══════════════════════════════════════════════════
# 7. ESTADÍSTICAS
# ══════════════════════════════════════════════════


def run_stats(source: Path):
    """Muestra estadísticas; distingue archivos encontrados de imágenes legibles."""
    source = source.expanduser().resolve()
    images = find_images(source, warn_raw=False)

    raw_files = []
    for dirpath, _, filenames in os.walk(source):
        for fname in filenames:
            if Path(fname).suffix.lower() in RAW_EXTENSIONS:
                raw_files.append(Path(dirpath) / fname)

    if not images and not raw_files:
        print("  No se encontraron imágenes ni archivos RAW.")
        return

    total_size = 0
    formats = defaultdict(int)
    res_buckets = {"< 2 MP": 0, "2-8 MP": 0, "8-20 MP": 0, "> 20 MP": 0}
    dates = []
    folders = set()
    readable_count = 0
    unreadable = []

    for i, img_path in enumerate(images, 1):
        # 1) stat: puede fallar por permisos / archivo borrado
        try:
            size = img_path.stat().st_size
        except OSError:
            unreadable.append(img_path)
            continue
        total_size += size
        formats[img_path.suffix.lower()] += 1
        folders.add(str(img_path.parent))

        # 2) validación de cabecera rápida con verify(); no decodifica píxeles.
        try:
            with Image.open(img_path) as check:
                check.verify()
        except Exception:
            unreadable.append(img_path)
            continue
        readable_count += 1

        # 3) metadata (resolución y fecha) sin forzar decode completo.
        try:
            with Image.open(img_path) as img:
                mp = (img.size[0] * img.size[1]) / 1e6
                if mp < 2:
                    res_buckets["< 2 MP"] += 1
                elif mp < 8:
                    res_buckets["2-8 MP"] += 1
                elif mp < 20:
                    res_buckets["8-20 MP"] += 1
                else:
                    res_buckets["> 20 MP"] += 1

                date_added = False
                try:
                    exif = img.getexif()
                    if exif:
                        for tag_id in (36867, 36868, 306):
                            val = exif.get(tag_id)
                            if isinstance(val, str) and val:
                                try:
                                    dates.append(datetime.strptime(val, "%Y:%m:%d %H:%M:%S"))
                                    date_added = True
                                    break
                                except ValueError:
                                    continue
                except Exception:
                    pass
                if not date_added:
                    try:
                        dates.append(datetime.fromtimestamp(img_path.stat().st_mtime))
                    except (OSError, OverflowError, ValueError):
                        pass
        except Exception:
            # Cabecera OK pero metadata ilegible: no es "unreadable" total,
            # pero no podemos bucketizar. Lo dejamos fuera de resolución/fecha.
            pass

        if i % 100 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}]", end="\r")

    raw_total_size = 0
    for rp in raw_files:
        try:
            sz = rp.stat().st_size
            raw_total_size += sz
            total_size += sz
            formats[rp.suffix.lower()] += 1
            folders.add(str(rp.parent))
        except OSError:
            pass

    total_count = len(images) + len(raw_files)
    banner(f"ESTADÍSTICAS: {source.name}")
    print(f"  Total de archivos: {total_count:,}")
    print(f"    Imágenes: {len(images):,} | Legibles: {readable_count:,} | RAW: {len(raw_files):,}")
    if unreadable:
        print(f"    No accesibles/ilegibles: {len(unreadable):,}")
        for path in unreadable[:5]:
            print(f"      - {path.name}")
        if len(unreadable) > 5:
            print(f"      ... y {len(unreadable) - 5} más")
    print(f"  Tamaño total:      {format_size(total_size)}")
    if raw_files:
        print(f"    (RAW: {format_size(raw_total_size)})")
    print(f"  Carpetas:          {len(folders)}")
    print()

    print("  Formatos:")
    for ext, count in sorted(formats.items(), key=lambda x: -x[1]):
        pct = count * 100 / max(total_count, 1)
        print(f"    {ext:8s} {count:6d}  ({pct:.0f}%)")
    print()

    if readable_count:
        print("  Resoluciones (sobre imágenes legibles):")
        for label, count in res_buckets.items():
            bar = "█" * (count * 30 // max(readable_count, 1))
            print(f"    {label:8s} {count:6d}  {bar}")
        print()
    elif images:
        print("  Resoluciones: no disponibles (ninguna imagen pudo leerse).")
    else:
        print("  Resoluciones: no disponibles (solo archivos RAW).")

    valid_dates = sorted(d for d in dates if 1900 <= d.year <= 2100)
    if valid_dates:
        print(f"  Rango de fechas: {valid_dates[0].strftime('%Y-%m-%d')}"
              f" → {valid_dates[-1].strftime('%Y-%m-%d')}")
        years = defaultdict(int)
        for d in valid_dates:
            years[d.year] += 1
        print("  Por año:")
        for year in sorted(years):
            bar = "█" * (years[year] * 30 // max(max(years.values()), 1))
            print(f"    {year}  {years[year]:6d}  {bar}")
    elif raw_files and not images:
        print("  ℹ Fechas no calculadas (solo archivos RAW encontrados)")


# ══════════════════════════════════════════════════
# MENÚ INTERACTIVO
# ══════════════════════════════════════════════════


def menu_dedup():
    banner("DEDUPLICAR FOTOS")
    source = ask_folder("Carpeta a analizar")
    if not source:
        return
    output = ask_folder("Carpeta de salida (Enter = auto)", must_exist=False)
    if not output:
        output = source / "FOTOS_DEDUP"
    threshold = ask_int("Threshold similitud (6=normal, 3=estricto)", 6, min_val=1, max_val=20)
    no_convert = not _HEIF
    if _HEIF:
        no_convert = not ask_confirm("¿Convertir a HEIF?", default_yes=True)
    quality = ask_int("Calidad (85=recomendado)", 85, min_val=1, max_val=100) if not no_convert else 85
    keep = ask_confirm("¿Conservar originales?", default_yes=True)
    dry_run = ask_confirm("¿Solo simular (dry run)?", default_yes=False)

    run_dedup(source, output, threshold=threshold, no_convert=no_convert,
              quality=quality, keep_originals=keep, dry_run=dry_run)


def menu_compare():
    banner("COMPARAR DOS CARPETAS")
    source = ask_folder("Carpeta A")
    if not source:
        return
    compare = ask_folder("Carpeta B")
    if not compare:
        return
    output = ask_folder("Carpeta de salida (Enter = auto)", must_exist=False)
    if not output:
        output = source / "COMPARACION"
    threshold = ask_int("Threshold similitud", 6, min_val=1, max_val=20)
    no_convert = not _HEIF
    if _HEIF:
        no_convert = not ask_confirm("¿Convertir a HEIF?", default_yes=True)
    quality = ask_int("Calidad", 85, min_val=1, max_val=100) if not no_convert else 85
    keep = ask_confirm("¿Conservar originales?", default_yes=True)

    run_dedup(source, output, compare=compare, threshold=threshold,
              no_convert=no_convert, quality=quality, keep_originals=keep)


def menu_enhance():
    banner("MEJORAR FOTOS (OpenCV / red neuronal)")
    if not _CV2:
        print("  ✗ Requiere OpenCV. Instala:")
        print("    pip install opencv-contrib-python==4.10.0.84")
        return

    source = ask_folder("Carpeta de fotos")
    if not source:
        return
    output = ask_folder("Carpeta de salida (Enter = auto)", must_exist=False)
    if not output:
        output = source / "MEJORADAS"

    sr_tag = " (IA)" if _SUPERRES else " (no disponible)"
    print(f"""
  Modos de mejora:
    1. Auto-enhance (contraste + color + nitidez)
    2. Super-resolución neuronal FSRCNN ×2{sr_tag}
    3. Super-resolución neuronal ESPCN ×2{sr_tag}
    4. Reducir ruido
    5. Corregir contraste (CLAHE)
    6. Corregir iluminación (gamma)
    7. Aumentar nitidez (sharpen)
""")
    opt = ask("Modo", "1")
    mode_map = {
        "1": "auto", "2": "superres_fsrcnn", "3": "superres_espcn",
        "4": "denoise", "5": "clahe", "6": "gamma", "7": "sharpen",
    }
    mode = mode_map.get(opt)
    if not mode:
        print("  Opción no válida.")
        return

    if mode.startswith("superres") and not _SUPERRES:
        print("  ✗ dnn_superres no disponible. Instala opencv-contrib-python==4.10.0.84")
        return

    run_enhance(source, output, mode)


def menu_organize():
    banner("ORGANIZAR POR FECHA")
    source = ask_folder("Carpeta de fotos")
    if not source:
        return
    output = ask_folder("Carpeta de salida (Enter = auto)", must_exist=False)
    if not output:
        output = source / "ORGANIZADAS"
    move = ask_confirm("¿Mover archivos? (No = copiar)", default_yes=False)
    run_organize(source, output, move=move)


# [FIX #8] menu_blur con try/except en float
def menu_blur():
    banner("DETECTAR FOTOS BORROSAS")
    source = ask_folder("Carpeta de fotos")
    if not source:
        return
    raw_threshold = ask("Threshold (100=normal, 50=estricto)", "100")
    try:
        threshold = float(raw_threshold)
        if threshold <= 0 or not math.isfinite(threshold):
            threshold = 100.0
    except (ValueError, OverflowError):
        print("  ⚠ Valor inválido, usando 100.0")
        threshold = 100.0
    move = ask_confirm("¿Mover borrosas a carpeta aparte?", default_yes=False)
    run_blur_detect(source, threshold=threshold, move_blurry=move)


def menu_compress():
    banner("COMPRIMIR / CONVERTIR")
    source = ask_folder("Carpeta de fotos")
    if not source:
        return
    output = ask_folder("Carpeta de salida (Enter = auto)", must_exist=False)
    if not output:
        output = source / "COMPRIMIDAS"

    heif_tag = " (recomendado)" if _HEIF else " (no disponible)"
    print(f"""
  Formato:
    1. HEIF{heif_tag} — ~50% reducción
    2. WebP — ~30% reducción
    3. JPEG optimizado — ~15% reducción
""")
    opt = ask("Formato", "1" if _HEIF else "2")
    fmt_map = {"1": "heif", "2": "webp", "3": "jpeg"}
    fmt = fmt_map.get(opt)
    if not fmt:
        print("  Opción no válida.")
        return
    if fmt == "heif" and not _HEIF:
        print("  ✗ HEIF no disponible. Instala: pip install pillow-heif")
        return
    quality = ask_int("Calidad (85=recomendado)", 85, min_val=1, max_val=100)
    run_compress(source, output, fmt=fmt, quality=quality)


def menu_stats():
    banner("ESTADÍSTICAS")
    source = ask_folder("Carpeta de fotos")
    if not source:
        return
    run_stats(source)


def main_menu():
    while True:
        cls()
        deps = []
        if _HEIF:
            deps.append("HEIF")
        if _CV2:
            deps.append("OpenCV")
        if _SUPERRES:
            deps.append("SuperRes-IA")
        dep_str = " + ".join(deps) if deps else "solo Pillow"

        print(f"""
╔══════════════════════════════════════════════╗
║         PHOTO MANAGER CLI v{VERSION:<18s}║
╠══════════════════════════════════════════════╣
║                                              ║
║  1. Deduplicar fotos                         ║
║  2. Comparar dos carpetas                    ║
║  3. Mejorar fotos (OpenCV / red neuronal)    ║
║  4. Organizar por fecha                      ║
║  5. Detectar fotos borrosas                  ║
║  6. Comprimir / Convertir                    ║
║  7. Estadísticas de carpeta                  ║
║                                              ║
║  0. Salir                                    ║
║                                              ║
╠══════════════════════════════════════════════╣
║  Módulos: {dep_str:<34s}║
╚══════════════════════════════════════════════╝
""")
        opt = input("  Opción: ").strip()

        if opt == "0":
            print("\n  Hasta luego.\n")
            break
        elif opt == "1":
            menu_dedup()
        elif opt == "2":
            menu_compare()
        elif opt == "3":
            menu_enhance()
        elif opt == "4":
            menu_organize()
        elif opt == "5":
            menu_blur()
        elif opt == "6":
            menu_compress()
        elif opt == "7":
            menu_stats()
        else:
            print("  Opción no válida.")

        if opt in ("1", "2", "3", "4", "5", "6", "7"):
            print()
            input("  Enter para volver al menú...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  Interrumpido. Hasta luego.\n")
