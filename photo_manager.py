#!/usr/bin/env python3
"""
Photo Manager CLI v1.15.0
Herramienta de gestión de fotos.

Funcionalidades:
  1. Deduplicar fotos (pHash + BK-tree + complete-link, caché SQLite)
  2. Comparar dos carpetas en busca de duplicados cruzados
  3. Mejorar fotos:
       - Automático: analiza cada foto y corrige solo lo necesario
         (super-resolución hasta 4K con Real-ESRGAN, ruido, color, rango,
         HDR fuerte en fotos oscuras, nitidez)
       - Real-ESRGAN: amplía hasta 4K (máximo ×4)
       - HDR: aclara sombras sin quemar luces (suave / normal / fuerte)
       - Diagnóstico: reporte de lo que haría el modo automático
  4. Detectar fotos borrosas (varianza del Laplaciano, resolución normalizada)
  5. Comprimir / Convertir (HEIF, WebP, JPEG; recomprime HEIC; reduce resolución)

Dependencias:
  pip install Pillow imagehash pillow-heif opencv-contrib-python==4.10.0.84

Real-ESRGAN (opcional, sin PyTorch ni CUDA; usa GPU vía Vulkan):
  Descarga realesrgan-ncnn-vulkan desde
  https://github.com/xinntao/Real-ESRGAN/releases y descomprímelo en una
  carpeta "realesrgan" junto a este script (o en ~/.photo_manager/realesrgan/),
  o indica la ruta del ejecutable en la variable PHOTO_MANAGER_REALESRGAN.

Las carpetas de salida se marcan con el archivo .photo_manager_output y se
ignoran en los escaneos, para que nunca se comparen con los originales.

Autor: Josmar Angulo / Claude
"""

import csv
import hashlib
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
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
    from PIL import Image, ImageFilter, ImageOps
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

# ══════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════

VERSION = "1.15.0"

# [FIX #12] Separar formatos soportados por Pillow vs RAW
PILLOW_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".gif",
}
RAW_EXTENSIONS = {".raw", ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2"}
IMAGE_EXTENSIONS = PILLOW_EXTENSIONS | RAW_EXTENSIONS

# Real-ESRGAN (binario ncnn-vulkan externo)
REALESRGAN_ENV = "PHOTO_MANAGER_REALESRGAN"
REALESRGAN_MODEL = "realesrgan-x4plus"      # modelo para fotos reales
REALESRGAN_DIR = Path.home() / ".photo_manager" / "realesrgan"
SCRIPT_DIR = Path(__file__).resolve().parent

# Caché de hashes (idea tomada de czkawka): evita re-decodificar y re-hashear
CACHE_PATH = Path.home() / ".photo_manager" / "hash_cache.sqlite"
CACHE_ALGO = 2   # subir si cambia el cálculo del pHash/fecha → invalida caché

# Lado máximo al que se reduce la imagen para medir borrosidad
BLUR_MAX_SIDE = 1000
# Tamaño mínimo pedido a JPEG draft() antes del pHash (pHash usa 4×hash_size)
DRAFT_SIZE = 256

# Carpetas de salida: nunca se escanean como origen
OUTPUT_MARKER = ".photo_manager_output"
MANAGED_DIR_NAMES = {
    "FOTOS_DEDUP", "COMPARACION", "MEJORADAS", "ORGANIZADAS",
    "COMPRIMIDAS", "_BORROSAS",
}

# Etiquetas EXIF
EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME_DIGITIZED = 36868
TAG_DATETIME = 306            # IFD0: fecha de última modificación/edición
TAG_ORIENTATION = 274

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


def clean_path_input(raw: str) -> str:
    """Normaliza una ruta pegada o arrastrada a la terminal.

    Quita comillas envolventes ('...' o "..."), el prefijo file:// y los
    escapes con barra invertida ('Mis\\ Fotos') que añaden los gestores de
    archivos y terminales de Linux/macOS al arrastrar una carpeta.
    """
    s = raw.strip()
    if s.startswith("& "):           # PowerShell al arrastrar: & 'C:\\ruta'
        s = s[2:].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    elif os.name != "nt" and "\\" in s:
        s = re.sub(r"\\(.)", r"\1", s)
    if s.startswith("file://"):
        from urllib.parse import unquote, urlparse
        s = unquote(urlparse(s).path)
    return s


def ask_folder(prompt: str, must_exist: bool = True) -> Path | None:
    raw = clean_path_input(ask(prompt))
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


def _unique_path(path: Path) -> Path:
    """Devuelve path, o path_1, path_2... si ya existe."""
    if not path.exists():
        return path
    stem, sfx, c = path.stem, path.suffix, 1
    while True:
        candidate = path.with_name(f"{stem}_{c}{sfx}")
        if not candidate.exists():
            return candidate
        c += 1


def _prepare_output_dir(output: Path) -> None:
    """Crea la carpeta de salida y la marca para excluirla de futuros escaneos."""
    output.mkdir(parents=True, exist_ok=True)
    marker = output / OUTPUT_MARKER
    if not marker.exists():
        try:
            marker.write_text(
                "Carpeta generada por Photo Manager. Se ignora al escanear.\n",
                encoding="utf-8")
        except OSError:
            pass


def _is_managed_dir(path: Path) -> bool:
    """Carpeta de salida de Photo Manager (por nombre conocido o marcador)."""
    if path.name in MANAGED_DIR_NAMES:
        return True
    try:
        return (path / OUTPUT_MARKER).exists()
    except OSError:
        return False


def _walk_files(root: Path, exclude: Path | None = None,
                skip_managed: bool = True, stats: dict | None = None):
    """os.walk que omite `exclude` y las carpetas de salida de Photo Manager.

    Produce (Path directorio, nombres de archivo). La raíz nunca se omite,
    así que se puede analizar una carpeta de salida apuntando a ella.
    """
    root = root.expanduser().resolve()
    excluded = exclude.expanduser().resolve() if exclude else None

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # os.walk(followlinks=False) no sigue symlinks: dirpath ya está
        # bajo root resuelto. Evitamos resolve() por directorio (FS call).
        dp = Path(dirpath)
        if excluded and (dp == excluded or excluded in dp.parents):
            dirnames[:] = []
            continue
        kept = []
        for name in dirnames:
            child = dp / name
            if excluded and (child == excluded or excluded in child.parents):
                continue
            if skip_managed and _is_managed_dir(child):
                if stats is not None:
                    stats.setdefault("managed", []).append(child)
                continue
            kept.append(name)
        dirnames[:] = kept
        yield dp, filenames


def find_images(root: Path, exclude: Path | None = None,
                warn_raw: bool = True, skip_managed: bool = True) -> list[Path]:
    """Encuentra imágenes recursivamente.

    Excluye `exclude` y, por defecto, cualquier carpeta de salida de Photo
    Manager (MEJORADAS, ORGANIZADAS... o marcada con .photo_manager_output):
    así una copia generada nunca compite con su original en la deduplicación.
    """
    images, raw_count, stats = [], 0, {}
    for dp, filenames in _walk_files(root, exclude, skip_managed, stats):
        for fname in filenames:
            suffix = Path(fname).suffix.lower()
            if suffix in RAW_EXTENSIONS:
                raw_count += 1
            elif suffix in PILLOW_EXTENSIONS:
                images.append(dp / fname)

    managed = stats.get("managed", [])
    if managed and warn_raw:
        names = ", ".join(sorted({m.name for m in managed})[:4])
        print(f"  ℹ {len(managed)} carpeta(s) de salida omitida(s): {names}")
    if raw_count and warn_raw:
        print(f"  ⚠ {raw_count} archivos RAW ignorados (requiere rawpy)")
    return sorted(images)


# ── Fechas ────────────────────────────────────────

_FILENAME_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"
    r"(?:[ _T.-]+(?:at[ _])?([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d))?",
    re.IGNORECASE,
)


def _valid_year(dt: datetime | None) -> datetime | None:
    return dt if dt and 1900 <= dt.year <= 2100 else None


def _parse_exif_datetime(val) -> datetime | None:
    if isinstance(val, bytes):
        val = val.decode("ascii", errors="ignore")
    if not isinstance(val, str):
        return None
    val = val.strip().rstrip("\x00")
    try:
        return _valid_year(datetime.strptime(val[:19], "%Y:%m:%d %H:%M:%S"))
    except ValueError:
        return None


def _date_from_filename(name: str) -> datetime | None:
    """Fecha en nombres tipo IMG_20190505_123456, IMG-20190505-WA0001,
    'WhatsApp Image 2019-05-05 at 12.34.56', Screenshot_2019-05-05-12-34-56."""
    for m in _FILENAME_DATE_RE.finditer(name):
        y, mo, d, hh, mi, ss = m.groups()
        try:
            if hh:
                dt = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss))
            else:
                dt = datetime(int(y), int(mo), int(d))
        except ValueError:
            continue          # p. ej. 30 de febrero
        if dt <= datetime.now():
            return dt
    return None


def _exif_dates(exif) -> tuple[datetime | None, datetime | None]:
    """(fecha de captura, fecha IFD0). DateTimeOriginal/Digitized viven en el
    sub-IFD Exif (0x8769), no en el IFD0 que devuelve getexif()."""
    if not exif:
        return None, None
    try:
        sub = exif.get_ifd(EXIF_IFD)
    except Exception:
        sub = {}
    capture = None
    for tag in (TAG_DATETIME_ORIGINAL, TAG_DATETIME_DIGITIZED):
        capture = _parse_exif_datetime(sub.get(tag)) or _parse_exif_datetime(exif.get(tag))
        if capture:
            break
    return capture, _parse_exif_datetime(exif.get(TAG_DATETIME))


def _mtime_date(filepath: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(filepath.stat().st_mtime)
    except (OSError, OverflowError, ValueError):
        return None


def resolve_photo_date(filepath: Path, exif=None) -> datetime | None:
    """Prioridad: EXIF captura → nombre de archivo → EXIF DateTime → mtime.

    DateTime (306) va después del nombre porque es la fecha de edición, y
    el mtime al final porque en copias/backups es la fecha de la copia.
    """
    capture, modified = _exif_dates(exif)
    return (capture or _date_from_filename(filepath.name)
            or modified or _mtime_date(filepath))


def get_photo_date(filepath: Path) -> datetime | None:
    """Fecha de la foto (ver resolve_photo_date)."""
    exif = None
    try:
        with Image.open(filepath) as img:
            exif = img.getexif()
    except Exception:
        pass
    return resolve_photo_date(filepath, exif)


class HashCache:
    """Caché SQLite de pHash + SHA-256 + metadatos por (ruta, tamaño, mtime).

    Si el archivo no cambió (mismo tamaño y mtime_ns), se reutiliza el
    resultado sin abrirlo. Si SQLite falla, la caché se desactiva sola.
    """

    MULTIPAGE = "multipage"

    def __init__(self, path: Path | None = None):
        self.path = path or CACHE_PATH
        self.conn = None
        self.hits = self.misses = 0
        self._pending = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path))
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    path TEXT NOT NULL, hash_size INTEGER NOT NULL,
                    algo INTEGER NOT NULL, size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL, multipage INTEGER NOT NULL,
                    phash TEXT, file_hash TEXT, width INTEGER, height INTEGER,
                    date TEXT,
                    PRIMARY KEY (path, hash_size, algo))""")
        except (sqlite3.Error, OSError) as e:
            print(f"  ⚠ Caché de hashes desactivada: {e}")
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get(self, filepath: Path, st: os.stat_result, hash_size: int):
        if not self.conn:
            return None
        try:
            row = self.conn.execute(
                "SELECT size, mtime_ns, multipage, phash, file_hash, width, height, date "
                "FROM images WHERE path=? AND hash_size=? AND algo=?",
                (str(filepath), hash_size, CACHE_ALGO)).fetchone()
        except sqlite3.Error:
            return None
        if not row or row[0] != st.st_size or row[1] != st.st_mtime_ns:
            self.misses += 1
            return None
        self.hits += 1
        if row[2]:
            return self.MULTIPAGE
        _, _, _, phash, file_hash, w, h, date = row
        return {
            "path": filepath,
            "phash": imagehash.hex_to_hash(phash),
            "resolution": w * h, "width": w, "height": h,
            "date": datetime.fromisoformat(date),
            "file_size": st.st_size,
            "file_hash": file_hash,
        }

    def put(self, filepath: Path, st: os.stat_result, hash_size: int,
            info: dict | None, multipage: bool = False):
        if not self.conn:
            return
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (str(filepath), hash_size, CACHE_ALGO, st.st_size, st.st_mtime_ns,
                 int(multipage),
                 None if multipage else str(info["phash"]),
                 None if multipage else info["file_hash"],
                 None if multipage else info["width"],
                 None if multipage else info["height"],
                 None if multipage else info["date"].isoformat()))
            self._pending += 1
            if self._pending >= 200:
                self.conn.commit()
                self._pending = 0
        except sqlite3.Error:
            pass

    def close(self):
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
            except sqlite3.Error:
                pass
            self.conn = None


def _hash_task(args: tuple) -> dict:
    """Trabajo de un proceso: hash + metadatos de una foto (no lanza excepciones)."""
    path, hash_size = args
    skips: dict = {}
    try:
        st = path.stat()
    except OSError as e:
        return {"path": path, "st": None, "info": None,
                "skips": {"error": 1, "error_samples": [f"{path.name}: {e}"]}}
    info = get_image_info(path, hash_size, skip_counts=skips)
    return {"path": path, "st": st, "info": info, "skips": skips}


def _sha256_file(filepath: Path) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_image_info(filepath: Path, hash_size: int = 12,
                   skip_counts: dict | None = None,
                   cache: HashCache | None = None) -> dict | None:
    """Hash perceptual + metadata de una imagen.

    skip_counts, si se proporciona, acumula motivos de omisión para resumirlos.
    cache, si se proporciona, evita recalcular archivos sin cambios.
    """
    try:
        st = filepath.stat()
        if cache:
            cached = cache.get(filepath, st, hash_size)
            if cached == HashCache.MULTIPAGE:
                if skip_counts is not None:
                    skip_counts["multipage"] = skip_counts.get("multipage", 0) + 1
                return None
            if cached:
                return cached

        with Image.open(filepath) as img:
            if getattr(img, "n_frames", 1) > 1:
                if skip_counts is not None:
                    skip_counts["multipage"] = skip_counts.get("multipage", 0) + 1
                if cache:
                    cache.put(filepath, st, hash_size, None, multipage=True)
                return None
            w, h = img.size              # tamaño real, antes de draft()
            exif = img.getexif()
            if img.format == "JPEG":
                # Decodifica a 1/2, 1/4 u 1/8 directamente desde el DCT: el
                # pHash solo usa 4×hash_size px, así que no cambia el resultado
                # de forma apreciable y es varias veces más rápido.
                img.draft("RGB", (DRAFT_SIZE, DRAFT_SIZE))
            # Hash de la imagen tal como se ve (orientación EXIF aplicada)
            oriented = ImageOps.exif_transpose(img)
            phash = imagehash.phash(oriented.convert("RGB"), hash_size=hash_size)

        info = {
            "path": filepath,
            "phash": phash,
            "resolution": w * h,
            "width": w,
            "height": h,
            "date": resolve_photo_date(filepath, exif) or datetime.fromtimestamp(0),
            "file_size": st.st_size,
            "file_hash": _sha256_file(filepath),
        }
        if cache:
            cache.put(filepath, st, hash_size, info)
        return info
    except Exception as e:
        if skip_counts is not None:
            skip_counts["error"] = skip_counts.get("error", 0) + 1
            skip_counts.setdefault("error_samples", []).append(f"{filepath.name}: {e}")
        else:
            print(f"  ⚠ Error: {filepath.name} — {e}")
        return None


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


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Progress:
    """Indicador de avance en una sola línea: barra, %, tiempo y estimación.

    `sub` (0-1) permite mostrar el avance dentro de la foto actual, p. ej.
    los porcentajes que va informando Real-ESRGAN. Si la salida no es una
    terminal (redirigida a archivo), imprime solo cada 10 %.
    """

    def __init__(self, total: int, label: str = "", min_interval: float = 0.2):
        self.total = max(total, 1)
        self.label = label
        self.min_interval = min_interval
        self.t0 = time.time()
        self._last = 0.0
        self._last_decile = -1
        self.tty = sys.stdout.isatty()

    def update(self, done: int, current: str = "", sub: float | None = None,
               force: bool = False):
        now = time.time()
        if not force and now - self._last < self.min_interval and done < self.total:
            return
        self._last = now
        frac = (done + (sub or 0.0)) / self.total
        frac = min(max(frac, 0.0), 1.0)
        elapsed = now - self.t0
        eta = elapsed / frac - elapsed if frac > 0.01 else None
        width = 20
        filled = int(frac * width)
        bar = "█" * filled + "░" * (width - filled)
        parts = [f"  {self.label} [{done}/{self.total}] {bar} {frac * 100:3.0f}%",
                 _fmt_duration(elapsed)]
        if eta is not None and done < self.total:
            parts.append(f"quedan ~{_fmt_duration(eta)}")
        line = " | ".join(parts)
        if current:
            line += f" | {current}"
            if sub is not None and done < self.total:
                line += f" {sub * 100:.0f}%"
        if self.tty:
            cols = shutil.get_terminal_size((100, 20)).columns
            print("\r" + line[:cols - 1].ljust(cols - 1), end="", flush=True)
        else:
            decile = int(frac * 10)
            if decile != self._last_decile or done >= self.total:
                self._last_decile = decile
                print(line, flush=True)

    def finish(self, current: str = ""):
        self.update(self.total, current, force=True)
        if self.tty:
            print()


# ── Procesamiento en paralelo ─────────────────────

MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
PARALLEL_MIN_ITEMS = 4


def _worker_init():
    # Cada proceso usa 1 hilo de OpenCV: evita que 4 procesos × N hilos
    # compitan por los mismos núcleos.
    if _CV2:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass


def parallel_map(fn, items: list, label: str, workers: int | None = None,
                 describe=None):
    """Aplica fn a cada elemento usando varios procesos, con indicador de avance.

    Produce (índice, resultado) en orden de finalización. `fn` debe ser una
    función de módulo que NO lance excepciones (que devuelva el error como
    resultado). Si no se puede crear el pool de procesos, trabaja en serie.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = min(workers or MAX_WORKERS, max(len(items), 1))
    if len(items) < PARALLEL_MIN_ITEMS:
        workers = 1         # arrancar procesos cuesta más de lo que ahorra
    describe = describe or (lambda item: "")
    progress = Progress(len(items), label)

    pool = None
    if workers > 1:
        try:
            # "spawn": procesos nuevos. Con "fork" (defecto en Linux) un hijo
            # puede heredar un candado tomado por un hilo de OpenCV y quedarse
            # bloqueado para siempre.
            pool = ProcessPoolExecutor(
                max_workers=workers, initializer=_worker_init,
                mp_context=multiprocessing.get_context("spawn"))
        except (OSError, NotImplementedError, ImportError) as e:
            print(f"  ⚠ Sin procesamiento paralelo ({e}); se sigue en serie.")

    if pool is None:
        for i, item in enumerate(items):
            progress.update(i, describe(item), force=True)
            yield i, fn(item)
        progress.finish()
        return

    from concurrent.futures.process import BrokenProcessPool
    finished: set[int] = set()
    broken = None
    with pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        try:
            for fut in as_completed(futures):
                i = futures[fut]
                result = fut.result()
                finished.add(i)
                progress.update(len(finished), describe(items[i]))
                yield i, result
        except BrokenProcessPool as e:
            broken = e
            for f in futures:
                f.cancel()

    if broken is not None:
        # Un proceso murió (memoria insuficiente, entorno sin spawn...):
        # terminar en serie solo lo que falta, sin repetir lo ya hecho.
        print(f"\n  ⚠ Falló el procesamiento paralelo ({broken}); "
              "se continúa en serie.")
        for i, item in enumerate(items):
            if i in finished:
                continue
            progress.update(len(finished), describe(item), force=True)
            result = fn(item)
            finished.add(i)
            yield i, result
    progress.finish()


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
            icc = img.info.get("icc_profile")
            # [v1.3 C2] Detectar canal alfa — HEIF no soporta alfa
            has_alpha = (
                img.mode in ("RGBA", "LA", "PA", "La")
                or (img.mode == "P" and "transparency" in img.info)
            )
            if has_alpha:
                # No convertir a HEIF: perdería canal alfa → copiar en formato original
                fallback = _unique_path(dst.with_suffix(src.suffix))
                shutil.copy2(str(src), str(fallback))
                return (fallback, "alpha") if fallback.exists() else (None, None)
            if img.mode not in ("RGB",):
                img = img.convert("RGB")
            kw = {"format": "HEIF", "quality": quality}
            if exif:
                kw["exif"] = exif
            if icc:
                kw["icc_profile"] = icc
            img.save(str(dst), **kw)
            if dst.exists() and dst.stat().st_size > 0:
                return (dst, None)
            return (None, None)
    except Exception:
        # Fallback: copiar en formato original (path distinto a dst)
        try:
            fallback = _unique_path(dst.with_suffix(src.suffix))
            shutil.copy2(str(src), str(fallback))
            return (fallback, "exception") if fallback.exists() else (None, None)
        except Exception:
            return (None, None)


def run_dedup(source: Path, output: Path, compare: Path | None = None,
              hash_size: int = 12, threshold: int = 6,
              no_convert: bool = False, quality: int = 85,
              keep_originals: bool = True, dry_run: bool = False,
              use_cache: bool = True):
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
    cache = HashCache() if use_cache else None
    try:
        # 1) Caché (rápido, en el proceso principal)
        pending = []
        for img in images:
            cached = None
            if cache:
                try:
                    cached = cache.get(img, img.stat(), hash_size)
                except OSError:
                    cached = None
            if cached == HashCache.MULTIPAGE:
                skip_counts["multipage"] += 1
            elif cached:
                infos.append(cached)
            else:
                pending.append(img)
        # 2) Lo que falta, en paralelo (decodificar + pHash + SHA-256)
        tasks = [(img, hash_size) for img in pending]
        for _, res in parallel_map(_hash_task, tasks, "Hash",
                                   describe=lambda t: t[0].name):
            for key in ("multipage", "error"):
                skip_counts[key] += res["skips"].get(key, 0)
            skip_counts["error_samples"] += res["skips"].get("error_samples", [])
            if cache and res["st"] is not None and (res["info"] or res["skips"].get("multipage")):
                cache.put(res["path"], res["st"], hash_size, res["info"],
                          multipage=bool(res["skips"].get("multipage")))
            if res["info"]:
                infos.append(res["info"])
    finally:
        if cache:
            cache.close()
    infos.sort(key=lambda x: str(x["path"]))   # orden estable para agrupar
    print(f"    {len(infos)} procesadas OK")
    if cache and cache.hits:
        print(f"    ⚡ {cache.hits} desde caché (sin re-decodificar)")
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
        _prepare_output_dir(output)

        orig_total, conv_total = 0, 0
        heic_copied = alpha_copied = exception_copied = 0
        progress = Progress(len(winners), "Guardando")
        for i, w in enumerate(winners, 1):
            progress.update(i - 1, w['path'].name)
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
            dst = _unique_path(dst)

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

        progress.finish()
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
                    if p != root_dir and p != output and not _is_managed_dir(p):
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
# 3. MEJORA DE FOTOS
# ══════════════════════════════════════════════════


def _ensure_cv2():
    if not _CV2:
        print("  ✗ OpenCV no disponible. Instala:")
        print("    pip install opencv-contrib-python==4.10.0.84")
        return False
    return True


def _load_oriented_rgb(img_path: Path) -> Image.Image:
    """Abre con Pillow (soporta HEIC), aplica la orientación EXIF y pasa a RGB.

    Leer siempre por Pillow evita la ambigüedad de v1.6: cv2.imread rotaba
    según EXIF pero el fallback de Pillow no.
    """
    with Image.open(img_path) as opened:
        if getattr(opened, "n_frames", 1) > 1:
            raise ValueError("imagen animada/multipágina: omitida")
        return ImageOps.exif_transpose(opened).convert("RGB")


def _pil_to_bgr(img: Image.Image):
    return np.asarray(img)[:, :, ::-1].copy()


def _bgr_to_pil(arr) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]))


_PIL_SAVE_FORMATS = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
    ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP",
    ".heic": "HEIF", ".heif": "HEIF",
}


def _source_metadata(source: Path) -> dict:
    """EXIF (con orientación=1, los píxeles ya están rotados), ICC y DPI."""
    meta = {}
    try:
        with Image.open(source) as orig:
            exif = orig.getexif()
            if exif:
                if TAG_ORIENTATION in exif:
                    exif[TAG_ORIENTATION] = 1
                meta["exif"] = exif.tobytes()
            if orig.info.get("icc_profile"):
                meta["icc_profile"] = orig.info["icc_profile"]
            if orig.info.get("dpi"):
                meta["dpi"] = orig.info["dpi"]
    except Exception:
        pass
    return meta


def _save_pil(img: Image.Image, dst: Path, source: Path,
              quality: int = 95) -> tuple[Path | None, bool]:
    """Guarda en UNA sola codificación, con los metadatos del original.

    v1.6 escribía con OpenCV a q95 y luego Pillow re-guardaba sin `quality`
    (q75 por defecto): doble pérdida. Retorna (ruta escrita, metadatos_ok).
    """
    fmt = _PIL_SAVE_FORMATS.get(dst.suffix.lower())
    if fmt == "HEIF" and not _HEIF:
        fmt = None
    if fmt is None:
        # GIF y similares → PNG (sin pérdida); resto → JPEG
        new_ext = ".png" if dst.suffix.lower() == ".gif" else ".jpg"
        dst = _unique_path(dst.with_suffix(new_ext))
        fmt = "PNG" if new_ext == ".png" else "JPEG"

    base_kw = {"format": fmt}
    if fmt in ("JPEG", "WEBP", "HEIF"):
        base_kw["quality"] = quality
    meta = _source_metadata(source)

    for with_meta in ((True, False) if meta else (False,)):
        kw = {**base_kw, **(meta if with_meta else {})}
        try:
            img.save(str(dst), **kw)
            if dst.exists() and dst.stat().st_size > 0:
                return dst, with_meta or not meta
        except Exception:
            dst.unlink(missing_ok=True)
    return None, False


def enhance_denoise(img, strength: int = 6):
    """Reducción de ruido preservando detalle."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def enhance_sharpen(img, amount: float = 0.5):
    """Unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


_REALESRGAN_NAMES = ("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe")


def _realesrgan_search_dirs() -> list[Path]:
    """Carpeta 'realesrgan' junto al script y ~/.photo_manager/realesrgan."""
    return [SCRIPT_DIR / "realesrgan", REALESRGAN_DIR]


def _ensure_executable(path: Path) -> Path:
    """Algunos descompresores pierden el permiso de ejecución: restaurarlo."""
    if os.name != "nt" and not os.access(path, os.X_OK):
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except OSError:
            pass
    return path


def find_realesrgan() -> Path | None:
    """Busca realesrgan-ncnn-vulkan. Ubicación principal: la carpeta
    'realesrgan' junto a photo_manager.py (también un nivel más adentro,
    por si el zip se descomprimió dentro de su propia subcarpeta).

    Solo si ahí no está, usa como respaldo: variable de entorno
    PHOTO_MANAGER_REALESRGAN → ~/.photo_manager/realesrgan → PATH.
    """
    primary, *fallback_dirs = _realesrgan_search_dirs()
    found = _find_in_dir(primary)
    if found:
        return found
    env = os.environ.get(REALESRGAN_ENV)
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return _ensure_executable(p)
    for base in fallback_dirs:
        found = _find_in_dir(base)
        if found:
            return found
    found = shutil.which("realesrgan-ncnn-vulkan")
    return Path(found) if found else None


def _find_in_dir(base: Path) -> Path | None:
    """Ejecutable en base/ o en base/<subcarpeta>/."""
    if not base.is_dir():
        return None
    candidates = [base / n for n in _REALESRGAN_NAMES]
    try:
        candidates += sorted(c for sub in base.iterdir() if sub.is_dir()
                             for c in (sub / n for n in _REALESRGAN_NAMES))
    except OSError:
        pass
    for p in candidates:
        if p.is_file():
            return _ensure_executable(p)
    return None


# Resoluciones finales. Las de pantalla definen un recuadro (lado largo ×
# lado corto) en el que la foto debe CABER, girado si la foto es vertical:
# 640×480 → Full HD = 1440×1080 (no 1920×1440). Los factores son fijos.
UPSCALE_TARGETS = {
    "fullhd": {"label": "Full HD (1920×1080)", "box": (1920, 1080), "suffix": "_FHD"},
    "2k":     {"label": "2K (2560×1440)",      "box": (2560, 1440), "suffix": "_2K"},
    "4k":     {"label": "4K (3840×2160)",      "box": (3840, 2160), "suffix": "_4K"},
    "x2":     {"label": "×2 (el doble)",       "factor": 2.0,       "suffix": "_x2"},
    "x4":     {"label": "×4 (máximo)",         "factor": 4.0,       "suffix": "_x4"},
}
DEFAULT_TARGET = "fullhd"
# "Máximo" del menú: 4K con tope ×4. Las fotos pequeñas se amplían todo lo
# que da el modelo; las que ya son 4K o más no se tocan. (×4 literal en una
# foto de 12 MP daría 192 MP: archivos de ~100 MB y horas de GPU.)
AUTO_TARGET = "4k"
REALESRGAN_MAX_SCALE = 4.0   # el modelo x4plus no da más de ×4
AUTO_MIN_UPSCALE = 1.25      # en modo auto, no vale la pena ampliar menos


def upscale_factor(width: int, height: int, target: str) -> float:
    """Factor necesario para llegar a la resolución `target` (sin límite)."""
    spec = UPSCALE_TARGETS[target]
    if "factor" in spec:
        return spec["factor"]
    box_long, box_short = spec["box"]
    long_side, short_side = max(width, height), min(width, height)
    return min(box_long / long_side, box_short / short_side)


def target_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round(width * scale)), max(1, round(height * scale))


REALESRGAN_TIMEOUT_PER_IMAGE = 1800   # s; protección ante un bloqueo de la GPU
REALESRGAN_CHUNK = 8                  # fotos por ejecución del programa
REALESRGAN_CHUNK_MPX = 400            # límite de megapíxeles de salida por lote
_PCT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


def _realesrgan_cmd(exe: Path, src: Path, dst: Path) -> list[str]:
    cmd = [str(exe), "-i", str(src), "-o", str(dst),
           "-n", REALESRGAN_MODEL, "-s", "4", "-f", "png"]
    models = exe.parent / "models"
    if models.is_dir():
        cmd += ["-m", str(models)]
    return cmd


def realesrgan_batch(images: list[Image.Image], exe: Path, scales: list[float],
                     on_progress=None) -> list:
    """Amplía varias imágenes; si el lote completo falla, reintenta una a una.

    El reintento cubre versiones del programa que no acepten carpetas o
    errores puntuales con un archivo concreto.
    """
    results = _realesrgan_run(images, exe, scales, on_progress, as_dir=True)
    if len(images) > 1 and all(isinstance(r, Exception) for r in results):
        results = []
        for i, (img, scale) in enumerate(zip(images, scales)):
            cb = (lambda done, frac, base=i: on_progress(base + done, frac)) if on_progress else None
            results += _realesrgan_run([img], exe, [scale], cb, as_dir=False)
    return results


def _realesrgan_run(images: list[Image.Image], exe: Path, scales: list[float],
                    on_progress=None, as_dir: bool = True) -> list:
    """Amplía varias imágenes en UNA ejecución de Real-ESRGAN.

    El modelo y la GPU se inicializan una sola vez por lote, en vez de una
    vez por foto. on_progress(terminadas, fracción_de_la_actual|None) se
    llama a medida que el programa informa porcentajes.
    Devuelve, por imagen, la imagen ampliada (PIL) o la excepción.
    """
    import threading
    from collections import deque

    results: list = [None] * len(images)
    with tempfile.TemporaryDirectory(prefix="pm_esrgan_") as tmp:
        in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        names = []
        for i, img in enumerate(images):
            name = f"{i:05d}.png"
            img.save(in_dir / name, compress_level=1)   # temporal: rápido
            names.append(name)

        if as_dir:
            cmd = _realesrgan_cmd(exe, in_dir, out_dir)
        else:
            cmd = _realesrgan_cmd(exe, in_dir / names[0], out_dir / names[0])
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, errors="replace")
        except OSError as e:
            err = RuntimeError(f"No se pudo ejecutar Real-ESRGAN: {e}")
            return [err] * len(images)

        timer = threading.Timer(REALESRGAN_TIMEOUT_PER_IMAGE * len(images), proc.kill)
        timer.start()
        last_lines = deque(maxlen=5)
        try:
            for line in proc.stderr:
                line = line.strip()
                if not line:
                    continue
                last_lines.append(line)
                if on_progress:
                    m = _PCT_RE.search(line)
                    frac = float(m.group(1).replace(",", ".")) / 100 if m else None
                    finished = sum(1 for n in names if (out_dir / n).exists())
                    on_progress(finished, frac)
            proc.wait()
        finally:
            timer.cancel()

        detail = next((ln for ln in reversed(last_lines) if not _PCT_RE.fullmatch(ln)), "")
        for i, (img, name, scale) in enumerate(zip(images, names, scales)):
            out = out_dir / name
            if not out.exists():
                # El binario a veces sale con código 0 aunque falle Vulkan
                results[i] = RuntimeError(
                    "Real-ESRGAN falló" + (f": {detail}" if detail
                                           else f" (código {proc.returncode}, sin salida)"))
                continue
            try:
                with Image.open(out) as up:
                    up = up.convert("RGB")
                size = target_size(img.width, img.height,
                                   min(float(scale), REALESRGAN_MAX_SCALE))
                if up.size != size:
                    up = up.resize(size, Image.Resampling.LANCZOS)
                results[i] = up
            except Exception as e:
                results[i] = e
    return results


def realesrgan_upscale(img: Image.Image, exe: Path, out_scale: float = 2) -> Image.Image:
    """Super-resolución de una sola imagen (Real-ESRGAN x4plus + LANCZOS)."""
    result = realesrgan_batch([img], exe, [out_scale])[0]
    if isinstance(result, Exception):
        raise result if isinstance(result, RuntimeError) else RuntimeError(str(result))
    return result


def _esrgan_chunks(items: list, sizes: list[tuple[int, int]]):
    """Agrupa en lotes de hasta REALESRGAN_CHUNK fotos y REALESRGAN_CHUNK_MPX
    megapíxeles de salida (×4), para no llenar el disco con temporales."""
    chunk, mpx = [], 0.0
    for item, (w, h) in zip(items, sizes):
        out_mpx = w * h * 16 / 1e6
        if chunk and (len(chunk) >= REALESRGAN_CHUNK or mpx + out_mpx > REALESRGAN_CHUNK_MPX):
            yield chunk
            chunk, mpx = [], 0.0
        chunk.append(item)
        mpx += out_mpx
    if chunk:
        yield chunk


# ── Modo automático adaptativo ────────────────────
# Umbrales calibrados con imágenes sintéticas, NO con una biblioteca real.
# Si el diagnóstico (mejora_reporte.csv) no coincide con tu criterio, ajústalos.
AUTO_LOWRES_SIDE = 1600   # lado mayor (px) por debajo → baja resolución
AUTO_NOISE_SIGMA = 4.0    # ruido estimado (σ en escala 0-255)
AUTO_BRIGHT_MEAN = 185    # luminancia media por encima → sobreexpuesta
AUTO_LOW_RANGE = 170      # rango tonal p0.5–p99.5 por debajo → plana/lavada
AUTO_CAST = 5.0           # tinte medio de los píxeles neutros (a/b en LAB)
AUTO_NEUTRAL_CHROMA = 22  # croma máx. para considerar un píxel "casi neutro"
AUTO_NEUTRAL_MIN_SHARE = 0.05  # mínimo de neutros (5 %) para poder juzgar
AUTO_SOFT = 100.0         # varianza del Laplaciano a 1000 px → poco nítida
AUTO_HDR_MEAN = 110       # brillo medio por debajo → HDR (levantar sombras)
AUTO_HDR_SHADOWS = 0.30   # …o si más del 30 % de la foto está en sombra
HDR_SHADOW_LEVEL = 70     # luminancia (0-255) por debajo de la cual es "sombra"
# Intensidad del HDR: (exponente de la exposición clara, fuerza de la mezcla).
# Medido en una foto oscura (brillo 97) y en un interior junto a una ventana
# (sombras 47): suave → 104 / 66, normal → 109 / 80, fuerte → 115 / 94.
HDR_LEVELS = {
    "suave":  (0.55, 0.7),
    "normal": (0.45, 0.9),
    "fuerte": (0.40, 1.0),
}
DEFAULT_HDR = "normal"
AUTO_HDR = "normal"        # valor por defecto para run_enhance (el auto lo elige solo)
HDR_WEIGHT_MAX_PX = 2_000_000   # resolución máx. para calcular los pesos


def estimate_noise(gray) -> float:
    """σ del ruido (método de Immerkær) medido solo en zonas lisas.

    Los bordes y texturas se excluyen (30 % de píxeles con menor gradiente)
    para no confundir detalle con ruido.
    """
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    # Recorte central para fotos grandes: el ruido es homogéneo
    if h * w > 4_000_000:
        cy, cx, r = h // 2, w // 2, 1024
        gray = gray[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
    g = gray.astype(np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    resp = np.abs(cv2.filter2D(g, -1, kernel, borderType=cv2.BORDER_REFLECT))
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    flat = grad <= np.percentile(grad, 30)
    if flat.sum() < 100:
        return 0.0
    return float(math.sqrt(math.pi / 2) * resp[flat].mean() / 6)


def analyze_image(bgr) -> dict:
    """Mide lo que la foto necesita: resolución, ruido, exposición,
    rango tonal, dominante de color y nitidez.

    Tono, color y nitidez se miden sobre una copia de BLUR_MAX_SIDE px
    (las estadísticas apenas cambian y es ~10× más rápido en 12 MP); el
    ruido se mide a resolución completa porque al reducir desaparece.
    """
    h, w = bgr.shape[:2]
    small_bgr = bgr
    if max(h, w) > BLUR_MAX_SIDE:
        f = BLUR_MAX_SIDE / max(h, w)
        small_bgr = cv2.resize(bgr, (max(1, round(w * f)), max(1, round(h * f))),
                               interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]
    p_lo, p_hi = (float(v) for v in np.percentile(lum, (0.5, 99.5)))

    # Dominante de color: se mide SOLO en píxeles casi neutros de tonos medios.
    # Una foto de bosque o un atardecer tiene media verde/naranja pero sus
    # grises siguen siendo grises; con una dominante real, los grises se tiñen.
    # (El "gray world" de v1.6 promediaba toda la foto y neutralizaba eso.)
    a_ch = lab[:, :, 1].astype(np.float32) - 128
    b_ch = lab[:, :, 2].astype(np.float32) - 128
    mid = (lum > 40) & (lum < 230)
    neutral = mid & (np.hypot(a_ch, b_ch) < AUTO_NEUTRAL_CHROMA)
    neutral_share = float(neutral.sum()) / max(int(mid.sum()), 1)
    if neutral.sum() > 500 and neutral_share >= AUTO_NEUTRAL_MIN_SHARE:
        da = float(a_ch[neutral].mean())
        db = float(b_ch[neutral].mean())
    else:
        da = db = 0.0          # sin suficientes neutros: no se puede saber

    small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "width": w, "height": h,
        "low_res": max(w, h) < AUTO_LOWRES_SIDE,
        "noise": estimate_noise(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
        "mean": float(lum.mean()),
        "p_lo": p_lo, "p_hi": p_hi, "range": p_hi - p_lo,
        "da": da, "db": db, "cast": math.hypot(da, db),
        "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
    }


def _lab_apply(bgr, fn):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fn(lab)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _correct_cast(bgr, da: float, db: float, strength: float = 0.7):
    """Desplaza a/b en LAB hacia el neutro (parcial: respeta tonos cálidos)."""
    def fn(lab):
        lab[:, :, 1] -= da * strength
        lab[:, :, 2] -= db * strength
    return _lab_apply(bgr, fn)


def _stretch_levels(bgr, p_lo: float, p_hi: float, max_gain: float = 1.6):
    """Estira la luminancia p_lo→0 y p_hi→255 (solo L: no altera el color)."""
    span = max(p_hi - p_lo, 1.0)
    gain = min(255.0 / span, max_gain)
    center = (p_lo + p_hi) / 2

    def fn(lab):
        lab[:, :, 0] = (lab[:, :, 0] - center) * gain + 127.5
    return _lab_apply(bgr, fn)


def _pyr_levels(h: int, w: int) -> int:
    return int(min(max(math.log2(max(min(h, w), 1)) - 3, 3), 8))


def _gauss_pyr(img, levels: int) -> list:
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyr(img, levels: int) -> list:
    g = _gauss_pyr(img, levels)
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
           for i in range(levels - 1)]
    lap.append(g[-1])
    return lap


def _hdr_exposures(lum, bright: float = 0.45):
    """Exposiciones sintéticas de la luminancia (0-1): más oscura (protege
    luces), original, más clara (abre sombras) y con contraste local."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    local = clahe.apply((lum * 255).astype(np.uint8)).astype(np.float32) / 255
    yield np.power(lum, 1.5)
    yield lum
    yield np.power(lum, bright)
    yield local


def _hdr_weight(exp):
    """Peso de Mertens: bien expuesto (cerca de 0,5) × contraste local.

    El contraste se mide tras un leve desenfoque para que el ruido de las
    sombras no cuente como 'detalle'.
    """
    well_exposed = np.exp(-((exp - 0.5) ** 2) / (2 * 0.2 ** 2))
    contrast = np.abs(cv2.Laplacian(cv2.GaussianBlur(exp, (0, 0), 1.0), cv2.CV_32F))
    return well_exposed * (contrast + 0.02) + 1e-6


def _well_exposed(x):
    return np.exp(-((x - 0.5) ** 2) / (2 * 0.2 ** 2)) + 1e-6


def _hdr_fuse_guided(lum, bright: float):
    """Fusión con pesos suavizados por filtro guiado (Li et al. 2013).

    El filtro respeta los bordes de la propia foto: una zona en sombra junto
    a una ventana no hereda el peso de la ventana, así que no aparece el
    halo oscuro de la mezcla por pirámides (medido: −10 → ~0).
    El peso se evalúa sobre la base suavizada (escala grande) y se aplica a
    la exposición completa, que conserva el detalle fino.
    """
    gf = cv2.ximgproc.guidedFilter
    h, w = lum.shape
    # Los pesos son mapas suaves: se calculan sobre una copia de ≤2 MP y se
    # amplían. Reduce memoria y tiempo ~4× en fotos grandes sin efecto visible.
    f = min(1.0, math.sqrt(HDR_WEIGHT_MAX_PX / (h * w)))
    small = (cv2.resize(lum, (max(1, round(w * f)), max(1, round(h * f))),
                        interpolation=cv2.INTER_AREA) if f < 1 else lum)
    r = max(4, min(small.shape) // 48)
    eps = 1e-3
    base = gf(small, small, r, eps)
    num = np.zeros_like(lum)
    den = np.zeros_like(lum)
    for exp, exp_base in zip(_hdr_exposures(lum, bright), _hdr_exposures(base, bright)):
        wmap = np.maximum(gf(small, _well_exposed(exp_base), r, eps), 1e-6)
        if f < 1:
            wmap = cv2.resize(wmap, (w, h), interpolation=cv2.INTER_LINEAR)
        num += wmap * exp
        den += wmap
    return num / den


def _hdr_fuse_pyramid(lum, bright: float):
    """Respaldo sin ximgproc: fusión de Mertens por pirámides de Laplace
    (puede dejar un halo oscuro leve junto a bordes muy contrastados)."""
    h, w = lum.shape
    levels = _pyr_levels(h, w)
    weight_sum = sum(_hdr_weight(e) for e in _hdr_exposures(lum, bright))
    fused = None
    for exp in _hdr_exposures(lum, bright):
        wpyr = _gauss_pyr(_hdr_weight(exp) / weight_sum, levels)
        lpyr = _lap_pyr(exp, levels)
        contrib = [wl * ll for wl, ll in zip(wpyr, lpyr)]
        fused = contrib if fused is None else [f + c for f, c in zip(fused, contrib)]
    del weight_sum
    out = fused[-1]
    for lvl in reversed(fused[:-1]):
        out = cv2.pyrUp(out, dstsize=(lvl.shape[1], lvl.shape[0])) + lvl
    return out


_HAS_GUIDED = _CV2 and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")


def enhance_hdr(bgr, level: str = DEFAULT_HDR):
    """HDR de una sola foto: fusión de exposiciones sintéticas.

    Genera versiones más oscura, original, más clara y con contraste local
    de la luminancia, y las mezcla zona a zona tomando la mejor expuesta.
    No inventa información (un JPEG no tiene la de varias exposiciones
    reales, como el iPhone), pero aclara sombras sin quemar las luces.
    Solo actúa sobre la luminancia: los colores se conservan.
    """
    bright, strength = HDR_LEVELS[level]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum = lab[:, :, 0] / 255.0
    fused = (_hdr_fuse_guided(lum, bright) if _HAS_GUIDED
             else _hdr_fuse_pyramid(lum, bright))
    fused = np.clip(fused, 0.0, 1.0)
    lab[:, :, 0] = np.clip((lum + strength * (fused - lum)) * 255, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# Brillo medio objetivo tras aclarar, según intensidad (LAB 0-255; ~118 = medio)
HDR_TARGET_MEAN = {"suave": 100, "normal": 110, "fuerte": 120}
# Cuánto del negro original se recupera tras el HDR (0 = nada, foto lavada;
# 1 = todo, deshace parte del aclarado y destapa grano en zonas oscuras).
# Elegido sobre una foto real (selfie nocturno de WhatsApp) y una escena
# sintética de interior con ventana.
HDR_BLACK_RESTORE = 0.6
HDR_DEEP_SHADOW = 45   # luminancia original por debajo: limpieza fuerte tras aclarar


def _restore_black_point(bgr, black: float, knee: float = 40.0):
    """Devuelve el negro (percentil 0,5) a `black` solo en los tonos más bajos.

    El HDR levanta también los negros y la foto queda con neblina. Una
    curva de "pie": baja el negro hasta `black` y el efecto se desvanece
    hasta desaparecer `knee` niveles más arriba, así que los tonos medios
    (lo que el HDR aclaró) no se tocan.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lo = float(np.percentile(lab[:, :, 0], 0.5))
    if lo <= black + 1:
        return bgr
    x = np.arange(256, dtype=np.float32)
    offset = (lo - black) * np.clip(1 - (x - lo) / knee, 0, 1)
    lut = np.where(x <= lo, x * black / max(lo, 1), x - offset)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], np.clip(lut, 0, 255).astype(np.uint8))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _restore_saturation(before, after, amount: float = 0.7, max_gain: float = 1.8):
    """Aumenta el color en proporción al aclarado de cada zona.

    Al subir solo la luminancia, los colores conservan su intensidad
    absoluta y se ven grises (piel apagada). El factor sale de un mapa
    suavizado para no amplificar el ruido de color.
    """
    lb = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    la = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    gain = cv2.GaussianBlur((la[:, :, 0] + 8) / (lb[:, :, 0] + 8), (0, 0), 5)
    gain = np.clip(1 + amount * (gain - 1), 1.0, max_gain)
    for c in (1, 2):
        la[:, :, c] = 128 + (la[:, :, c] - 128) * gain
    return cv2.cvtColor(np.clip(la, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def brighten_dark(bgr, level: str = DEFAULT_HDR) -> tuple:
    """Aclara una foto oscura: HDR + punto negro + brillo + color + limpieza.

    1. HDR (fusión de exposiciones): abre sombras sin quemar luces.
    2. Recupera parte del negro original (HDR_BLACK_RESTORE): quita la
       neblina sin deshacer el aclarado de los tonos medios.
    3. Brillo global hacia HDR_TARGET_MEAN[level] si aún queda oscura
       (las fotos muy subexpuestas no llegan solo con el HDR).
    4. Saturación proporcional al aclarado: evita la piel gris.
    5. Limpieza proporcional al aclarado: el ruido y las manchas de color
       que estaban ocultos en la oscuridad se hacen visibles al aclarar
       (el ruido medido ANTES de aclarar no lo refleja). Fuerte en color,
       suave en luminancia, para conservar el detalle; y fuerte en todo
       donde el original era casi negro.
    Devuelve (imagen, aclarado medio ×, h de limpieza o 0).
    """
    lum_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    black = float(np.percentile(lum_before, 0.5))
    out = enhance_hdr(bgr, level)
    lifted_black = float(np.percentile(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0], 0.5))
    out = _restore_black_point(
        out, lifted_black - HDR_BLACK_RESTORE * (lifted_black - black))
    target = HDR_TARGET_MEAN[level]
    mean = float(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
    if mean < target:
        out = _gamma_toward(out, mean, target=target, amount=0.8)
    out = _restore_saturation(bgr, out)

    out, gain, h = _denoise_after_lift(lum_before, out)
    return out, gain, h


def _denoise_after_lift(lum_before, out) -> tuple:
    """Limpieza proporcional a cuánto se aclaró la foto.

    El ruido y las manchas de color que estaban ocultos en la oscuridad se
    hacen visibles al aclarar (el ruido medido ANTES no lo refleja). Fuerte
    en color y suave en luminancia, para conservar el detalle; fuerte en
    todo donde el original era casi negro. Devuelve (imagen, aclarado ×, h).
    """
    gain = (float(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].mean()) + 8) / \
           (float(lum_before.mean()) + 8)
    h = 0
    if gain >= 1.3:
        h = int(np.clip(round(1 + 2.5 * (gain - 1)), 2, 6))
        h_color = int(np.clip(round(4 + 8 * (gain - 1)), 4, 14))
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h_color, 7, 21)
        # Lo que era casi negro se aclaró muchísimo más que el resto y queda
        # con grano: limpieza fuerte solo ahí, con transición suave.
        mask = np.clip((HDR_DEEP_SHADOW - lum_before.astype(np.float32)) / 30, 0, 1)
        if mask.mean() > 0.03:
            mask = cv2.GaussianBlur(mask, (0, 0), 6)[..., None]
            strong = cv2.fastNlMeansDenoisingColored(out, None, 8, 12, 7, 21)
            out = (mask * strong + (1 - mask) * out).astype(np.uint8)
    return out, gain, h


# ── Aclarado clásico con brillo automático ────────
# Cadena de un fotógrafo: exposición (igual en R, G, B, con "hombro" que
# frena antes del blanco) → curva S → vibrance → calidez si está azulada →
# limpieza. Con una foto real (selfie nocturno) quedó más natural que la
# fusión HDR: mismo brillo y el triple de contraste en la cara.
AUTO_BRIGHT_MIN = 100        # objetivos de brillo medio que se prueban
AUTO_BRIGHT_MAX = 140
AUTO_BRIGHT_STEP = 5
# Límite para las zonas claras (percentil 98 de la luminancia). Calibrado con
# la foto real: objetivo 120 → 222 (natural), 140 → 235 (cielo y pelo
# blanquecinos). Si las luces ya estaban altas, se permite +8 sobre ellas.
AUTO_HIGHLIGHT_P98 = 224
AUTO_HIGHLIGHT_MARGIN = 8
AUTO_MAX_BURN = 0.05         # % máx. de píxeles quemados (≥ 250)
AUTO_MAX_GAIN = 3.0          # aclarado máx.: más allá el ruido se dispara
AUTO_BACKLIT_GAIN = 1.08     # si la cadena no aclara más que esto → contraluz
AUTO_BACKLIT_HDR = "normal"  # intensidad del HDR para contraluces


def _exposure_lut(k: float):
    """Curva de exposición con hombro: y = (1 − e^(−kx)) / (1 − e^(−k))."""
    x = np.arange(256) / 255.0
    return np.clip((1 - np.exp(-k * x)) / (1 - np.exp(-k)) * 255, 0, 255).astype(np.uint8)


def _s_curve(bgr, amount: float = 0.25):
    """Curva S suave sobre la luminancia: devuelve volumen tras aclarar."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    x = np.arange(256) / 255.0
    curve = x + amount * (x - 0.5) * (1 - np.abs(2 * x - 1))
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], np.clip(curve * 255, 0, 255).astype(np.uint8))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _vibrance(bgr, amount: float = 0.5):
    """Sube más el color de lo poco saturado (piel) que de lo ya intenso."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    factor = 1 + amount * np.clip(1 - chroma / 40, 0, 1)
    for c in (1, 2):
        lab[..., c] = 128 + (lab[..., c] - 128) * factor
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _classic_chain(bgr, k: float):
    return _s_curve(cv2.LUT(bgr, _exposure_lut(k)))


def choose_brightness(bgr) -> tuple[int | None, float]:
    """Elige el brillo objetivo más alto que no blanquee las zonas claras.

    Prueba de AUTO_BRIGHT_MAX hacia abajo (sobre una copia reducida, rápido)
    y devuelve (objetivo, k de la curva). (None, 0) si ninguno cumple.
    """
    h, w = bgr.shape[:2]
    small = bgr
    if max(h, w) > BLUR_MAX_SIDE:
        f = BLUR_MAX_SIDE / max(h, w)
        small = cv2.resize(bgr, (max(1, round(w * f)), max(1, round(h * f))),
                           interpolation=cv2.INTER_AREA)
    lum = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    l0 = lum(small)
    limit = max(AUTO_HIGHLIGHT_P98, float(np.percentile(l0, 98)) + AUTO_HIGHLIGHT_MARGIN)
    for target in range(AUTO_BRIGHT_MAX, AUTO_BRIGHT_MIN - 1, -AUTO_BRIGHT_STEP):
        lo, hi = 0.01, 20.0                      # k que da ese brillo medio
        for _ in range(20):
            k = (lo + hi) / 2
            if lum(cv2.LUT(small, _exposure_lut(k))).mean() < target:
                lo = k
            else:
                hi = k
        l1 = lum(_classic_chain(small, k))
        gain = (l1.mean() + 8) / (l0.mean() + 8)
        if (np.percentile(l1, 98) <= limit and (l1 >= 250).mean() * 100 <= AUTO_MAX_BURN
                and gain <= AUTO_MAX_GAIN):
            return target, k
    return None, 0.0


def brighten_classic(bgr, analysis: dict | None = None) -> tuple:
    """Aclara con la cadena clásica y brillo elegido automáticamente.

    Devuelve (imagen, objetivo | None, aclarado ×). Si ningún objetivo
    respeta las zonas claras, devuelve la imagen sin cambios.
    """
    target, k = choose_brightness(bgr)
    lum_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    if target is None:
        return bgr, None, 1.0
    out = _vibrance(_classic_chain(bgr, k))
    a = analysis or {}
    if a.get("db", 0) < -3:                      # neutros azulados → calentar
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 2] += min(-a["db"], 10) * 0.7
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out, gain, _ = _denoise_after_lift(lum_before, out)
    return out, target, gain


def _shadow_share(bgr) -> float:
    """Fracción de la foto en sombra (medida sobre una copia reducida)."""
    h, w = bgr.shape[:2]
    small = bgr
    if max(h, w) > BLUR_MAX_SIDE:
        f = BLUR_MAX_SIDE / max(h, w)
        small = cv2.resize(bgr, (max(1, round(w * f)), max(1, round(h * f))),
                           interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
    return float((lum < HDR_SHADOW_LEVEL).mean())


def _gamma_toward(bgr, mean: float, target: float = 118.0, amount: float = 0.6):
    """Gamma en L que acerca la media a `target`, solo parcialmente."""
    mean = min(max(mean, 1.0), 254.0)
    gamma = math.log(target / 255) / math.log(mean / 255)
    gamma = 1 + amount * (gamma - 1)
    gamma = min(max(gamma, 0.55), 1.6)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                   dtype=np.float32)

    def fn(lab):
        lab[:, :, 0] = lut[lab[:, :, 0].astype(np.uint8)]
    return _lab_apply(bgr, fn)


def enhance_adaptive(bgr, esrgan: Path | None = None, dry_run: bool = False,
                     target: str = DEFAULT_TARGET, analysis: dict | None = None,
                     upscaled=None, hdr: str = DEFAULT_HDR):
    """Analiza la foto y aplica solo las correcciones que necesita.

    Retorna (imagen BGR, análisis, lista de acciones legibles).
    Orden: super-resolución → ruido → color → rango → exposición → nitidez.
    Con dry_run=True decide igual pero omite los pasos lentos (super-
    resolución y reducción de ruido); sirve para el modo diagnóstico.
    `target` (clave de UPSCALE_TARGETS) fija la resolución final: solo se
    amplía si la foto no la alcanza por al menos AUTO_MIN_UPSCALE.
    `analysis`: análisis ya calculado (evita repetirlo).
    `upscaled`: imagen BGR ya ampliada por un lote de Real-ESRGAN; si se
    pasa, se usa en lugar de llamar al programa.
    """
    a = analysis or analyze_image(bgr)
    actions = []

    needed = upscale_factor(a["width"], a["height"], target)
    a["upscale_needed"] = needed
    a["needs_upscale"] = needed >= AUTO_MIN_UPSCALE

    pre_upscaled, upscaled = upscaled, False
    if a["needs_upscale"] and (esrgan or pre_upscaled is not None):
        # Real-ESRGAN además limpia ruido y artefactos JPEG
        scale = min(needed, REALESRGAN_MAX_SCALE)
        new_w, new_h = target_size(a["width"], a["height"], scale)
        if pre_upscaled is not None:
            bgr = pre_upscaled
        elif not dry_run:
            bgr = _pil_to_bgr(realesrgan_upscale(_bgr_to_pil(bgr), esrgan, scale))
        capped = " (limitado a ×4)" if needed > REALESRGAN_MAX_SCALE else ""
        actions.append(f"super-resolución ×{scale:.2g} ({a['width']}x{a['height']} → "
                       f"{new_w}x{new_h}){capped}")
        upscaled = True

    if not upscaled and a["noise"] > AUTO_NOISE_SIGMA:
        strength = int(min(max(round(a["noise"] * 1.2), 3), 12))
        if not dry_run:
            bgr = enhance_denoise(bgr, strength=strength)
        actions.append(f"reducir ruido (σ={a['noise']:.1f}, h={strength})")

    if a["cast"] > AUTO_CAST:
        bgr = _correct_cast(bgr, a["da"], a["db"])
        actions.append(f"corregir dominante de color ({a['cast']:.0f})")

    if a["range"] < AUTO_LOW_RANGE:
        bgr = _stretch_levels(bgr, a["p_lo"], a["p_hi"])
        actions.append(f"ampliar rango tonal ({a['range']:.0f} → ~255)")

    mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
    shadows = _shadow_share(bgr)
    a["shadows"] = shadows
    brightened = False
    if mean < AUTO_HDR_MEAN or shadows > AUTO_HDR_SHADOWS:
        target, k = choose_brightness(bgr)
        small_gain = 1.0
        if target is not None:
            small = cv2.resize(bgr, (max(1, bgr.shape[1] // 4), max(1, bgr.shape[0] // 4)),
                               interpolation=cv2.INTER_AREA)
            lum_s = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            lum_c = cv2.cvtColor(_classic_chain(small, k), cv2.COLOR_BGR2LAB)[:, :, 0]
            small_gain = (lum_c.mean() + 8) / (lum_s.mean() + 8)
        if target is None or small_gain < AUTO_BACKLIT_GAIN:
            # Luces ya altas (contraluz, ventana, cielo): la exposición global
            # no puede abrir las sombras sin blanquearlas → fusión HDR.
            if not dry_run:
                bgr, gain, _ = brighten_dark(bgr, AUTO_BACKLIT_HDR)
                a["brighten_gain"] = gain
            actions.append(f"HDR: aclarar sombras, luces ya altas (brillo {mean:.0f}, "
                           f"sombras {shadows * 100:.0f} %)")
        else:
            if not dry_run:
                bgr, _, gain = brighten_classic(bgr, a)
                a["brighten_gain"] = gain
            a["brightness_target"] = target
            actions.append(f"aclarar: brillo {mean:.0f} → {target} (automático)")
        brightened = True
    elif mean > AUTO_BRIGHT_MEAN:
        bgr = _gamma_toward(bgr, mean)
        actions.append(f"oscurecer (brillo {mean:.0f})")

    if not upscaled and not brightened and a["sharpness"] < AUTO_SOFT:
        # No se enfoca tras aclarar: realzaría el grano que el aclarado destapa.
        # Enfoque suave si hay ruido o baja resolución: evita realzar artefactos
        amount = 0.3 if (a["noise"] > AUTO_NOISE_SIGMA or a["low_res"]) else 0.6
        bgr = enhance_sharpen(bgr, amount=amount)
        actions.append(f"enfocar ({amount}, nitidez {a['sharpness']:.0f})")

    return bgr, a, actions


def _auto_report_row(path: Path, a: dict, actions: list[str]) -> dict:
    return {
        "archivo": str(path),
        "resolucion": f"{a['width']}x{a['height']}",
        "baja_resolucion": "SI" if a["low_res"] else "",
        "ampliacion_necesaria": (f"×{a['upscale_needed']:.2f}"
                                 if a.get("needs_upscale") else ""),
        "ruido_sigma": f"{a['noise']:.1f}",
        "brillo_medio": f"{a['mean']:.0f}",
        "sombras_pct": f"{a['shadows'] * 100:.0f}" if "shadows" in a else "",
        "rango_tonal": f"{a['range']:.0f}",
        "dominante_color": f"{a['cast']:.1f}",
        "nitidez": f"{a['sharpness']:.0f}",
        "acciones": "; ".join(actions) if actions else "ninguna (ya está bien)",
    }


def _oriented_size(path: Path) -> tuple[int, int]:
    """Tamaño tal como se ve la foto (orientación EXIF aplicada), sin decodificar."""
    with Image.open(path) as img:
        w, h = img.size
        try:
            if img.getexif().get(TAG_ORIENTATION) in (5, 6, 7, 8):
                w, h = h, w
        except Exception:
            pass
    return w, h


def _enhance_task(job: dict) -> dict:
    """Procesa una foto en un proceso aparte (CPU). Nunca lanza excepciones.

    status: ok | copied | esrgan (necesita ampliación: la hace el proceso
    principal en la fase de GPU) | error
    """
    src, dst, mode = job["src"], job["dst"], job["mode"]
    res = {"src": src, "status": "ok", "row": None, "analysis": None,
           "actions": [], "meta_ok": True, "error": None}
    try:
        arr = _pil_to_bgr(_load_oriented_rgb(src))
        if mode in ("auto", "diagnose"):
            analysis = analyze_image(arr)
            needed = upscale_factor(analysis["width"], analysis["height"], job["target"])
            if mode == "auto" and job["esrgan"] and needed >= AUTO_MIN_UPSCALE:
                res.update(status="esrgan", analysis=analysis)
                return res
            arr, analysis, actions = enhance_adaptive(
                arr, job["esrgan"], dry_run=(mode == "diagnose"),
                target=job["target"], analysis=analysis, hdr=job["hdr"])
            res.update(analysis=analysis, actions=actions,
                       row=_auto_report_row(src, analysis, actions))
            if mode == "diagnose":
                return res
            if not actions:
                # Nada que corregir: copia exacta, sin re-comprimir
                shutil.copy2(str(src), str(dst))
                res["status"] = "copied"
                return res
        else:                       # mode == "hdr"
            arr, _, _ = brighten_dark(arr, job["hdr"])
        written, meta_ok = _save_pil(_bgr_to_pil(arr), dst, src)
        if not written:
            raise OSError(f"no se pudo escribir {dst.name}")
        res["meta_ok"] = meta_ok
    except Exception as e:
        res.update(status="error", error=f"{src.name}: {e}")
    return res


def run_enhance(source: Path, output: Path, mode: str,
                target: str = DEFAULT_TARGET, workers: int | None = None,
                hdr: str = DEFAULT_HDR):
    """Ejecuta mejora de fotos.

    Modos: auto (analiza y corrige solo lo necesario), diagnose (solo
    reporte), realesrgan (super-resolución) y hdr (aclarar sombras).
    `target` (clave de UPSCALE_TARGETS): resolución final para Real-ESRGAN,
    tanto en el modo realesrgan como en auto.

    Dos fases: 1) CPU en paralelo (análisis, ajustes, guardado);
    2) GPU, en serie y por lotes, para las fotos que Real-ESRGAN debe ampliar.
    """
    if target not in UPSCALE_TARGETS:
        print(f"  ✗ Resolución desconocida: {target}")
        return
    if mode != "realesrgan" and not _ensure_cv2():
        return
    if not _NUMPY:
        print("  ✗ Requiere numpy.")
        return

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not _validate_source_output(source, output):
        return

    esrgan = None
    if mode == "realesrgan":
        esrgan = find_realesrgan()
        if not esrgan:
            print("  ✗ No se encontró realesrgan-ncnn-vulkan.")
            print("    Descárgalo de https://github.com/xinntao/Real-ESRGAN/releases")
            print(f"    y descomprímelo en {SCRIPT_DIR / 'realesrgan'}")
            return
    elif mode in ("auto", "diagnose"):
        esrgan = find_realesrgan()   # opcional: solo para lo que no llega al objetivo
    elif mode != "hdr":
        print(f"  ✗ Modo desconocido: {mode}")
        return

    images = find_images(source, exclude=output)
    if not images:
        print("  No se encontraron imágenes.")
        return

    _prepare_output_dir(output)
    adaptive = mode in ("auto", "diagnose")
    diagnose_only = mode == "diagnose"
    verb = "a analizar" if diagnose_only else "a procesar"
    print(f"  {len(images)} imágenes {verb}")
    if esrgan:
        print(f"  Resolución final: {UPSCALE_TARGETS[target]['label']}")
    print()
    suffix = UPSCALE_TARGETS[target]["suffix"]

    # Destinos (calculados en el proceso principal: sin carreras de nombres)
    jobs = []
    for img_path in images:
        try:
            rel = img_path.relative_to(source)
        except ValueError:
            rel = Path(img_path.name)
        plain = output / rel
        dst = plain.with_stem(plain.stem + suffix) if mode == "realesrgan" else plain
        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs.append({"src": img_path, "dst": _unique_path(dst), "plain": plain,
                     "mode": mode, "target": target, "esrgan": esrgan, "hdr": hdr})

    ok = errors = metadata_warnings = already_big = lowres_without_esrgan = 0
    report_rows, error_msgs = [], []
    action_counts = defaultdict(int)
    esrgan_jobs = []        # (job, análisis | None)
    t0 = time.time()

    def record(res):
        nonlocal ok, errors, metadata_warnings, lowres_without_esrgan
        if res["status"] == "error":
            errors += 1
            error_msgs.append(res["error"])
            return
        ok += 1
        if not res["meta_ok"]:
            metadata_warnings += 1
        if res["row"]:
            report_rows.append(res["row"])
            a = res["analysis"]
            if a and a.get("needs_upscale") and not esrgan:
                lowres_without_esrgan += 1
            for act in res["actions"]:
                action_counts[act.split(" (")[0].split(" ×")[0]] += 1

    # ── Fase 1: CPU ────────────────────────────────
    if mode == "realesrgan":
        for job in jobs:
            try:
                w, h = _oriented_size(job["src"])
            except Exception as e:
                record({"status": "error", "error": f"{job['src'].name}: {e}"})
                continue
            if upscale_factor(w, h, target) <= 1.0:
                # Ya alcanza la resolución pedida: copia exacta, sin ampliar
                shutil.copy2(str(job["src"]), str(_unique_path(job["plain"])))
                already_big += 1
                ok += 1
            else:
                esrgan_jobs.append((job, None))
    else:
        label = "Analizando" if diagnose_only else "Procesando"
        for i, res in parallel_map(_enhance_task, jobs, label, workers=workers,
                                   describe=lambda j: j["src"].name):
            if res["status"] == "esrgan":
                esrgan_jobs.append((jobs[i], res["analysis"]))
            else:
                record(res)

    # ── Fase 2: GPU (Real-ESRGAN por lotes) ─────────
    if esrgan_jobs:
        print(f"\n  Real-ESRGAN: {len(esrgan_jobs)} foto(s) a ampliar")
        progress = Progress(len(esrgan_jobs), "Real-ESRGAN")
        sizes = []
        for job, analysis in esrgan_jobs:
            if analysis:
                sizes.append((analysis["width"], analysis["height"]))
            else:
                try:
                    sizes.append(_oriented_size(job["src"]))
                except Exception:
                    sizes.append((1, 1))
        done_before = 0
        for chunk in _esrgan_chunks(esrgan_jobs, sizes):
            loaded, scales, valid = [], [], []
            for job, analysis in chunk:
                try:
                    img = _load_oriented_rgb(job["src"])
                except Exception as e:
                    record({"status": "error", "error": f"{job['src'].name}: {e}"})
                    continue
                loaded.append(img)
                scales.append(upscale_factor(img.width, img.height, target))
                valid.append((job, analysis))

            def on_progress(finished, frac, base=done_before, chunk_jobs=valid):
                current = chunk_jobs[min(finished, len(chunk_jobs) - 1)][0]["src"].name
                progress.update(base + finished, current, sub=frac)

            if valid:
                progress.update(done_before, valid[0][0]["src"].name, force=True)
                ups = realesrgan_batch(loaded, esrgan, scales, on_progress)
            else:
                ups = []
            for (job, analysis), img, up in zip(valid, loaded, ups):
                res = {"src": job["src"], "status": "ok", "row": None,
                       "analysis": None, "actions": [], "meta_ok": True}
                try:
                    if isinstance(up, Exception):
                        raise up
                    if analysis is not None:          # modo auto: resto de ajustes
                        arr, analysis, actions = enhance_adaptive(
                            _pil_to_bgr(img), esrgan, target=target,
                            analysis=analysis, upscaled=_pil_to_bgr(up), hdr=hdr)
                        up = _bgr_to_pil(arr)
                        res.update(analysis=analysis, actions=actions,
                                   row=_auto_report_row(job["src"], analysis, actions))
                    written, meta_ok = _save_pil(up, job["dst"], job["src"])
                    if not written:
                        raise OSError(f"no se pudo escribir {job['dst'].name}")
                    res["meta_ok"] = meta_ok
                except Exception as e:
                    res.update(status="error", error=f"{job['src'].name}: {e}")
                record(res)
            done_before += len(chunk)
            progress.update(done_before, force=True)
        progress.finish()

    # ── Resumen ──────────────────────────────────────
    elapsed = time.time() - t0
    done = "analizadas" if diagnose_only else "mejoradas"
    print(f"\n  ✓ {ok} {done} | {errors} errores | {_fmt_duration(elapsed)}")
    for msg in error_msgs[:10]:
        print(f"    ⚠ {msg}")
    if len(error_msgs) > 10:
        print(f"    ... y {len(error_msgs) - 10} errores más")
    if already_big:
        print(f"  ℹ {already_big} foto(s) ya tenían {UPSCALE_TARGETS[target]['label']} "
              "o más: copiadas sin cambios")

    if adaptive and report_rows:
        report_rows.sort(key=lambda r: r["archivo"])
        untouched = sum(1 for r in report_rows if r["acciones"].startswith("ninguna"))
        print("\n  Diagnóstico:" if diagnose_only else "\n  Correcciones aplicadas:")
        for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            name = name.split(":")[0]
            print(f"    {count:5d}  {name}")
        print(f"    {untouched:5d}  sin cambios (ya estaban bien)")
        if lowres_without_esrgan:
            print(f"\n  ⚠ {lowres_without_esrgan} foto(s) por debajo de "
                  f"{UPSCALE_TARGETS[target]['label']}.")
            print("    Los ajustes de tono no recuperan detalle que no existe.")
            print("    Para mejorarlas de verdad instala Real-ESRGAN:")
            print("    https://github.com/xinntao/Real-ESRGAN/releases")
        report = output / "mejora_reporte.csv"
        try:
            with open(report, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(report_rows[0]))
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"\n  ✓ Reporte por foto: {report}")
        except OSError as e:
            print(f"  ⚠ No se pudo escribir el reporte: {e}")

    if metadata_warnings:
        print(f"  ⚠ {metadata_warnings} archivo(s) sin metadatos restaurados (EXIF/ICC/DPI)")
    print(f"  ✓ Resultado en: {output}")


# ══════════════════════════════════════════════════
# 4. DETECTAR FOTOS BORROSAS
# ══════════════════════════════════════════════════


def blur_score(filepath: Path) -> float | None:
    """Varianza del Laplaciano — menor = más borrosa.

    La imagen se reduce primero a BLUR_MAX_SIDE px de lado mayor: sin esto,
    la misma escena puntuaba ~30 veces distinto a 4000 px que a 1000 px y
    un umbral fijo no significaba nada.
    """
    try:
        with Image.open(filepath) as img:
            if img.format == "JPEG":
                img.draft("L", (BLUR_MAX_SIDE, BLUR_MAX_SIDE))
            gray = img.convert("L")
        if max(gray.size) > BLUR_MAX_SIDE:
            gray.thumbnail((BLUR_MAX_SIDE, BLUR_MAX_SIDE), Image.Resampling.BOX)
    except Exception:
        return None

    try:
        if _CV2:
            return float(cv2.Laplacian(np.asarray(gray), cv2.CV_64F).var())
        # Sin OpenCV: mismo kernel Laplaciano 3×3 con Pillow. Aproximado:
        # Pillow recorta la respuesta a ±127, OpenCV no.
        lap = ImageFilter.Kernel((3, 3), (0, 1, 0, 1, -4, 1, 0, 1, 0),
                                 scale=1, offset=128)
        edges = gray.filter(lap)
        if _NUMPY:
            return float((np.asarray(edges, dtype="float64") - 128).var())
        n = s = s2 = 0
        for px in edges.getdata():
            v = px - 128
            n += 1
            s += v
            s2 += v * v
        if n == 0:
            return None
        mean = s / n
        return float(s2 / n - mean * mean)
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

    for i, score in parallel_map(blur_score, images, "Analizando",
                                 describe=lambda p: p.name):
        img_path = images[i]
        if score is None:
            skipped.append(img_path)
        elif score < threshold:
            blurry.append((img_path, score))
        else:
            sharp.append((img_path, score))

    blurry.sort(key=lambda x: x[1])
    sharp.sort(key=lambda x: str(x[0]))
    skipped.sort()
    print(f"\n  Nítidas: {len(sharp)} | Borrosas: {len(blurry)}", end="")
    if skipped:
        print(f" | No analizadas: {len(skipped)}")
    else:
        print()

    # [v1.3 M10] Mapeo de paths movidos para actualizar el CSV
    moved_paths = {}  # original_path → new_path

    if blurry:
        print("\n  Top 10 más borrosas:")
        for path, score in blurry[:10]:
            print(f"    {score:8.1f}  {path.name}")

        if move_blurry:
            blur_dir = source / "_BORROSAS"
            # [v1.3 M9] Verificar que blur_dir no sea un archivo
            if blur_dir.exists() and not blur_dir.is_dir():
                print(f"  ✗ {blur_dir} existe pero no es un directorio.")
            else:
                _prepare_output_dir(blur_dir)
                moved = 0
                for path, _ in blurry:
                    try:
                        # Conserva la estructura de subcarpetas para poder
                        # devolver cada foto a su sitio.
                        try:
                            rel = path.relative_to(source)
                        except ValueError:
                            rel = Path(path.name)
                        dst = blur_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst = _unique_path(dst)
                        shutil.move(str(path), str(dst))
                        moved_paths[path] = dst
                        moved += 1
                    except Exception as e:
                        print(f"  ⚠ No se pudo mover {path.name}: {e}")
                print(f"\n  ✓ {moved} borrosas movidas a {blur_dir}")

    # [FIX v1.5+] Reporte CSV en _BORROSAS (si existe) o junto a source, nunca dentro de source
    if move_blurry and blurry:
        report = (source / "_BORROSAS") / "blur_reporte.csv"
    else:
        report = source.parent / f"blur_reporte_{source.name}.csv"
    try:
        with open(report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # archivo_original permite deshacer el movimiento
            w.writerow(["archivo_original", "archivo_actual", "score", "estado"])
            for path, score in blurry:
                actual = moved_paths.get(path, path)
                w.writerow([str(path), str(actual), f"{score:.1f}", "BORROSA"])
            for path, score in sharp:
                w.writerow([str(path), str(path), f"{score:.1f}", "NITIDA"])
            # [v1.3 H4] Incluir archivos no analizados en el reporte
            for path in skipped:
                w.writerow([str(path), str(path), "", "NO_ANALIZADA"])
        print(f"  ✓ Reporte: {report}")
    except Exception as e:
        print(f"  ⚠ No se pudo escribir reporte: {e}")


# ══════════════════════════════════════════════════
# 5. COMPRIMIR / CONVERTIR
# ══════════════════════════════════════════════════


# Resolución máxima al comprimir: límite del LADO LARGO (una foto 4:3 del
# iPhone a "Full HD" queda en 1920×1440). WhatsApp envía 1600 px en calidad
# estándar y 4096 px en HD.
COMPRESS_MAX_SIDES = {
    "original": ("Original (sin cambio)", None),
    "4k": ("4K (lado largo 3840 px)", 3840),
    "2k": ("2K (lado largo 2560 px)", 2560),
    "fullhd": ("Full HD (lado largo 1920 px)", 1920),
    "whatsapp": ("WhatsApp estándar (lado largo 1600 px)", 1600),
}
COMPRESS_DEFAULT_QUALITY = 75


def _compress_task(job: dict) -> dict:
    """Comprime una foto (proceso aparte). Nunca lanza excepciones.

    status: ok | kept (el resultado no era más pequeño: se conserva el
    original) | copied (animada/multipágina: copia sin tocar) | error
    """
    src, dst, fmt = job["src"], job["dst"], job["fmt"]
    res = {"src": src, "status": "ok", "out": None, "resized": False,
           "orig_size": 0, "new_size": 0, "error": None}
    try:
        res["orig_size"] = src.stat().st_size
        with Image.open(src) as opened:
            if getattr(opened, "n_frames", 1) > 1:
                out = _unique_path(dst.with_suffix(src.suffix))
                shutil.copy2(str(src), str(out))
                res.update(status="copied", out=out, new_size=out.stat().st_size)
                return res
            img = opened.copy()
            info = dict(opened.info)
        exif, icc, dpi = info.get("exif"), info.get("icc_profile"), info.get("dpi")
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in info)

        # Reducir por el lado largo. No se aplica la orientación EXIF: los
        # píxeles se escalan tal cual y la etiqueta de orientación sigue valiendo.
        max_side = job["max_side"]
        if max_side and max(img.size) > max_side:
            f = max_side / max(img.size)
            size = (max(1, round(img.width * f)), max(1, round(img.height * f)))
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGBA" if has_alpha else "RGB")
            img = img.resize(size, Image.Resampling.LANCZOS)
            res["resized"] = True

        kw = {"quality": job["quality"]}
        if exif:
            kw["exif"] = exif
        if icc:
            kw["icc_profile"] = icc
        if fmt == "heif":
            img = img.convert("RGBA" if has_alpha else "RGB")
            kw["format"] = "HEIF"
        elif fmt == "webp":
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if has_alpha else "RGB")
            kw.update({"format": "WEBP", "method": 4})
        else:
            # JPEG no admite alfa; componer explícitamente sobre blanco.
            if has_alpha:
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            kw.update({"format": "JPEG", "optimize": True})
            if dpi:
                kw["dpi"] = dpi
        img.save(str(dst), **kw)

        new_size = dst.stat().st_size
        if not res["resized"] and new_size >= res["orig_size"]:
            # Recomprimir no ahorró nada: conservar el original intacto
            dst.unlink()
            out = _unique_path(dst.with_suffix(src.suffix))
            shutil.copy2(str(src), str(out))
            res.update(status="kept", out=out, new_size=out.stat().st_size)
            return res
        res.update(out=dst, new_size=new_size)
    except Exception as e:
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        res.update(status="error", error=f"{src.name}: {e}")
    return res


def run_compress(source: Path, output: Path, fmt: str = "heif",
                 quality: int = COMPRESS_DEFAULT_QUALITY, keep_structure: bool = True,
                 max_side: int | None = None, workers: int | None = None):
    """Comprime/convierte imágenes (también las que ya son HEIF/HEIC).

    max_side: límite del lado largo en píxeles (None = sin cambio).
    Nunca deja un archivo más grande que el original si no se redujo la
    resolución: en ese caso conserva el original.
    """
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not _validate_source_output(source, output):
        return
    if fmt == "heif" and not _HEIF:
        print("  ✗ HEIF no disponible. Instala: pip install pillow-heif")
        return
    images = find_images(source, exclude=output)
    if not images:
        print("  No se encontraron imágenes.")
        return

    _prepare_output_dir(output)
    size_txt = f", lado largo ≤ {max_side} px" if max_side else ""
    print(f"  {len(images)} imágenes → {fmt.upper()} (calidad {quality}{size_txt})\n")

    target_ext = {"heif": ".heif", "webp": ".webp", "jpeg": ".jpg"}.get(fmt, ".jpg")
    jobs, reserved = [], set()
    for img_path in images:
        if keep_structure:
            try:
                rel = img_path.relative_to(source)
            except ValueError:
                rel = Path(img_path.name)
            dst = output / rel.with_suffix(target_ext)
        else:
            dst = output / (img_path.stem + target_ext)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # foto.jpg y foto.png irían ambas a foto.heif: reservar nombres
        candidate, c = _unique_path(dst), 1
        while candidate in reserved:
            candidate = dst.with_name(f"{dst.stem}_{c}{dst.suffix}")
            c += 1
        reserved.add(candidate)
        jobs.append({"src": img_path, "dst": candidate, "fmt": fmt,
                     "quality": quality, "max_side": max_side})

    orig_total = new_total = 0
    ok = errors = kept = copied = resized = 0
    error_msgs = []
    for _, res in parallel_map(_compress_task, jobs, "Comprimiendo", workers=workers,
                               describe=lambda j: j["src"].name):
        if res["status"] == "error":
            errors += 1
            error_msgs.append(res["error"])
            continue
        ok += 1
        orig_total += res["orig_size"]
        new_total += res["new_size"]
        kept += res["status"] == "kept"
        copied += res["status"] == "copied"
        resized += res["resized"]

    print(f"\n  Original:  {format_size(orig_total)}")
    print(f"  Resultado: {format_size(new_total)}")
    if orig_total:
        print(f"  Reducción: {(1 - new_total / orig_total) * 100:.0f}%")
    print(f"  ✓ {ok} procesadas | {errors} errores")
    for msg in error_msgs[:10]:
        print(f"    ⚠ {msg}")
    if resized:
        print(f"  ℹ {resized} foto(s) reducida(s) a lado largo {max_side} px")
    if kept:
        print(f"  ℹ {kept} foto(s) conservada(s) tal cual: recomprimir no las hacía más pequeñas")
    if copied:
        print(f"  ℹ {copied} archivo(s) animado(s)/multipágina copiado(s) sin cambios")
    print(f"  ✓ Resultado en: {output}")


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
    dry_run = ask_confirm("¿Solo simular (dry run)?", default_yes=False)

    run_dedup(source, output, compare=compare, threshold=threshold,
              no_convert=no_convert, quality=quality, keep_originals=keep,
              dry_run=dry_run)


def menu_enhance():
    banner("MEJORAR FOTOS")
    if not _CV2:
        print("  ✗ Requiere OpenCV. Instala:")
        print("    pip install opencv-contrib-python==4.10.0.84")
        return
    esrgan = find_realesrgan()

    source = ask_folder("Carpeta de fotos")
    if not source:
        return
    output = ask_folder("Carpeta de salida (Enter = auto)", must_exist=False)
    if not output:
        output = source / "MEJORADAS"

    es_tag = "" if esrgan else " (Real-ESRGAN no instalado)"
    print(f"""
  Modos de mejora:
    1. Automático: analiza cada foto y corrige solo lo necesario
       (amplía hasta {UPSCALE_TARGETS[AUTO_TARGET]['label']}, aclara fotos oscuras con brillo automático){es_tag}
    2. Real-ESRGAN: ampliar hasta {UPSCALE_TARGETS[AUTO_TARGET]['label']} (máx. ×4){es_tag}
    3. HDR: aclarar sombras sin quemar luces (a todas)
    4. Solo diagnosticar (no modifica nada, genera reporte)
""")
    opt = ask("Modo", "1")
    mode = {"1": "auto", "2": "realesrgan", "3": "hdr", "4": "diagnose"}.get(opt)
    if not mode:
        print("  Opción no válida.")
        return

    if mode == "realesrgan" and not esrgan:
        print("  ✗ Real-ESRGAN no instalado. Descarga realesrgan-ncnn-vulkan de")
        print("    https://github.com/xinntao/Real-ESRGAN/releases")
        print(f"    y descomprímelo en {SCRIPT_DIR / 'realesrgan'}")
        return

    # Automático, Real-ESRGAN y diagnóstico: sin preguntas, al máximo útil.
    hdr = AUTO_HDR
    if mode == "hdr":
        hdr = ask_hdr_level()
        if not hdr:
            return

    run_enhance(source, output, mode, target=AUTO_TARGET, hdr=hdr)


def ask_hdr_level() -> str | None:
    """Pregunta cuánto aclarar las sombras (HDR)."""
    keys = list(HDR_LEVELS)
    print("\n  Intensidad del HDR (cuánto aclarar las fotos oscuras):")
    labels = {"suave": "Suave", "normal": "Normal (recomendado)", "fuerte": "Fuerte (aspecto HDR marcado)"}
    for n, key in enumerate(keys, 1):
        print(f"    {n}. {labels.get(key, key)}")
    opt = ask("Intensidad", str(keys.index(DEFAULT_HDR) + 1))
    try:
        return keys[int(opt) - 1]
    except (ValueError, IndexError):
        print("  Opción no válida.")
        return None


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

    heif_tag = " (recomendado; también recomprime HEIC/HEIF)" if _HEIF else " (no disponible)"
    print(f"""
  Formato:
    1. HEIF{heif_tag}
    2. WebP
    3. JPEG optimizado
""")
    opt = ask("Formato", "1" if _HEIF else "2")
    fmt = {"1": "heif", "2": "webp", "3": "jpeg"}.get(opt)
    if not fmt:
        print("  Opción no válida.")
        return
    if fmt == "heif" and not _HEIF:
        print("  ✗ HEIF no disponible. Instala: pip install pillow-heif")
        return
    quality = ask_int(f"Calidad ({COMPRESS_DEFAULT_QUALITY}=recomendado)",
                      COMPRESS_DEFAULT_QUALITY, min_val=1, max_val=100)

    keys = list(COMPRESS_MAX_SIDES)
    print("\n  Resolución máxima (reducir es irreversible):")
    for n, key in enumerate(keys, 1):
        print(f"    {n}. {COMPRESS_MAX_SIDES[key][0]}")
    opt = ask("Resolución", "1")
    try:
        max_side = COMPRESS_MAX_SIDES[keys[int(opt) - 1]][1]
    except (ValueError, IndexError):
        print("  Opción no válida.")
        return
    run_compress(source, output, fmt=fmt, quality=quality, max_side=max_side)


def main_menu():
    while True:
        cls()
        deps = []
        if _HEIF:
            deps.append("HEIF")
        if _CV2:
            deps.append("OpenCV")
        if find_realesrgan():
            deps.append("ESRGAN")
        dep_str = " + ".join(deps) if deps else "solo Pillow"

        print(f"""
╔══════════════════════════════════════════════╗
║         PHOTO MANAGER CLI v{VERSION:<18s}║
╠══════════════════════════════════════════════╣
║                                              ║
║  1. Deduplicar fotos                         ║
║  2. Comparar dos carpetas                    ║
║  3. Mejorar fotos (IA / HDR)                 ║
║  4. Detectar fotos borrosas                  ║
║  5. Comprimir / Convertir                    ║
║                                              ║
║  0. Salir                                    ║
║                                              ║
╠══════════════════════════════════════════════╣
║  Módulos: {dep_str:<35s}║
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
            menu_blur()
        elif opt == "5":
            menu_compress()
        else:
            print("  Opción no válida.")

        if opt in ("1", "2", "3", "4", "5"):
            print()
            input("  Enter para volver al menú...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  Interrumpido. Hasta luego.\n")
