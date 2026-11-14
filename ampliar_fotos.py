#!/usr/bin/env python3
"""
Ampliar Fotos v1.0.0 — super-resolución con Real-ESRGAN

Amplía una foto, varias fotos o una carpeta entera (con subcarpetas).

Uso:
  python ampliar_fotos.py                  → pregunta; arrastra la foto o carpeta
  python ampliar_fotos.py foto.jpg         → una foto
  python ampliar_fotos.py carpeta/ a.heic  → carpetas y fotos mezcladas

Qué hace:
  - Amplía hasta 4K (3840×2160) con un máximo de ×4, usando el modelo
    realesrgan-x4plus (fotos reales). Las fotos que ya son 4K se omiten.
  - Foto suelta   → foto_4K.jpg junto al original.
  - Carpeta       → carpeta/AMPLIADAS/ con la misma estructura.
  - Nunca modifica los originales. Conserva fecha, GPS, perfil de color.
  - Lee JPG, PNG, WebP, TIFF, BMP y HEIC/HEIF.

Requisitos:
  pip install Pillow pillow-heif
  Real-ESRGAN: descarga realesrgan-ncnn-vulkan (v0.2.5.0) desde
  https://github.com/xinntao/Real-ESRGAN/releases y descomprímelo en una
  carpeta "realesrgan" junto a este script.
  En macOS, además:  xattr -dr com.apple.quarantine realesrgan
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("Falta Pillow. Instala con:  pip install Pillow pillow-heif")
    sys.exit(1)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIF = True
except ImportError:
    _HEIF = False

VERSION = "1.0.0"

# ══════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif",
}
OUTPUT_DIR_NAME = "AMPLIADAS"
# Carpetas que nunca se recorren: salidas de este script y de Photo Manager
SKIP_DIR_NAMES = {OUTPUT_DIR_NAME, "MEJORADAS", "FOTOS_DEDUP", "COMPARACION",
                  "COMPRIMIDAS", "_BORROSAS"}
OUTPUT_MARKER = ".photo_manager_output"
TAG_ORIENTATION = 274


REALESRGAN_ENV = "PHOTO_MANAGER_REALESRGAN"
REALESRGAN_MODEL = "realesrgan-x4plus"
REALESRGAN_DIR = Path.home() / ".photo_manager" / "realesrgan"
SCRIPT_DIR = Path(__file__).resolve().parent

UPSCALE_TARGETS = {
    "fullhd": {"label": "Full HD (1920×1080)", "box": (1920, 1080), "suffix": "_FHD"},
    "2k":     {"label": "2K (2560×1440)",      "box": (2560, 1440), "suffix": "_2K"},
    "4k":     {"label": "4K (3840×2160)",      "box": (3840, 2160), "suffix": "_4K"},
    "x2":     {"label": "×2 (el doble)",       "factor": 2.0,       "suffix": "_x2"},
    "x4":     {"label": "×4 (máximo)",         "factor": 4.0,       "suffix": "_x4"},
}
TARGET = "4k"                # resolución final (clave de UPSCALE_TARGETS)
REALESRGAN_MAX_SCALE = 4.0   # el modelo x4plus no da más de ×4
REALESRGAN_TIMEOUT_PER_IMAGE = 1800
REALESRGAN_CHUNK = 8
REALESRGAN_CHUNK_MPX = 400
_PCT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


# ══════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════
# LEER Y GUARDAR IMÁGENES
# ══════════════════════════════════════════════════


def _load_oriented_rgb(img_path: Path) -> Image.Image:
    """Abre con Pillow (soporta HEIC), aplica la orientación EXIF y pasa a RGB.

    Leer siempre por Pillow evita la ambigüedad de v1.6: cv2.imread rotaba
    según EXIF pero el fallback de Pillow no.
    """
    with Image.open(img_path) as opened:
        if getattr(opened, "n_frames", 1) > 1:
            raise ValueError("imagen animada/multipágina: omitida")
        return ImageOps.exif_transpose(opened).convert("RGB")


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


# ══════════════════════════════════════════════════
# REAL-ESRGAN
# ══════════════════════════════════════════════════


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
            if proc.poll() is None:      # p. ej. Ctrl+C: no dejar la GPU ocupada
                proc.kill()

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


# ══════════════════════════════════════════════════
# AMPLIAR
# ══════════════════════════════════════════════════


def parse_paths(raw: str) -> list[Path]:
    """Convierte lo escrito o arrastrado a la terminal en una lista de rutas.

    Admite una ruta sola (con o sin comillas, con espacios) o varias fotos
    arrastradas a la vez, separadas por espacios y entre comillas o con
    los espacios escapados, como las pega la terminal.
    """
    raw = raw.strip()
    if not raw:
        return []
    whole = Path(clean_path_input(raw)).expanduser()
    if whole.exists():
        return [whole]
    try:
        parts = shlex.split(raw, posix=(os.name != "nt"))
    except ValueError:
        parts = [raw]
    return [Path(clean_path_input(p)).expanduser() for p in parts if p.strip()]


def _is_output_dir(path: Path) -> bool:
    if path.name in SKIP_DIR_NAMES:
        return True
    try:
        return (path / OUTPUT_MARKER).exists()
    except OSError:
        return False


def _images_in(folder: Path) -> list[Path]:
    """Fotos de una carpeta y sus subcarpetas, sin las carpetas de salida."""
    found = []
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        dp = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not _is_output_dir(dp / d))
        found += [dp / f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(found)


def plan_jobs(paths: list[Path], target: str = TARGET) -> tuple[list, list[str]]:
    """Decide qué se amplía y dónde se guarda.

    Devuelve (trabajos, avisos); cada trabajo es (origen, destino, (ancho, alto)).
    Se omiten: lo que ya alcanza la resolución, lo ya ampliado antes (existe
    el archivo de salida) y los propios resultados (*_4K).
    """
    suffix = UPSCALE_TARGETS[target]["suffix"]
    label = UPSCALE_TARGETS[target]["label"]
    jobs, notes, seen = [], [], set()
    already_big = already_done = 0

    candidates = []                       # (origen, destino)
    for p in paths:
        try:
            p = p.resolve()
        except OSError:
            pass
        if p.is_dir():
            out_dir = p / OUTPUT_DIR_NAME
            for img in _images_in(p):
                if img.stem.endswith(suffix):
                    continue              # resultado de una ejecución anterior
                rel = img.relative_to(p)
                candidates.append((img, out_dir / rel.with_name(img.stem + suffix + img.suffix)))
        elif p.is_file():
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                notes.append(f"✗ No es una imagen compatible: {p.name}")
                continue
            candidates.append((p, p.with_name(p.stem + suffix + p.suffix)))
        else:
            notes.append(f"✗ No existe: {p}")

    for src, dst in candidates:
        if src in seen:
            continue
        seen.add(src)
        if dst.exists():
            already_done += 1
            continue
        try:
            size = _oriented_size(src)
        except Exception as e:
            notes.append(f"⚠ {src.name}: no se pudo leer ({e})")
            continue
        if upscale_factor(*size, target) <= 1.0:
            already_big += 1
            continue
        jobs.append((src, dst, size))

    if already_big:
        notes.append(f"ℹ {already_big} foto(s) ya tienen {label} o más: omitidas")
    if already_done:
        notes.append(f"ℹ {already_done} foto(s) ya estaban ampliadas (existe su *{suffix}): "
                     "omitidas. Borra ese archivo para repetirlas.")
    return jobs, notes


def upscale(paths: list[Path], target: str = TARGET) -> int:
    """Amplía fotos y/o carpetas. Devuelve el número de fotos ampliadas."""
    esrgan = find_realesrgan()
    if not esrgan:
        print("  ✗ No se encontró realesrgan-ncnn-vulkan.")
        print("    Descárgalo de https://github.com/xinntao/Real-ESRGAN/releases")
        print(f"    y descomprímelo en {SCRIPT_DIR / 'realesrgan'}")
        return 0

    jobs, notes = plan_jobs(paths, target)
    for note in notes:
        print(f"  {note}")
    if not jobs:
        print("  Nada que ampliar.")
        return 0

    label = UPSCALE_TARGETS[target]["label"]
    print(f"\n  {len(jobs)} foto(s) → hasta {label}, máximo ×{REALESRGAN_MAX_SCALE:g}\n")

    ok, errors, meta_warnings = 0, [], 0
    out_dirs, single_dirs = set(), set()
    t0 = time.time()
    progress = Progress(len(jobs), "Real-ESRGAN")
    done_before = 0
    for chunk in _esrgan_chunks(jobs, [j[2] for j in jobs]):
        loaded, scales, valid = [], [], []
        for src, dst, _ in chunk:
            try:
                img = _load_oriented_rgb(src)
            except Exception as e:
                errors.append(f"{src.name}: {e}")
                continue
            loaded.append(img)
            scales.append(upscale_factor(img.width, img.height, target))
            valid.append((src, dst))

        def on_progress(finished, frac, base=done_before, names=valid):
            current = names[min(finished, len(names) - 1)][0].name
            progress.update(base + finished, current, sub=frac)

        ups = []
        if valid:
            progress.update(done_before, valid[0][0].name, force=True)
            ups = realesrgan_batch(loaded, esrgan, scales, on_progress)
        for (src, dst), up in zip(valid, ups):
            try:
                if isinstance(up, Exception):
                    raise up
                dst.parent.mkdir(parents=True, exist_ok=True)
                in_out = [d for d in [dst.parent, *dst.parents] if d.name == OUTPUT_DIR_NAME]
                (out_dirs if in_out else single_dirs).add(in_out[-1] if in_out else dst.parent)
                written, meta_ok = _save_pil(up, _unique_path(dst), src)
                if not written:
                    raise OSError(f"no se pudo escribir {dst.name}")
                ok += 1
                meta_warnings += not meta_ok
            except Exception as e:
                errors.append(f"{src.name}: {e}")
        done_before += len(chunk)
        progress.update(done_before, force=True)
    progress.finish()

    for d in out_dirs:                    # marcar las carpetas de salida
        try:
            (d / OUTPUT_MARKER).write_text(
                "Carpeta generada por Ampliar Fotos. Se ignora al escanear.\n",
                encoding="utf-8")
        except OSError:
            pass

    print(f"\n  ✓ {ok} ampliada(s) | {len(errors)} error(es) | {_fmt_duration(time.time() - t0)}")
    for msg in errors[:10]:
        print(f"    ⚠ {msg}")
    if len(errors) > 10:
        print(f"    ... y {len(errors) - 10} errores más")
    if meta_warnings:
        print(f"  ⚠ {meta_warnings} archivo(s) sin metadatos (EXIF/ICC)")
    for d in sorted(out_dirs):
        print(f"  ✓ Carpeta de resultados: {d}")
    for d in sorted(single_dirs):
        print(f"  ✓ Resultados junto a los originales en: {d}")
    return ok


def main(argv: list[str]) -> int:
    print(f"\n  AMPLIAR FOTOS v{VERSION} — Real-ESRGAN ({REALESRGAN_MODEL})")
    if not _HEIF:
        print("  ℹ Sin pillow-heif: no se podrán leer fotos HEIC/HEIF.")
    if argv:
        upscale([Path(clean_path_input(a)).expanduser() for a in argv])
        return 0
    while True:
        print("\n  Arrastra aquí una foto o una carpeta y pulsa Enter (Enter vacío = salir)")
        try:
            raw = input("  > ")
        except EOFError:
            break
        paths = parse_paths(raw)
        if not paths:
            break
        upscale(paths)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n\n  Interrumpido.\n")
        sys.exit(130)
