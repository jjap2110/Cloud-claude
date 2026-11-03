"""Tests de regresión para photo_manager.py.

Cada test reproduce un bug concreto encontrado en v1.6.4 o cubre una mejora
de v1.7.0. Ejecutar con:  python -m pytest -q
"""

import csv
import os
import stat
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import photo_manager as pm  # noqa: E402

EXIF_IFD = 0x8769


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def scene(w: int = 320, h: int = 240, seed: int = 0) -> Image.Image:
    """Imagen sintética con estructura (bloques + ruido suavizado) para pHash."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.Resampling.NEAREST)
    return img.filter(ImageFilter.GaussianBlur(1))


def exif_bytes(original: str | None = None, ifd0: str | None = None,
               orientation: int | None = None) -> bytes:
    ex = Image.Exif()
    if ifd0:
        ex[306] = ifd0
    if orientation:
        ex[274] = orientation
    if original:
        ex.get_ifd(EXIF_IFD)[36867] = original
    return ex.tobytes()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Caché y modelos en un directorio temporal; confirmaciones automáticas."""
    home = tmp_path / "_home"
    monkeypatch.setattr(pm, "CACHE_PATH", home / "hash_cache.sqlite")
    monkeypatch.setattr(pm, "ask_confirm", lambda *a, **k: True)
    monkeypatch.delenv(pm.REALESRGAN_ENV, raising=False)
    yield


# ──────────────────────────────────────────────────────────────
# Fechas
# ──────────────────────────────────────────────────────────────

def test_date_read_from_exif_subifd_not_ifd0(tmp_path):
    """v1.6.4 leía 36867 del IFD0 (siempre None) y usaba DateTime (edición)."""
    p = tmp_path / "foto.jpg"
    scene().save(p, exif=exif_bytes(original="2019:05:05 12:00:00",
                                    ifd0="2024:01:01 10:00:00"))
    assert pm.get_photo_date(p) == datetime(2019, 5, 5, 12, 0, 0)
    info = pm.get_image_info(p)
    assert info["date"] == datetime(2019, 5, 5, 12, 0, 0)


@pytest.mark.parametrize("name, expected", [
    ("IMG_20190505_123456.jpg", datetime(2019, 5, 5, 12, 34, 56)),
    ("IMG-20190505-WA0001.jpg", datetime(2019, 5, 5)),
    ("WhatsApp Image 2019-05-05 at 12.34.56.jpeg", datetime(2019, 5, 5, 12, 34, 56)),
    ("Screenshot_2021-11-30-08-15-00.png", datetime(2021, 11, 30, 8, 15, 0)),
    ("PXL_20230101_101010123.jpg", datetime(2023, 1, 1, 10, 10, 10)),
    ("IMG_1234.jpg", None),
    ("DSC_20191345.jpg", None),       # mes 13
    ("scan_20190230.jpg", None),      # 30 de febrero
    ("x120190505.jpg", None),         # precedido de dígito
])
def test_date_from_filename(name, expected):
    assert pm._date_from_filename(name) == expected


def test_filename_date_beats_mtime_when_no_exif(tmp_path):
    p = tmp_path / "IMG-20190505-WA0001.jpg"
    scene().save(p)
    t = datetime(2025, 3, 1).timestamp()
    os.utime(p, (t, t))
    assert pm.get_photo_date(p).date() == datetime(2019, 5, 5).date()


def test_capture_date_beats_filename(tmp_path):
    p = tmp_path / "IMG_20200101_000000.jpg"
    scene().save(p, exif=exif_bytes(original="2018:07:07 07:07:07"))
    assert pm.get_photo_date(p) == datetime(2018, 7, 7, 7, 7, 7)


def test_organize_uses_capture_date(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    scene().save(src / "a.jpg", exif=exif_bytes(original="2019:05:05 12:00:00",
                                                ifd0="2024:01:01 10:00:00"))
    pm.run_organize(src, out)
    assert (out / "2019" / "05-Mayo" / "a.jpg").exists()


# ──────────────────────────────────────────────────────────────
# Dedup: nunca borrar el original por una copia generada
# ──────────────────────────────────────────────────────────────

def test_dedup_does_not_delete_original_in_favor_of_upscaled_copy(tmp_path):
    """v1.6.4: MEJORADAS/foto_x2 ganaba por resolución y se borraba el original."""
    src = tmp_path / "fotos"
    (src / "MEJORADAS").mkdir(parents=True)
    img = scene()
    img.save(src / "foto.jpg", quality=95)
    img.resize((640, 480), Image.Resampling.LANCZOS).save(
        src / "MEJORADAS" / "foto_x2.jpg", quality=95)

    pm.run_dedup(src, src / "FOTOS_DEDUP", no_convert=True,
                 keep_originals=False)

    assert (src / "foto.jpg").exists()
    assert (src / "MEJORADAS" / "foto_x2.jpg").exists()


def test_custom_output_dir_is_marked_and_skipped(tmp_path):
    src = tmp_path / "fotos"
    src.mkdir()
    scene().save(src / "a.jpg")
    custom = src / "mi_salida"
    pm.run_organize(src, custom)
    assert (custom / pm.OUTPUT_MARKER).exists()
    found = pm.find_images(src)
    assert found == [src / "a.jpg"]


def test_dedup_still_finds_real_duplicates(tmp_path):
    src = tmp_path / "fotos"
    (src / "sub").mkdir(parents=True)
    img = scene(seed=3)
    img.save(src / "a.jpg", quality=95)
    img.save(src / "sub" / "a_copia.png")             # mismo contenido, otro formato
    (src / "a_exacta.jpg").write_bytes((src / "a.jpg").read_bytes())
    scene(seed=99).save(src / "otra.jpg", quality=95)

    infos = [pm.get_image_info(p) for p in pm.find_images(src)]
    groups = pm.group_duplicates(infos, threshold=6)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 3]


# ──────────────────────────────────────────────────────────────
# Mejora: una sola codificación con metadatos
# ──────────────────────────────────────────────────────────────

def _icc_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_enhance_encodes_once_and_keeps_metadata(tmp_path):
    """v1.6.4 re-guardaba con Pillow a calidad 75 por defecto."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    icc = _icc_bytes()
    scene().save(src / "a.jpg", quality=95, icc_profile=icc,
                 exif=exif_bytes(original="2019:05:05 12:00:00"))

    pm.run_enhance(src, out, "sharpen")

    with Image.open(out / "a.jpg") as res:
        # Tabla de cuantización de luminancia: q95 → primer coef. 2; q75 → 8
        assert res.quantization[0][0] <= 3
        assert res.info.get("icc_profile") == icc
        assert res.getexif().get_ifd(EXIF_IFD).get(36867) == "2019:05:05 12:00:00"


def test_enhance_applies_orientation_once(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    # 320x240 almacenada, orientación 6 (rotar 90°) → se ve 240x320
    scene().save(src / "r.jpg", quality=95, exif=exif_bytes(orientation=6))
    pm.run_enhance(src, out, "sharpen")
    with Image.open(out / "r.jpg") as res:
        assert res.size == (240, 320)
        assert res.getexif().get(274) in (None, 1)


# ──────────────────────────────────────────────────────────────
# HEIF conserva ICC
# ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not pm._HEIF, reason="pillow-heif no instalado")
def test_convert_heif_keeps_icc(tmp_path):
    src = tmp_path / "a.jpg"
    icc = _icc_bytes()
    scene().save(src, quality=95, icc_profile=icc)
    written, reason = pm._convert_heif(src, tmp_path / "a.heif")
    assert reason is None
    with Image.open(written) as res:
        assert res.info.get("icc_profile")


# ──────────────────────────────────────────────────────────────
# Borrosas
# ──────────────────────────────────────────────────────────────

def test_blur_move_keeps_structure_and_original_path_in_csv(tmp_path):
    src = tmp_path / "fotos"
    (src / "viaje").mkdir(parents=True)
    Image.new("RGB", (400, 300), (128, 128, 128)).save(src / "viaje" / "plana.jpg")
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(
        src / "nitida.png")

    pm.run_blur_detect(src, threshold=100.0, move_blurry=True)

    moved = src / "_BORROSAS" / "viaje" / "plana.jpg"
    assert moved.exists()
    with open(src / "_BORROSAS" / "blur_reporte.csv", encoding="utf-8") as f:
        rows = {r["estado"]: r for r in csv.DictReader(f)}
    assert rows["BORROSA"]["archivo_original"] == str(src / "viaje" / "plana.jpg")
    assert rows["BORROSA"]["archivo_actual"] == str(moved)


def test_blur_score_is_resolution_independent(tmp_path):
    """La misma escena a 4000 px y a 1000 px debe puntuar parecido."""
    big = scene(4000, 3000, seed=5)
    small = big.resize((1000, 750), Image.Resampling.LANCZOS)
    big.save(tmp_path / "big.png")
    small.save(tmp_path / "small.png")
    s_big = pm.blur_score(tmp_path / "big.png")
    s_small = pm.blur_score(tmp_path / "small.png")
    ratio = max(s_big, s_small) / min(s_big, s_small)
    assert ratio < 1.5, (s_big, s_small)


# ──────────────────────────────────────────────────────────────
# Caché de hashes + draft
# ──────────────────────────────────────────────────────────────

def test_hash_cache_hit_skips_decoding(tmp_path, monkeypatch):
    p = tmp_path / "a.jpg"
    scene().save(p)
    with pm.HashCache(pm.CACHE_PATH) as cache:
        first = pm.get_image_info(p, cache=cache)
    assert first is not None

    def boom(*a, **k):
        raise AssertionError("no debería decodificar con caché válida")
    monkeypatch.setattr(pm.imagehash, "phash", boom)

    with pm.HashCache(pm.CACHE_PATH) as cache:
        second = pm.get_image_info(p, cache=cache)
        assert cache.hits == 1
    assert str(second["phash"]) == str(first["phash"])
    assert second["file_hash"] == first["file_hash"]
    assert second["date"] == first["date"]


def test_hash_cache_invalidated_when_file_changes(tmp_path):
    p = tmp_path / "a.jpg"
    scene(seed=1).save(p)
    with pm.HashCache(pm.CACHE_PATH) as cache:
        first = pm.get_image_info(p, cache=cache)
    scene(seed=2).save(p)
    t = p.stat().st_mtime + 10
    os.utime(p, (t, t))
    with pm.HashCache(pm.CACHE_PATH) as cache:
        second = pm.get_image_info(p, cache=cache)
        assert cache.hits == 0
    assert second["file_hash"] != first["file_hash"]


def test_draft_keeps_real_dimensions(tmp_path):
    p = tmp_path / "big.jpg"
    scene(4000, 3000).save(p, quality=90)
    info = pm.get_image_info(p)
    assert (info["width"], info["height"]) == (4000, 3000)


# ──────────────────────────────────────────────────────────────
# Real-ESRGAN (binario simulado: prueba la integración, no el modelo)
# ──────────────────────────────────────────────────────────────

FAKE_ESRGAN = textwrap.dedent("""\
    #!{python}
    import sys
    from PIL import Image
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    s = int(args["-s"])
    with Image.open(args["-i"]) as im:
        im.resize((im.width * s, im.height * s)).save(args["-o"])
""")


def test_realesrgan_integration_with_fake_binary(tmp_path, monkeypatch):
    exe = tmp_path / "realesrgan-ncnn-vulkan"
    exe.write_text(FAKE_ESRGAN.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(exe))

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    scene().save(src / "a.jpg", quality=95,
                 exif=exif_bytes(original="2019:05:05 12:00:00"))

    pm.run_enhance(src, out, "realesrgan")

    with Image.open(out / "a_x2.jpg") as res:
        assert res.size == (640, 480)
        assert res.getexif().get_ifd(EXIF_IFD).get(36867) == "2019:05:05 12:00:00"


def test_realesrgan_failure_is_reported_not_silent(tmp_path, monkeypatch):
    exe = tmp_path / "realesrgan-ncnn-vulkan"
    exe.write_text(f"#!{sys.executable}\nimport sys; sys.exit(0)\n")  # no escribe salida
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(exe))
    img = scene()
    with pytest.raises(RuntimeError):
        pm.realesrgan_upscale(img, exe)
