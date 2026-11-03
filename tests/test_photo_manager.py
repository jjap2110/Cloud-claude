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
    # No detectar una instalación real de Real-ESRGAN en la máquina de tests
    monkeypatch.setattr(pm, "SCRIPT_DIR", tmp_path / "_script")
    monkeypatch.setattr(pm, "REALESRGAN_DIR", home / "realesrgan")
    monkeypatch.setattr(pm.shutil, "which", lambda *a, **k: None)
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
    # Simula realesrgan-ncnn-vulkan: archivo o carpeta, ×s, porcentajes en stderr
    import sys
    from pathlib import Path
    from PIL import Image
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    s = int(args["-s"])
    src, dst = Path(args["-i"]), Path(args["-o"])
    pairs = ([(f, dst / (f.stem + ".png")) for f in sorted(src.iterdir())]
             if src.is_dir() else [(src, dst)])
    for f, out in pairs:
        for pct in ("0.00%", "50.00%", "100.00%"):
            print(pct, file=sys.stderr, flush=True)
        with Image.open(f) as im:
            im.resize((im.width * s, im.height * s)).save(out)
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

    pm.run_enhance(src, out, "realesrgan", target="x2")

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


# ──────────────────────────────────────────────────────────────
# Rutas pegadas / arrastradas a la terminal
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("'/media/inst/Docs_Josmar/fotos/Alemania'", "/media/inst/Docs_Josmar/fotos/Alemania"),
    ("'/media/inst/fotos/Alemania'   ", "/media/inst/fotos/Alemania"),
    ('"/home/u/Mis Fotos"', "/home/u/Mis Fotos"),
    ("/home/u/Mis\\ Fotos", "/home/u/Mis Fotos"),
    ("file:///home/u/Mis%20Fotos", "/home/u/Mis Fotos"),
    ("& 'C:\\Users\\u\\Fotos'", "C:\\Users\\u\\Fotos"),
    ("/home/u/fotos", "/home/u/fotos"),
    ("", ""),
])
def test_clean_path_input(raw, expected):
    assert pm.clean_path_input(raw) == expected


def test_ask_folder_accepts_quoted_dragged_path(tmp_path, monkeypatch):
    d = tmp_path / "Mis Fotos"
    d.mkdir()
    monkeypatch.setattr("builtins.input", lambda _: f"'{d}' ")
    assert pm.ask_folder("Carpeta") == d.resolve()


# ──────────────────────────────────────────────────────────────
# Modo automático adaptativo
# ──────────────────────────────────────────────────────────────

def photo(w=2000, h=1500, gray_share=0.4, base=None, seed=0):
    """Foto sintética nítida: bloques grises neutros + bloques de color (BGR)."""
    rng = np.random.default_rng(seed)
    bh, bw = h // 25, w // 25
    lum = rng.integers(30, 230, (bh, bw, 1))
    img = np.repeat(lum, 3, axis=2).astype(np.int32)
    color = rng.random((bh, bw)) > gray_share
    hue = (np.broadcast_to(np.array(base), (bh, bw, 3)) if base is not None
           else rng.integers(0, 256, (bh, bw, 3)))
    img[color] = hue[color]
    small = np.clip(img, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.Resampling.NEAREST))


def kinds(actions):
    return {a.split(" (")[0] for a in actions}


def test_auto_leaves_good_photo_untouched():
    g = photo()
    out, _, actions = pm.enhance_adaptive(g)
    assert actions == []
    assert np.array_equal(out, g)


@pytest.mark.parametrize("name, base", [
    ("bosque", [40, 150, 50]),
    ("atardecer", [30, 110, 240]),
])
def test_auto_does_not_neutralize_intentional_colors(name, base):
    img = photo(gray_share=0.15, base=base)
    _, _, actions = pm.enhance_adaptive(img)
    assert "corregir dominante de color" not in kinds(actions), name


def test_auto_corrects_real_color_cast():
    img = np.clip(photo().astype(int) + [-12, 0, 22], 0, 255).astype(np.uint8)
    out, before, actions = pm.enhance_adaptive(img)
    assert "corregir dominante de color" in kinds(actions)
    assert pm.analyze_image(out)["cast"] < before["cast"] / 2


def test_auto_reduces_noise_only_when_noisy():
    rng = np.random.default_rng(1)
    g = photo()
    noisy = np.clip(g + rng.normal(0, 12, g.shape), 0, 255).astype(np.uint8)
    out, before, actions = pm.enhance_adaptive(noisy)
    assert "reducir ruido" in kinds(actions)
    assert pm.analyze_image(out)["noise"] < before["noise"] / 2


def test_auto_fixes_dark_and_flat():
    g = photo()
    for img in ((g * 0.35).astype(np.uint8), (g * 0.5 + 64).astype(np.uint8)):
        out, before, actions = pm.enhance_adaptive(img)
        after = pm.analyze_image(out)
        assert "ampliar rango tonal" in kinds(actions)
        assert after["range"] > before["range"] * 1.4
        assert 90 <= after["mean"] <= 185


def test_auto_sharpens_soft_photo():
    import cv2
    soft = cv2.GaussianBlur(photo(), (0, 0), 3)
    out, before, actions = pm.enhance_adaptive(soft)
    assert "enfocar" in kinds(actions)
    assert pm.analyze_image(out)["sharpness"] > before["sharpness"]


def test_auto_upscales_low_res_with_esrgan(tmp_path):
    exe = tmp_path / "realesrgan-ncnn-vulkan"
    exe.write_text(FAKE_ESRGAN.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    small = photo(800, 600)
    out, _, actions = pm.enhance_adaptive(small, esrgan=exe, target="fullhd")
    assert any(a.startswith("super-resolución") for a in actions)
    assert out.shape[:2] == (1080, 1440)      # cabe en 1920×1080


def test_diagnose_mode_writes_report_and_no_images(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.fromarray(photo()[:, :, ::-1]).save(src / "buena.jpg", quality=95)
    Image.fromarray((photo(seed=2) * 0.35).astype(np.uint8)).save(src / "oscura.jpg")
    pm.run_enhance(src, out, "diagnose")
    assert not list(out.glob("*.jpg"))
    with open(out / "mejora_reporte.csv", encoding="utf-8") as f:
        rows = {Path(r["archivo"]).name: r for r in csv.DictReader(f)}
    assert rows["buena.jpg"]["acciones"].startswith("ninguna")
    assert "ampliar rango tonal" in rows["oscura.jpg"]["acciones"]


def test_auto_run_writes_images_and_report(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.fromarray((photo() * 0.35).astype(np.uint8)).save(src / "oscura.jpg", quality=95)
    pm.run_enhance(src, out, "auto")
    assert (out / "oscura.jpg").exists()
    assert (out / "mejora_reporte.csv").exists()


def test_auto_copies_untouched_photos_byte_for_byte(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.fromarray(photo()[:, :, ::-1]).save(src / "buena.jpg", quality=92)
    pm.run_enhance(src, out, "auto")
    assert (out / "buena.jpg").read_bytes() == (src / "buena.jpg").read_bytes()


# ──────────────────────────────────────────────────────────────
# Localizar Real-ESRGAN
# ──────────────────────────────────────────────────────────────

def _fake_exe(path: Path, executable: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755 if executable else 0o644)
    return path


def test_find_realesrgan_next_to_script():
    exe = _fake_exe(pm.SCRIPT_DIR / "realesrgan" / "realesrgan-ncnn-vulkan")
    assert pm.find_realesrgan() == exe


def test_find_realesrgan_in_nested_zip_folder():
    exe = _fake_exe(pm.SCRIPT_DIR / "realesrgan"
                    / "realesrgan-ncnn-vulkan-20220424-ubuntu" / "realesrgan-ncnn-vulkan")
    assert pm.find_realesrgan() == exe


@pytest.mark.skipif(os.name == "nt", reason="permisos POSIX")
def test_find_realesrgan_restores_exec_permission():
    exe = _fake_exe(pm.SCRIPT_DIR / "realesrgan" / "realesrgan-ncnn-vulkan",
                    executable=False)
    assert pm.find_realesrgan() == exe
    assert os.access(exe, os.X_OK)


def test_find_realesrgan_none_when_missing():
    assert pm.find_realesrgan() is None


# ──────────────────────────────────────────────────────────────
# Resolución final (Full HD / 2K / 4K / ×2 / ×4)
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("w, h, target, size", [
    (640, 480, "fullhd", (1440, 1080)),     # 4:3 cabe por altura
    (960, 540, "fullhd", (1920, 1080)),     # 16:9 exacto
    (480, 640, "fullhd", (1080, 1440)),     # vertical: recuadro girado 1080×1920
    (1000, 750, "2k", (1920, 1440)),
    (1000, 750, "4k", (2880, 2160)),
    (800, 600, "x2", (1600, 1200)),
    (800, 600, "x4", (3200, 2400)),
])
def test_upscale_factor_fits_target_box(w, h, target, size):
    scale = pm.upscale_factor(w, h, target)
    assert pm.target_size(w, h, scale) == size


def _fake_esrgan_exe(tmp_path):
    exe = tmp_path / "fake_esrgan"
    exe.write_text(FAKE_ESRGAN.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_realesrgan_mode_resizes_to_target_and_names_file(tmp_path, monkeypatch):
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(_fake_esrgan_exe(tmp_path)))
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.fromarray(photo(640, 480)).save(src / "a.jpg", quality=95)
    pm.run_enhance(src, out, "realesrgan", target="fullhd")
    with Image.open(out / "a_FHD.jpg") as res:
        assert res.size == (1440, 1080)


def test_realesrgan_mode_caps_at_x4(tmp_path, monkeypatch):
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(_fake_esrgan_exe(tmp_path)))
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.fromarray(photo(320, 240)).save(src / "mini.jpg")   # 4K pediría ×9
    pm.run_enhance(src, out, "realesrgan", target="4k")
    with Image.open(out / "mini_4K.jpg") as res:
        assert res.size == (1280, 960)


def test_realesrgan_mode_copies_photos_already_at_target(tmp_path, monkeypatch):
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(_fake_esrgan_exe(tmp_path)))
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    Image.fromarray(photo(2000, 1500)).save(src / "grande.jpg", quality=92)
    pm.run_enhance(src, out, "realesrgan", target="fullhd")
    assert (out / "grande.jpg").read_bytes() == (src / "grande.jpg").read_bytes()
    assert not (out / "grande_FHD.jpg").exists()


def test_auto_does_not_upscale_photo_already_at_target(tmp_path):
    exe = _fake_esrgan_exe(tmp_path)
    _, a, actions = pm.enhance_adaptive(photo(2000, 1500), esrgan=exe, target="fullhd")
    assert not a["needs_upscale"]
    assert not any(x.startswith("super-resolución") for x in actions)
    # La misma foto sí necesita ampliarse para 4K
    _, a, actions = pm.enhance_adaptive(photo(2000, 1500), esrgan=exe,
                                        target="4k", dry_run=True)
    assert any(x.startswith("super-resolución") for x in actions)


def test_script_folder_has_priority_over_env(tmp_path, monkeypatch):
    local = _fake_exe(pm.SCRIPT_DIR / "realesrgan" / "realesrgan-ncnn-vulkan")
    other = _fake_exe(tmp_path / "otro" / "realesrgan-ncnn-vulkan")
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(other))
    assert pm.find_realesrgan() == local


# ──────────────────────────────────────────────────────────────
# Rendimiento: paralelo, lotes de Real-ESRGAN, indicador de avance
# ──────────────────────────────────────────────────────────────

def _many(src: Path, n: int = 6, w: int = 800, h: int = 600):
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = photo(w, h, seed=i)
        if i % 2:
            img = (img * 0.4).astype(np.uint8)
        Image.fromarray(img).save(src / f"f{i}.jpg", quality=92)


def test_parallel_auto_matches_serial(tmp_path):
    _many(tmp_path / "src")
    pm.run_enhance(tmp_path / "src", tmp_path / "s", "auto", workers=1)
    pm.run_enhance(tmp_path / "src", tmp_path / "p", "auto", workers=3)
    for f in sorted((tmp_path / "s").glob("*.jpg")):
        assert f.read_bytes() == (tmp_path / "p" / f.name).read_bytes(), f.name
    assert ((tmp_path / "s" / "mejora_reporte.csv").read_text()
            == (tmp_path / "p" / "mejora_reporte.csv").read_text())


def test_parallel_dedup_matches_serial(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _many(src, n=5)
    (src / "copia.jpg").write_bytes((src / "f0.jpg").read_bytes())
    found = {}
    for workers in (1, 3):
        monkeypatch.setattr(pm, "MAX_WORKERS", workers)
        monkeypatch.setattr(pm, "CACHE_PATH", tmp_path / f"cache{workers}.sqlite")
        pm.run_dedup(src, tmp_path / f"out{workers}", no_convert=True, dry_run=True)
        infos = [pm.get_image_info(p) for p in pm.find_images(src)]
        found[workers] = sorted(len(g) for g in pm.group_duplicates(infos, 6))
    assert found[1] == found[3]


def test_parallel_blur_detect(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "MAX_WORKERS", 3)
    src = tmp_path / "src"
    _many(src, n=5)
    Image.new("RGB", (400, 300), (90, 90, 90)).save(src / "plana.jpg")
    pm.run_blur_detect(src, threshold=100.0)
    with open(tmp_path / "blur_reporte_src.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert {r["estado"] for r in rows if r["archivo_original"].endswith("plana.jpg")} == {"BORROSA"}


COUNTING_ESRGAN = FAKE_ESRGAN.replace(
    "args = dict(", "open({counter!r}, 'a').write('x')\nargs = dict(")


def test_realesrgan_runs_once_per_batch(tmp_path, monkeypatch):
    counter = tmp_path / "calls.txt"
    exe = tmp_path / "fake_esrgan"
    exe.write_text(COUNTING_ESRGAN.replace("{counter!r}", repr(str(counter)))
                   .format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(pm.REALESRGAN_ENV, str(exe))
    src = tmp_path / "src"
    _many(src, n=5, w=640, h=480)
    pm.run_enhance(src, tmp_path / "out", "realesrgan", target="fullhd")
    assert counter.read_text() == "x"            # 5 fotos, 1 sola ejecución
    for i in range(5):
        with Image.open(tmp_path / "out" / f"f{i}_FHD.jpg") as res:
            assert res.size == (1440, 1080)


def test_realesrgan_batch_reports_progress(tmp_path):
    exe = _fake_esrgan_exe(tmp_path)
    seen = []
    ups = pm.realesrgan_batch([Image.fromarray(photo(64, 48))] * 2, exe, [2, 2],
                              on_progress=lambda done, frac: seen.append((done, frac)))
    assert [u.size for u in ups] == [(128, 96)] * 2
    assert (0, 0.5) in seen and any(d >= 1 for d, _ in seen)


def test_progress_non_tty_prints_by_deciles(capsys):
    prog = pm.Progress(100, "Test", min_interval=0)
    for i in range(100):
        prog.update(i, "x.jpg")
    prog.finish()
    out = capsys.readouterr().out.strip().splitlines()
    assert 5 <= len(out) <= 12
    assert "100%" in out[-1]


def test_realesrgan_falls_back_to_one_by_one(tmp_path):
    """Si el programa no acepta carpetas, se reintenta foto por foto."""
    exe = tmp_path / "file_only_esrgan"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys
        from pathlib import Path
        from PIL import Image
        a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
        if Path(a["-i"]).is_dir():
            print("decode image failed", file=sys.stderr); sys.exit(1)
        with Image.open(a["-i"]) as im:
            im.resize((im.width * 4, im.height * 4)).save(a["-o"])
    """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    ups = pm.realesrgan_batch([Image.fromarray(photo(64, 48))] * 3, exe, [2, 2, 2])
    assert [u.size for u in ups] == [(128, 96)] * 3
