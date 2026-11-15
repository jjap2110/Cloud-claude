"""Tests de ampliar_fotos.py (Real-ESRGAN simulado: prueba la integración, no el modelo)."""

import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ampliar_fotos as af  # noqa: E402

FAKE = textwrap.dedent("""\
    #!{python}
    import sys
    from pathlib import Path
    from PIL import Image
    open({counter!r}, "a").write("x")
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    src, dst = Path(args["-i"]), Path(args["-o"])
    pairs = ([(f, dst / (f.stem + ".png")) for f in sorted(src.iterdir())]
             if src.is_dir() else [(src, dst)])
    for f, out in pairs:
        print("50.00%", file=sys.stderr, flush=True)
        with Image.open(f) as im:
            im.resize((im.width * 4, im.height * 4)).save(out)
""")


@pytest.fixture(autouse=True)
def fake_esrgan(tmp_path, monkeypatch):
    counter = tmp_path / "_calls.txt"
    exe = tmp_path / "_bin" / "realesrgan-ncnn-vulkan"
    exe.parent.mkdir()
    exe.write_text(FAKE.format(python=sys.executable, counter=str(counter)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(af, "find_realesrgan", lambda: exe)
    return counter


def make(path: Path, w=640, h=480, seed=0, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path, **kw)
    return path


def test_single_photo_goes_next_to_original(tmp_path):
    src = make(tmp_path / "fotos" / "a.jpg")
    assert af.upscale([src]) == 1
    with Image.open(tmp_path / "fotos" / "a_4K.jpg") as r:
        assert r.size == (2560, 1920)                 # ×4 (4K pediría ×4,5)
    assert not (tmp_path / "fotos" / af.OUTPUT_DIR_NAME).exists()


def test_folder_goes_to_ampliadas_keeping_structure(tmp_path, fake_esrgan):
    d = tmp_path / "viaje"
    make(d / "a.jpg", seed=1)
    make(d / "dia2" / "b.png", seed=2)
    make(d / "grande.jpg", 4000, 3000, seed=3)
    assert af.upscale([d]) == 2
    out = d / af.OUTPUT_DIR_NAME
    assert (out / "a_4K.jpg").exists() and (out / "dia2" / "b_4K.png").exists()
    assert not list(out.rglob("grande*"))             # ya era 4K: omitida
    assert (out / af.OUTPUT_MARKER).exists()
    assert fake_esrgan.read_text() == "x"             # un solo lote


def test_rerun_skips_done_and_never_reprocesses_outputs(tmp_path, fake_esrgan):
    d = tmp_path / "viaje"
    make(d / "a.jpg")
    af.upscale([d])
    af.upscale([d / "a.jpg"])                          # deja a_4K.jpg junto al original
    before = sorted(p.name for p in d.rglob("*.jpg"))
    assert af.upscale([d]) == 0
    assert sorted(p.name for p in d.rglob("*.jpg")) == before
    assert fake_esrgan.read_text() == "xx"


def test_mixed_folder_and_files_and_missing(tmp_path, capsys):
    d = tmp_path / "carpeta"
    make(d / "x.jpg", seed=1)
    f = make(tmp_path / "suelta.webp", seed=2)
    txt = tmp_path / "nota.txt"
    txt.write_text("hola")
    assert af.upscale([d, f, txt, tmp_path / "no_existe.jpg", f]) == 2
    out = capsys.readouterr().out
    assert "No es una imagen compatible: nota.txt" in out and "No existe" in out
    assert (tmp_path / "suelta_4K.webp").exists()


def test_keeps_exif_and_orientation(tmp_path):
    ex = Image.Exif()
    ex[274] = 6
    ex.get_ifd(0x8769)[36867] = "2019:05:05 12:00:00"
    src = make(tmp_path / "v.jpg", exif=ex.tobytes(), quality=95)
    af.upscale([src])
    with Image.open(tmp_path / "v_4K.jpg") as r:
        assert r.size == (1920, 2560)                  # se ve vertical; ×4
        assert r.getexif().get_ifd(0x8769).get(36867) == "2019:05:05 12:00:00"
        assert r.getexif().get(274) in (None, 1)


@pytest.mark.skipif(not af._HEIF, reason="pillow-heif no instalado")
def test_heic_input(tmp_path):
    src = make(tmp_path / "iphone.heic", format="HEIF")
    assert af.upscale([src]) == 1
    with Image.open(tmp_path / "iphone_4K.heic") as r:
        assert r.size == (2560, 1920)


@pytest.mark.parametrize("build", [
    lambda a, b: f"'{a}'",                              # arrastrada (Linux)
    lambda a, b: str(a).replace(" ", "\\ "),            # arrastrada (macOS)
    lambda a, b: str(a),                                # escrita con espacios
    lambda a, b: f"'{a}' '{b}'",                        # dos a la vez
    lambda a, b: str(a).replace(" ", "\\ ") + " " + str(b),
])
def test_parse_paths(tmp_path, build):
    a = make(tmp_path / "Mis Fotos" / "a b.jpg", 8, 8)
    b = make(tmp_path / "c.jpg", 8, 8)
    got = af.parse_paths(build(a, b) + " ")
    assert got[0] == a and got[1:] in ([], [b])


def test_interactive_loop(tmp_path, monkeypatch):
    src = make(tmp_path / "a.jpg")
    feed = iter([f"'{src}'", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(feed))
    assert af.main([]) == 0
    assert (tmp_path / "a_4K.jpg").exists()


def test_relaunches_with_venv_next_to_script(tmp_path):
    """Con una carpeta venv junto al script, se relanza con ese Python."""
    import shutil
    import subprocess
    app = tmp_path / "app"
    (app / "venv" / "bin").mkdir(parents=True)
    shutil.copy(Path(af.__file__), app / "ampliar_fotos.py")
    marker = tmp_path / "relanzado.txt"
    fake_py = app / "venv" / "bin" / "python"
    fake_py.write_text(f'#!/bin/sh\necho "$AMPLIAR_FOTOS_VENV|$@" > "{marker}"\n')
    fake_py.chmod(0o755)
    subprocess.run([sys.executable, str(app / "ampliar_fotos.py"), "foto con espacio.jpg"],
                   check=True, timeout=60)
    flag, args = marker.read_text().strip().split("|", 1)
    assert flag == "1"
    assert args.endswith("ampliar_fotos.py foto con espacio.jpg")
