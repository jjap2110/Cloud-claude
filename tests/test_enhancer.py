"""Tests de enhancer.py. Los que usan el modelo se omiten si DeepFilterNet no está."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import enhancer as en  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
HAS_DF = all(importlib.util.find_spec(m) for m in ("df", "soundfile", "torch"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg no instalado")
needs_df = pytest.mark.skipif(not (HAS_FFMPEG and HAS_DF), reason="DeepFilterNet no instalado")


def make_video(path: Path, audio=True, subs=False, seconds=6):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
           f"testsrc=size=160x120:rate=10:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"anoisesrc=amplitude=0.1:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                "-filter_complex", "[1:a][2:a]join=inputs=2:channel_layout=stereo[a]"]
    if subs:
        srt = path.with_suffix(".srt")
        srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhola\n")
        cmd += ["-i", str(srt)]
    cmd += ["-map", "0:v"]
    if audio:
        cmd += ["-map", "[a]", "-c:a", "aac"]
    if subs:
        cmd += ["-map", "3:s" if audio else "1:s", "-c:s", "srt"]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", str(path)], check=True)
    return path


def streams(path):
    info = en.get_video_info(path)
    return [(s["codec_type"], s.get("channels")) for s in info["streams"]]


@pytest.mark.parametrize("value, expected", [("max", None), ("none", None), ("12", 12.0)])
def test_parse_atten(value, expected):
    assert en.parse_atten(value) == expected


@pytest.mark.parametrize("value", ["0", "-3", "fuerte"])
def test_parse_atten_rejects(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        en.parse_atten(value)


@needs_ffmpeg
def test_refuses_to_overwrite_input(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda m: True)
    v = make_video(tmp_path / "v.mp4")
    before = v.read_bytes()
    with pytest.raises(en.EnhancerError, match="mismo archivo"):
        en.process(v, v, quiet=True)
    link = tmp_path / "enlace.mp4"
    link.symlink_to(v)
    with pytest.raises(en.EnhancerError, match="mismo archivo"):
        en.process(v, link, quiet=True)
    assert v.read_bytes() == before


@needs_ffmpeg
def test_existing_output_and_missing_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda m: True)
    v = make_video(tmp_path / "v.mp4")
    out = tmp_path / "v_mejorado.mp4"
    out.write_text("ya existe")
    with pytest.raises(en.EnhancerError, match="Ya existe"):
        en.process(v, quiet=True)
    assert out.read_text() == "ya existe"
    mudo = make_video(tmp_path / "mudo.mp4", audio=False)
    with pytest.raises(en.EnhancerError, match="no tiene pista de audio"):
        en.process(mudo, quiet=True)


@needs_ffmpeg
def test_ffmpeg_errors_show_reason(tmp_path):
    bad = tmp_path / "roto.mp4"
    bad.write_text("esto no es un video")
    with pytest.raises(en.EnhancerError) as e:
        en.get_video_info(bad)
    assert "roto.mp4" in str(e.value) or "Invalid" in str(e.value)


def test_main_reports_errors_without_traceback(tmp_path, capsys):
    assert en.main([str(tmp_path / "no_existe.mp4")]) == 1
    assert "❌" in capsys.readouterr().err


@needs_df
def test_end_to_end_keeps_stereo_video_and_subtitles(tmp_path):
    v = make_video(tmp_path / "v.mkv", subs=True)
    out = en.process(v, quiet=True)
    assert out == tmp_path / "v_mejorado.mkv"
    assert streams(out) == [("video", None), ("audio", 2), ("subtitle", None)]
    d_in = float(en.get_video_info(v)["format"]["duration"])
    d_out = float(en.get_video_info(out)["format"]["duration"])
    assert abs(d_in - d_out) < 0.2


@needs_df
def test_chunked_matches_single_pass(tmp_path):
    import numpy as np
    import soundfile as sf
    v = make_video(tmp_path / "v.mp4", seconds=10)
    model, state, _ = en.load_model()
    wav = tmp_path / "in.wav"
    en.extract_audio(v, wav, int(state.sr()), 2)
    en.enhance_wav(wav, tmp_path / "uno.wav", model, state, chunk_s=1000)
    en.enhance_wav(wav, tmp_path / "tramos.wav", model, state, chunk_s=3)
    a, b = sf.read(tmp_path / "uno.wav")[0], sf.read(tmp_path / "tramos.wav")[0]
    assert a.shape == b.shape == sf.read(wav)[0].shape
    err = 10 * np.log10(np.mean((a - b) ** 2) + 1e-12)
    sig = 10 * np.log10(np.mean(a ** 2) + 1e-12)
    assert err < sig - 20
