#!/usr/bin/env python3
"""
enhancer.py — Limpia el ruido del audio de un video con DeepFilterNet (v8).

Uso:
  python enhancer.py video.mp4                       → video_mejorado.mp4
  python enhancer.py video.mp4 -o salida.mp4
  python enhancer.py video.mp4 --atten 12            → limpieza más suave

Cambios v8:
  • Se elimina la "optimización" de atten_lim: su métrica (potencia de lo
    conservado / potencia de lo eliminado) premiaba limpiar menos y elegía
    siempre el valor más débil. Ahora --atten se elige a mano (de oído).
  • Se conserva el estéreo (antes se convertía todo a mono).
  • Nunca sobrescribe el original, ni una salida existente sin --overwrite.
  • El audio se procesa por tramos: la memoria no crece con la duración.
  • Los errores de ffmpeg se muestran; se conservan los subtítulos.

Notas:
  • DeepFilterNet NO es compatible con MPS (llama a .numpy() sobre tensores
    en GPU). Se usa CPU; con --device mps se avisa y se usa CPU.
  • Requiere ffmpeg/ffprobe en el PATH y: pip install deepfilternet soundfile
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CHUNK_S = 120.0      # duración de cada tramo de proceso
CONTEXT_S = 2.0      # audio previo que se da al modelo como contexto y se descarta
OVERLAP_S = 0.5      # solape entre tramos, mezclado con fundido cruzado


# ----------------------------------------------------------------
#  Utilidades
# ----------------------------------------------------------------
class EnhancerError(Exception):
    """Error esperado: se muestra sin traza."""


def run(cmd, what):
    """Ejecuta ffmpeg/ffprobe y, si falla, muestra el motivo real."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except FileNotFoundError:
        raise EnhancerError(f"No se encontró '{cmd[0]}'. Instala ffmpeg "
                            "(macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg).")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-6:])
        raise EnhancerError(f"{what} falló (código {r.returncode}):\n{tail}")
    return r


def fmt_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


# ----------------------------------------------------------------
#  Selección de device (solo CPU y CUDA son viables)
# ----------------------------------------------------------------
def pick_device(prefer=None):
    """
    Devuelve el device a usar.
      - None/'auto' → cuda si hay GPU NVIDIA, si no CPU.
      - 'mps'  → se avisa y se devuelve 'cpu' (DeepFilterNet no soporta MPS).
      - 'cuda' / 'cpu' → se respeta tal cual.
    """
    import torch

    if prefer == "mps":
        print("⚠️  DeepFilterNet no es compatible con MPS; se usará CPU.\n")
        return "cpu"
    if prefer and prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device):
    return {"cuda": "CUDA (GPU NVIDIA)", "cpu": "CPU"}.get(device, device)


# ----------------------------------------------------------------
#  Video
# ----------------------------------------------------------------
def get_video_info(video_path):
    r = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(video_path)], "Leer el video")
    return json.loads(r.stdout)


def extract_audio(video_path, output_wav, sr, channels):
    run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-map", "0:a:0",
         "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", str(channels),
         str(output_wav)], "Extraer el audio")


def remux_video(video_path, audio_path, output_path, channels):
    """Video original (sin recomprimir) + audio nuevo. Conserva subtítulos
    si el contenedor de salida los admite; si no, reintenta sin ellos."""
    bitrate = "192k" if channels >= 2 else "128k"

    def cmd(with_subs):
        c = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-map", "0:v?", "-map", "1:a:0"]
        if with_subs:
            c += ["-map", "0:s?", "-c:s", "copy"]
        return c + ["-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
                    "-map_metadata", "0", str(output_path)]

    try:
        run(cmd(True), "Reensamblar el video")
        return True
    except EnhancerError:
        run(cmd(False), "Reensamblar el video")
        return False


# ----------------------------------------------------------------
#  DeepFilterNet
# ----------------------------------------------------------------
_MODEL_CACHE = {}


def load_model(variant="DeepFilterNet3", prefer_device=None):
    device = pick_device(prefer_device)
    key = (variant, device)
    if key not in _MODEL_CACHE:
        from df.enhance import init_df
        model, df_state, _ = init_df(
            model_base_dir=variant, post_filter=True, log_level="error",
        )
        _MODEL_CACHE[key] = (model.to(device).eval(), df_state, device)
    return _MODEL_CACHE[key]


def enhance_wav(in_wav, out_wav, model, df_state, atten_lim=None, progress=None,
                chunk_s=CHUNK_S, context_s=CONTEXT_S, overlap_s=OVERLAP_S):
    """Limpia un WAV por tramos y lo escribe según avanza.

    Cada tramo se procesa con `context_s` de audio anterior (el modelo
    arranca sin memoria y necesita contexto; esa parte se descarta) y se
    une al siguiente con un fundido cruzado de `overlap_s`, para que no se
    oiga el corte. La memoria usada no depende de la duración del video.
    El audio va a DeepFilterNet en CPU: él mismo lo pasa a NumPy.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from df.enhance import enhance

    with sf.SoundFile(str(in_wav)) as fin:
        sr, total, channels = fin.samplerate, fin.frames, fin.channels
        chunk = max(1, int(chunk_s * sr))
        ctx = int(context_s * sr)
        ov = int(overlap_s * sr)
        with sf.SoundFile(str(out_wav), "w", samplerate=sr, channels=channels,
                          subtype="PCM_16") as fout:
            pos, prev_tail = 0, None
            while pos < total:
                start = max(0, pos - ctx)
                fin.seek(start)
                block = fin.read(pos - start + chunk + ov, dtype="float32",
                                 always_2d=True)                    # [T, C]
                audio = torch.from_numpy(np.ascontiguousarray(block.T))  # [C, T]
                enh = enhance(model, df_state, audio, atten_lim_db=atten_lim)
                if hasattr(enh, "detach"):
                    enh = enh.detach().cpu().numpy()
                enh = np.asarray(enh, dtype=np.float32).reshape(channels, -1).T
                if len(enh) < len(block):                           # por si recorta
                    enh = np.pad(enh, ((0, len(block) - len(enh)), (0, 0)))
                enh = enh[pos - start:len(block)]                   # quitar contexto

                if prev_tail is not None and len(prev_tail):
                    n = min(len(prev_tail), len(enh))
                    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                    enh[:n] = prev_tail[:n] * (1 - fade) + enh[:n] * fade

                if start + len(block) >= total:                     # último tramo
                    fout.write(enh)
                    pos = total
                else:
                    fout.write(enh[:chunk])
                    prev_tail = enh[chunk:chunk + ov].copy()
                    pos += chunk
                if progress:
                    progress(min(pos, total) / total)
    return sr, channels


# ----------------------------------------------------------------
#  Flujo principal
# ----------------------------------------------------------------
def default_output(video_in):
    p = Path(video_in)
    return p.with_name(f"{p.stem}_mejorado{p.suffix}")


def process(video_in, video_out=None, atten_lim=None, variant="DeepFilterNet3",
            quiet=False, prefer_device=None, overwrite=False):
    video_in = Path(video_in).expanduser()
    video_out = Path(video_out).expanduser() if video_out else default_output(video_in)

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise EnhancerError(f"No se encontró '{tool}'. Instala ffmpeg "
                                "(macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg).")
    import importlib.util
    missing = [pkg for pkg, mod in (("deepfilternet", "df"), ("soundfile", "soundfile"))
               if not importlib.util.find_spec(mod)]
    if missing:
        raise EnhancerError("Faltan dependencias. Instala con:  pip install "
                            + " ".join(missing))
    if not video_in.is_file():
        raise EnhancerError(f"No existe el video: {video_in}")
    if video_out.resolve() == video_in.resolve() or (
            video_out.exists() and os.path.samefile(video_in, video_out)):
        raise EnhancerError("La salida es el mismo archivo que la entrada: "
                            "se destruiría el original. Elige otro nombre con -o.")
    if video_out.exists() and not overwrite:
        raise EnhancerError(f"Ya existe {video_out}. Usa --overwrite para reemplazarlo.")

    info = get_video_info(video_in)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if a is None:
        raise EnhancerError("El video no tiene pista de audio.")
    duration = float(info["format"].get("duration") or 0)
    src_channels = int(a.get("channels") or 1)
    channels = min(src_channels, 2)

    if not quiet:
        print("=" * 60)
        print("  DeepFilterNet Video Audio Enhancer")
        print("=" * 60)
        print("📊 Video original:")
        print(f"   • Duración:   {duration:.1f} s ({duration / 60:.1f} min)")
        print(f"   • Tamaño:     {video_in.stat().st_size / 1024 ** 2:.1f} MB")
        print(f"   • Códec:      {a.get('codec_name', '?')}")
        print(f"   • Frecuencia: {a.get('sample_rate', '?')} Hz")
        print(f"   • Canales:    {src_channels}"
              + (" → se mezclan a estéreo" if src_channels > 2 else ""))
        print("\n🧠 Cargando modelo...")

    model, df_state, device = load_model(variant, prefer_device)
    sr = int(df_state.sr())
    atten_txt = "sin límite (máxima limpieza)" if atten_lim is None else f"{atten_lim:g} dB"
    if not quiet:
        print(f"   • Modelo:     {variant} en {describe_device(device)}")
        print(f"   • Atenuación: {atten_txt}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="enhancer_"))
    try:
        full_wav, enh_wav = tmp_dir / "original.wav", tmp_dir / "limpio.wav"
        if not quiet:
            print("\n🎧 Extrayendo audio...")
        extract_audio(video_in, full_wav, sr, channels)

        t0 = time.time()

        def progress(frac):
            if quiet:
                return
            elapsed = time.time() - t0
            eta = f" | quedan ~{fmt_time(elapsed / frac - elapsed)}" if 0.02 < frac < 1 else ""
            print(f"\r🧠 Limpiando audio... {frac * 100:3.0f}% [{fmt_time(elapsed)}]{eta}   ",
                  end="", flush=True)

        progress(0.0)
        enhance_wav(full_wav, enh_wav, model, df_state, atten_lim, progress)
        if not quiet:
            print("\n🎬 Reensamblando video...")
        subs_kept = remux_video(video_in, enh_wav, video_out, channels)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("  ✅ PROCESO COMPLETADO")
    print("=" * 60)
    print(f"   • Original:   {video_in}")
    print(f"   • Mejorado:   {video_out}")
    print(f"   • Atenuación: {atten_txt}")
    print(f"   • Modelo:     {variant} en {describe_device(device)}")
    if not subs_kept:
        print("   • Aviso: los subtítulos no se pudieron copiar a este formato")
    print("=" * 60)
    return video_out


def parse_atten(value):
    if value.lower() in ("none", "max"):
        return None
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("usa un número de dB (p. ej. 12) o 'max'")
    if v <= 0:
        raise argparse.ArgumentTypeError("debe ser mayor que 0")
    return v


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Limpia el ruido del audio de un video con DeepFilterNet.")
    p.add_argument("input", help="video de entrada")
    p.add_argument("-o", "--output",
                   help="video de salida (por defecto: <nombre>_mejorado.<ext>)")
    p.add_argument("--atten", type=parse_atten, default=None, metavar="dB",
                   help="máximo de ruido a quitar, en dB: 6 = muy suave, 12 = suave, "
                        "24 = fuerte, 'max' = sin límite (por defecto). Si la voz "
                        "suena artificial, baja este valor.")
    p.add_argument("--model", default="DeepFilterNet3",
                   choices=["DeepFilterNet2", "DeepFilterNet3"])
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"],
                   help="por defecto auto. MPS no está soportado por DeepFilterNet.")
    p.add_argument("--overwrite", action="store_true",
                   help="reemplazar el archivo de salida si ya existe")
    p.add_argument("--quiet", action="store_true")
    # Opciones de v7 (optimización automática): se aceptan pero ya no hacen nada
    p.add_argument("--sample", type=int, help=argparse.SUPPRESS)
    p.add_argument("--segments", type=int, help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.sample is not None or a.segments is not None:
        print("ℹ️  --sample y --segments ya no se usan (se eliminó la optimización "
              "automática). Elige la intensidad con --atten.\n")
    try:
        process(a.input, a.output, atten_lim=a.atten, variant=a.model, quiet=a.quiet,
                prefer_device=None if a.device == "auto" else a.device,
                overwrite=a.overwrite)
    except EnhancerError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrumpido.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
