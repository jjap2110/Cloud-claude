#!/usr/bin/env python3
"""
enhancer.py — Mejora el audio de un video con DeepFilterNet (v7).

Notas v7:
  • DeepFilterNet NO es compatible con MPS: el propio modelo llama
    a .numpy() sobre tensores en GPU, lo que PyTorch no permite.
  • Por defecto se usa CPU. Si el usuario fuerza --device mps, se
    avisa y se hace fallback automático a CPU.
  • Se elimina toda la complejidad de MPS del flujo, ya que no sirve.
"""

import argparse, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")


# ----------------------------------------------------------------
#  Selección de device (solo CPU y CUDA son viables)
# ----------------------------------------------------------------
def pick_device(prefer=None):
    """
    Devuelve el device a usar.
      - 'auto' → cuda si hay GPU NVIDIA, si no CPU.
      - 'mps'  → se avisa y se devuelve 'cpu' (DeepFilterNet no soporta MPS).
      - 'cuda' / 'cpu' → se respeta tal cual.
    """
    import torch

    if prefer == "mps":
        print("⚠️  DeepFilterNet no es compatible con MPS.")
        print("   El modelo usa .numpy() internamente sobre tensores de la GPU.")
        print("   Se usará CPU automáticamente.\n")
        return "cpu"

    if prefer and prefer != "auto":
        return prefer

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def describe_device(device):
    return {
        "cuda": "CUDA (GPU NVIDIA)",
        "cpu": "CPU",
    }.get(device, device)


# ----------------------------------------------------------------
#  Temporizador de progreso
# ----------------------------------------------------------------
class ElapsedTimer:
    def __init__(self, message="Procesando", enabled=True):
        self.message = message
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.start_time = None
        self.thread = None
        self._last_len = 0

    def _run(self):
        while not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            line = f"\r{self.message}... [{mins:02d}:{secs:02d}]"
            pad = max(0, self._last_len - len(line))
            print(line + " " * pad, end="", flush=True)
            self._last_len = len(line)
            time.sleep(0.5)

    def __enter__(self):
        if not self.enabled:
            return self
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        if not self.enabled:
            return
        self.stop_event.set()
        self.thread.join()
        print("\r" + " " * self._last_len + "\r", end="", flush=True)


# ----------------------------------------------------------------
#  Utilidades de video
# ----------------------------------------------------------------
def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def extract_audio(video_path, output_wav, duration=None, sr=48000):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn",
           "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(output_wav)
    subprocess.run(cmd, check=True, capture_output=True)


def extract_audio_segments(video_path, output_wav, duration_total,
                            n_segments=5, seg_duration=10, sr=48000):
    import numpy as np

    margin = 5.0
    usable = duration_total - 2 * margin

    if usable <= seg_duration:
        extract_audio(video_path, output_wav,
                      duration=min(duration_total, seg_duration), sr=sr)
        return 1

    max_segments = max(1, int(usable / seg_duration))
    n_segments = min(n_segments, max_segments)

    if n_segments == 1:
        start = margin
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video_path,
               "-t", str(seg_duration), "-vn",
               "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
               output_wav]
        subprocess.run(cmd, check=True, capture_output=True)
        return 1

    starts = np.linspace(margin,
                         duration_total - margin - seg_duration,
                         n_segments)

    tmp_files = []
    for start in starts:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            seg_file = t.name
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video_path,
               "-t", str(seg_duration), "-vn",
               "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
               seg_file]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp_files.append(seg_file)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as t:
        list_file = t.name
        for f in tmp_files:
            t.write(f"file '{f}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", list_file, "-c", "copy", output_wav]
    subprocess.run(cmd, check=True, capture_output=True)

    for f in tmp_files + [list_file]:
        if os.path.exists(f):
            os.unlink(f)

    return n_segments


def remux_video(video_path, audio_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path]
    subprocess.run(cmd, check=True, capture_output=True)


# ----------------------------------------------------------------
#  DeepFilterNet
# ----------------------------------------------------------------
_MODEL_CACHE = {}


def load_model(variant="DeepFilterNet3", prefer_device=None):
    device = pick_device(prefer_device)
    key = (variant, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from df.enhance import init_df
    model, df_state, _ = init_df(
        model_base_dir=variant, post_filter=True, log_level="error",
    )
    model = model.to(device).eval()
    _MODEL_CACHE[key] = (model, df_state, device)
    return model, df_state, device


def _to_numpy(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).flatten()


def enhance_audio(input_audio, atten_lim, variant="DeepFilterNet3",
                  prefer_device=None):
    from df.enhance import enhance
    from df.io import load_audio
    model, df_state, device = load_model(variant, prefer_device)
    target_sr = int(df_state.sr())
    audio, _meta = load_audio(input_audio, sr=target_sr)

    # Solo movemos a device si NO es CPU (para cuda)
    if hasattr(audio, "to") and device == "cuda":
        audio = audio.to(device)

    enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim)
    return _to_numpy(enhanced), target_sr


def snr_proxy(original, enhanced):
    import numpy as np
    o = _to_numpy(original)
    e = _to_numpy(enhanced)
    n = min(len(o), len(e))
    o, e = o[:n], e[:n]
    p_sig = float(np.mean(e.astype(np.float64) ** 2)) + 1e-10
    p_noise = float(np.mean((o.astype(np.float64) - e.astype(np.float64)) ** 2)) + 1e-10
    return 10 * np.log10(p_sig / p_noise)


# ----------------------------------------------------------------
#  Optimización
# ----------------------------------------------------------------
def optimize(sample_wav, variant="DeepFilterNet3", quiet=False,
             prefer_device=None):
    from df.io import load_audio
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    original, _meta = load_audio(sample_wav, sr=48000)
    original = _to_numpy(original)

    atten_values = [6.0, 12.0, 18.0, 24.0, None]
    best_score, best_atten = -float("inf"), None
    scores = {}

    device = pick_device(prefer_device)
    if not quiet:
        print(f"🔬 Optimizando {len(atten_values)} valores de atten_lim "
              f"en {describe_device(device)}...")

    iterator = atten_values
    if has_tqdm and not quiet:
        iterator = tqdm(atten_values, desc="   Progreso", unit="paso",
                        bar_format="{l_bar}{bar:30}{r_bar}")

    for atten in iterator:
        try:
            enh, _ = enhance_audio(sample_wav, atten, variant, prefer_device)
            score = snr_proxy(original, enh)
            scores[atten] = score
            if score > best_score:
                best_score, best_atten = score, atten
        except Exception as e:
            if not quiet:
                print(f"\n   ⚠️  atten={atten}: {e}")

    if not quiet:
        print()
        for atten, score in scores.items():
            marker = " ✅" if atten == best_atten else ""
            print(f"   atten={str(atten):6s} → SNR: {score:6.2f} dB{marker}")

    if best_atten is None:
        print("⚠️  Optimización fallida, usando atten_lim=12.0 por defecto")
        return 12.0

    if not quiet:
        print(f"\n✅ Mejor atten_lim={best_atten} (SNR proxy: {best_score:.2f} dB)\n")
    return best_atten


# ----------------------------------------------------------------
#  Flujo principal
# ----------------------------------------------------------------
def process(video_in, video_out, sample_s=50, n_segments=5,
            variant="DeepFilterNet3", quiet=False, prefer_device=None):
    if not quiet:
        print("=" * 60)
        print("  DeepFilterNet Video Audio Enhancer")
        print("=" * 60)

    info = get_video_info(video_in)
    duration = float(info["format"]["duration"])
    size_mb = os.path.getsize(video_in) / (1024 * 1024)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)

    device = pick_device(prefer_device)

    if not quiet:
        print(f"📊 Video original:")
        print(f"   • Duración:   {duration:.1f} s ({duration/60:.1f} min)")
        print(f"   • Tamaño:     {size_mb:.1f} MB")
        if a:
            print(f"   • Códec:      {a.get('codec_name', '?')}")
            print(f"   • Frecuencia: {a.get('sample_rate', '?')} Hz")
            print(f"   • Canales:    {a.get('channels', '?')}")
        print(f"   • Device:     {describe_device(device)}")

    seg_dur = max(3, sample_s // n_segments)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        sample_wav = t.name
    try:
        if not quiet:
            print(f"\n🎧 Extrayendo {n_segments} fragmentos de ~{seg_dur} s "
                  f"distribuidos por el video...")
        actual_n = extract_audio_segments(
            video_in, sample_wav, duration,
            n_segments=n_segments, seg_duration=seg_dur
        )
        if not quiet and actual_n < n_segments:
            print(f"   (video corto: solo se pudieron extraer {actual_n})")

        best_atten = optimize(sample_wav, variant, quiet=quiet,
                              prefer_device=prefer_device)
    finally:
        if os.path.exists(sample_wav):
            os.unlink(sample_wav)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        full_wav = t.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        enh_wav = t.name

    try:
        if not quiet:
            print("🎧 Extrayendo audio completo...")
        extract_audio(video_in, full_wav)

        if not quiet:
            with ElapsedTimer("🧠 Aplicando DeepFilterNet", enabled=True):
                enhanced, sr = enhance_audio(full_wav, best_atten, variant,
                                             prefer_device)
        else:
            enhanced, sr = enhance_audio(full_wav, best_atten, variant,
                                         prefer_device)

        import soundfile as sf
        sf.write(enh_wav, enhanced, int(sr))

        from df.io import load_audio
        original_full, _meta = load_audio(full_wav, sr=48000)
        snr_final = snr_proxy(original_full, enhanced)

        if not quiet:
            print("🎬 Reensamblando video...")
        remux_video(video_in, enh_wav, video_out)
    finally:
        for f in (full_wav, enh_wav):
            if os.path.exists(f):
                os.unlink(f)

    print("\n" + "=" * 60)
    print("  ✅ PROCESO COMPLETADO")
    print("=" * 60)
    print(f"   • Original:  {video_in}")
    print(f"   • Mejorado:  {video_out}")
    print(f"   • Params:    atten_lim={best_atten} (post_filter activado)")
    print(f"   • Modelo:    {variant}")
    print(f"   • Device:    {describe_device(device)}")
    print(f"   • SNR proxy: {snr_final:.2f} dB")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--segments", type=int, default=5)
    p.add_argument("--model", default="DeepFilterNet3",
                   choices=["DeepFilterNet2", "DeepFilterNet3"])
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"],
                   help="Device (default: auto). MPS no soportado por DeepFilterNet.")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    prefer = None if a.device == "auto" else a.device
    process(a.input, a.output, sample_s=a.sample, n_segments=a.segments,
            variant=a.model, quiet=a.quiet, prefer_device=prefer)


if __name__ == "__main__":
    main()
