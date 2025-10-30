# srtforge (Parakeet‑TDT‑0.6B‑V2)

srtforge is an end-to-end clone of the "alt+8" flow from
[mpv-parakeet-transcriber](https://github.com/StiensGate928/mpv-parakeet-transcriber)
packaged for automation. The pipeline follows the exact stages used by the
original project—English stream selection, PCM extraction, FV4 vocal
separation, FFmpeg preprocessing, Parakeet ASR, and the Netflix-style
post-processing stack—so the produced subtitles match the behavior of the
reference implementation.

## Automated setup

The repo ships with platform-specific installers that provision a virtual
environment, install PyTorch + NeMo, fetch the required Hugging Face assets, and
register the `srtforge` CLI in editable mode.

### Linux (and WSL) quick start

```bash
git clone <your-repo> srtforge
cd srtforge
./install.sh           # auto-detects GPU, use --cpu or --gpu to override
source .venv/bin/activate
srtforge --help
```

Optional: export `HF_TOKEN=<hugging-face-token>` before running the installer to
authenticate against private or rate-limited model downloads.

If a CUDA-capable GPU is detected, the installer provisions both the CUDA
`torch` wheels and `onnxruntime-gpu` so the FV4 separator can run on the GPU. Use
`--cpu` to force CPU wheels (and the CPU ONNX Runtime build) when debugging or
running on a headless machine.

### Windows 11 step-by-step

1. Install [Python 3.12 (recommended)](https://www.python.org/downloads/)—any
   interpreter in the 3.10–3.12 range works—and make sure “Add python.exe to
   PATH” is ticked.
2. Install [Git for Windows](https://git-scm.com/download/win) (enable “Git Bash”
   integration if you prefer a Unix-like shell).
3. Clone the repository:
   ```powershell
   git clone <your-repo> srtforge
   cd srtforge
   ```
4. (Optional) `setx HF_TOKEN <hugging-face-token>` if you need authenticated
   Hugging Face access.
5. Run the installer from an elevated PowerShell prompt if CUDA drivers are
   present, otherwise a normal prompt is fine. When a CUDA-capable GPU is
   detected the script installs the matching `torch` wheels and the
   `onnxruntime-gpu` package; otherwise it falls back to CPU builds and prints a
   warning:
   ```powershell
   ./install.ps1              # auto-detects GPU
   ./install.ps1 -Cpu         # force CPU wheels
   ./install.ps1 -Gpu         # force CUDA wheels
   ```
   The installer lists every compatible Python 3.10–3.12 interpreter it finds and
   lets you choose which one to use (defaulting to the newest). Pass
   `-PythonVersion 3.12` or `-PythonPath C:\Python312\python.exe` to skip the
   prompt and force a specific interpreter.
   If you see "Python was not found; run without arguments to install from the
   Microsoft Store" when the script starts, install Python 3.10–3.12 manually or
   disable the Windows App execution alias before retrying.
   If the script reports that it only found unsupported Python versions, install
   a 3.10–3.12 release or re-run the installer with `-PythonPath` pointing to an
   interpreter in that range.
6. Activate the environment and verify the CLI:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   srtforge --help
   ```

Both installers will download `parakeet-tdt-0.6b-v2.nemo` and
`voc_fv4.ckpt` from Hugging Face into `./models`. Re-running the script is safe;
existing files are left untouched.

## How it works

1. **English stream discovery** – `FFmpegTooling.probe_audio_streams` enumerates
   audio tracks via `ffprobe`, and the pipeline picks the first English-tagged
   stream. If none is found we skip the file to mirror the upstream script.
2. **PCM extraction** – the selected stream is extracted losslessly to 48 kHz
   stereo float (`pcm_f32le`) using `ffmpeg`. This matches the `alt+8` job that
   feeds downstream separation.
3. **Vocal separation** – the FV4 MelBand Roformer checkpoint (downloaded during
   install) is run through `audio-separator` to isolate vocals only, aligning
   with the `alt+8` vocal-only workflow.
4. **Preprocessing filters** – the isolated stem is high/low-pass filtered and
   resampled with SoXr to 16 kHz mono float to produce the exact waveform that
   Parakeet expects.
5. **Parakeet ASR** – `parakeet_to_srt_with_alt8` restores the
   Parakeet-TDT-0.6B-V2 model from the local `.nemo` (or downloads it if
   missing), requests timestamps, and reconstructs the segment/word structure
   identical to the upstream `alt+8` script.
6. **Netflix-style post processing** – we vendor `segmenter.py` and
   `srt_utils.py` directly from the reference project. The final SRT is produced
   by the same `postprocess_segments` routine with matching defaults so that
   timing, line balancing, and cps constraints behave identically.

The `srtforge` CLI exposes the pipeline for single files (`srtforge run`), bulk
jobs (`srtforge series`), and the Sonarr custom-script entry point.

## GPU vocal separation

Both the separation and ASR stages default to GPU execution when CUDA is
available. The CLI forwards `--cpu` to disable GPU use globally, while the
configuration file exposes dedicated toggles:

```yaml
separation:
  prefer_gpu: true
parakeet:
  prefer_gpu: true
```

Internally `FFmpegTooling.isolate_vocals` probes PyTorch for CUDA support and
passes `use_autocast=True` to `audio_separator` when `onnxruntime-gpu` is
present, logging whether CUDA, DirectML, or pure CPU execution is being used
before the FV4 stem is rendered.

### ONNX Runtime vs direct PyTorch FV4

mpv-parakeet-transcriber executes the FV4 Roformer checkpoint directly through
its vendored PyTorch modules (`mel_band_roformer.py`, `bs_roformer.py`, etc.),
whereas srtforge wraps the same checkpoint with `audio-separator`, an
ONNX Runtime-based runner. The ONNX approach simplifies distribution—hardware
providers are handled by `onnxruntime` and the installer can swap between the
CPU and CUDA builds automatically—while remaining compatible with the upstream
weights and configuration files that ship in `models/`. Users who prefer the
original PyTorch code path can disable ONNX acceleration by setting
`separation.prefer_gpu: false` and installing only the CPU runtime, but by
default the project will consume GPU resources whenever CUDA is detected to
match the mpv experience without manual setup.

## Usage examples

```bash
# Single media file (auto output path)
srtforge run /path/to/video.mkv

# Explicit output path
srtforge run /path/to/video.mkv --output subtitles/episode.srt

# Batch process a season directory
srtforge series "/shows/My Anime/Season 1" --glob "**/*.mkv"
```

Sonarr integration is available through `srtforge sonarr-hook`, which reads the
standard Sonarr environment variables and invokes the same pipeline.

## Sonarr custom script integration

Add srtforge as a [Sonarr custom script](https://wiki.servarr.com/sonarr/settings#connect)
so subtitles are generated automatically after downloads:

1. Install srtforge in the same environment that Sonarr can access (for
   example the `.venv` created by `./install.sh`).
2. In **Settings → Connect**, create a new **Custom Script**.
3. Set **Path** to either the `srtforge-sonarr` console script or the CLI
   wrapper (`srtforge sonarr-hook`).
4. Leave **Arguments** empty—the hook reads `EpisodeFile.Path` and
   `EventType` from the environment.
5. Enable the trigger events you care about (srtforge reacts to **On Import**
   and **On Upgrade**).

When Sonarr fires the custom script, srtforge resolves the episode file path,
normalizes the event name, and runs the standard pipeline to produce the SRT in
place.
