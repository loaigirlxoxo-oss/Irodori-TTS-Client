"""Quick check whether silero-vad and faster-whisper are installed."""
import importlib

for pkg in ["silero_vad", "faster_whisper", "torch", "torchaudio"]:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"OK   {pkg}: {ver}  -> {m.__file__}")
    except Exception as e:
        print(f"MISS {pkg}: {type(e).__name__}: {e}")
