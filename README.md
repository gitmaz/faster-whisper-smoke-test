# faster-whisper-smoke-test

A small GPU check for the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) package. It transcribes the included 11-second Kennedy clip and prints the text, language, and timing.

This folder is not used by `faster-whisper-api`. That service has its own virtualenv and its own copy of the package.

## Setup

Requires an NVIDIA GPU visible in this environment, Python 3.14, and [uv](https://docs.astral.sh/uv/).

From this directory:

```bash
uv venv --python 3.14 .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

The first transcription of a model downloads its weights from Hugging Face. Later runs reuse that cache.

## Run

```bash
./run_gpu_test.sh
```

The script points `LD_LIBRARY_PATH` at the CUDA libraries inside `.venv`, then runs `test_gpu.py` on `samples/jfk.flac`. The default model is `small` in `float16`. The first GPU pass warms CUDA up. The printed time is the second pass.

A working run reports one CUDA device and this text:

> And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

Use another model with `WHISPER_MODEL`:

```bash
WHISPER_MODEL=medium ./run_gpu_test.sh
```

`WHISPER_COMPUTE` overrides the compute type. The default is `float16`.
