import modal
import subprocess
import os
import time
import urllib.request
import urllib.error
from difflib import get_close_matches

try:
    from fastapi import Header  # type: ignore
except ImportError:
    def Header(default=""):
        return default

app = modal.App("vazenespravy-processor")
secret = modal.Secret.from_name("vazenespravy")
ollama_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)

PROMPTS = {
    "sk": (
        "You are a highly talented Slovak journalist, and you want to classify articles as either liberal, "
        "neutral, or conservative. In your responses, you can only list a single classification, "
        "and cannot list any other words. For example, if you classify an article as politically neutral, "
        "you will only say the word neutral. If you classify an article as liberal, you will only "
        "say the word liberal. If you classify an article as conservative, you will also only say the word "
        "conservative. Under no circumstances should you list more than a single classification in your "
        "responses, or a single word other than the classification. Classify the following article:"
    ),
    "en": (
        "You are a highly talented journalist, and you want to classify articles as either liberal, "
        "neutral, or conservative. In your responses, you can only list a single classification, "
        "and cannot list any other words. For example, if you classify an article as politically neutral, "
        "you will only say the word neutral. If you classify an article as liberal, you will only "
        "say the word liberal. If you classify an article as conservative, you will also only say the word "
        "conservative. Under no circumstances should you list more than a single classification in your "
        "responses, or a single word other than the classification. Classify the following article:"
    ),
}

BIASES = ["conservative", "liberal", "neutral"]
BIAS_TO_NUM = {"liberal": -1, "neutral": 0, "conservative": 1}
NUM_TO_BIAS = {-1: "liberal", 0: "neutral", 1: "conservative"}

OLLAMA_VERSION = "0.15.1"

# --- GPU worker image: ollama + minimal Python deps ---
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    # Pinned ollama version for deterministic, cacheable layer
    .run_commands(
        f"curl -fsSL https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
        " | zstd -d | tar -x -C /usr/local"
    )
    .pip_install("ollama>=0.4.0")
    .env({
        "OLLAMA_MODELS": "/vol/ollama",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    .run_commands("python -m compileall -q /usr/local/lib/python3.11")
)

# --- Qwen host image: GPU image + orchestrator deps (fastapi, numpy, etc.) ---
qwen_host_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(
        f"curl -fsSL https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
        " | zstd -d | tar -x -C /usr/local"
    )
    .pip_install("ollama>=0.4.0", "numpy>=1.26", "scikit-learn==1.3.1", "joblib>=1.3")
    .pip_install("fastapi[standard]")
    .env({
        "OLLAMA_MODELS": "/vol/ollama",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    .run_commands("python -m compileall -q /usr/local/lib/python3.11")
    .add_local_file("logistic_model.joblib", "/app/logistic_model.joblib")
)


def _start_ollama():
    """Start Ollama server and wait for it to be ready."""
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/vol/ollama"

    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for i in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            print(f"Ollama ready after {i+1}s")
            return proc
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(1)

    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise RuntimeError(f"Ollama crashed: {stderr.decode()}")
    raise RuntimeError("Ollama failed to start within 60s")


def _warmup_model(model: str):
    """Send a tiny prompt to load model weights into memory."""
    import ollama  # type: ignore
    ollama.generate(model=model, prompt="hi", options={"num_predict": 1})
    print(f"Model {model} preloaded into memory")


def _classify(model: str, text: str, prompt: str) -> int:
    """Run classification with a specific model."""
    import ollama  # type: ignore

    result = ollama.generate(model=model, prompt=prompt + text, options={"temperature": 0.1})
    response = result["response"].split("</think>")[-1].strip().lower()
    matches = get_close_matches(response, BIASES, n=1)
    return BIAS_TO_NUM[matches[0]] if matches else 0


# --- GPU workers ---

@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=180,
    max_containers=1,
)
class GemmaClassifier:
    @modal.enter()
    def startup(self):
        self.proc = _start_ollama()
        _warmup_model("gemma3:12b")

    @modal.method()
    def classify(self, text: str, language: str) -> int:
        prompt = PROMPTS.get(language, PROMPTS["en"])
        return _classify("gemma3:12b", text, prompt)


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=180,
    max_containers=1,
)
class LlamaClassifier:
    @modal.enter()
    def startup(self):
        self.proc = _start_ollama()
        _warmup_model("llama3.1:8b")

    @modal.method()
    def classify(self, text: str, language: str) -> int:
        prompt = PROMPTS.get(language, PROMPTS["en"])
        return _classify("llama3.1:8b", text, prompt)


# --- Qwen host: runs qwen inference + serves the FastAPI endpoint + orchestrates ---

@app.cls(
    image=qwen_host_image,
    gpu="T4",
    secrets=[secret],
    volumes={"/vol/ollama": ollama_volume},
    timeout=360,
    scaledown_window=180,
    max_containers=1,
)
class QwenHost:
    @modal.enter()
    def startup(self):
        self.proc = _start_ollama()
        _warmup_model("qwen3:8b")

    @modal.fastapi_endpoint(method="POST")
    def analyze(self, data: dict, authorization: str = Header(default="")):
        """
        Classify text for political bias.

        Headers: Authorization: Bearer <api-key>
        Body: {"text": "...", "language": "sk"}
        Returns: {"politicalBias": "liberal" | "neutral" | "conservative"}
        """
        from fastapi import HTTPException  # type: ignore
        import numpy as np  # type: ignore
        from joblib import load  # type: ignore

        expected_key = os.environ.get("API_KEY", "")
        provided_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
        if not expected_key or provided_key != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

        text = data.get("text", "")
        language = data.get("language", "sk")

        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' field")

        prompt = PROMPTS.get(language, PROMPTS["en"])

        # Spawn gemma + llama on their GPU containers in parallel
        gemma = GemmaClassifier().classify.spawn(text, language)
        llama = LlamaClassifier().classify.spawn(text, language)

        # Run qwen locally on this container (already loaded in VRAM)
        qwen_result = _classify("qwen3:8b", text, prompt)

        # Collect results
        results = [gemma.get(), qwen_result, llama.get()]

        lr_model = load("/app/logistic_model.joblib")
        predictions = np.array([results])
        label = lr_model.predict(predictions)[0]

        return {"politicalBias": NUM_TO_BIAS[label]}


@app.function(
    image=gpu_image,
    volumes={"/vol/ollama": ollama_volume},
    timeout=1800,
)
def download_models():
    """Pre-download models. Run once: modal run modal_app.py::download_models"""
    import ollama  # type: ignore

    _start_ollama()

    for model in ["gemma3:12b", "qwen3:8b", "llama3.1:8b"]:
        print(f"Pulling {model}...")
        ollama.pull(model)
        print(f"Done: {model}")

    ollama_volume.commit()
    print("All models downloaded.")


@app.local_entrypoint()
def main():
    print("Downloading models...")
    download_models.remote()
    print("Done!")
