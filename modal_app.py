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

# --- GPU worker image: only ollama + minimal Python deps ---
# Layer 1: OS packages (rarely changes)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    # Layer 2: Python deps (changes occasionally)
    .pip_install("ollama>=0.4.0")
    .env({"OLLAMA_MODELS": "/vol/ollama"})
    # Layer 3: Compile all .pyc bytecode for faster startup
    .run_commands("python -m compileall -q /usr/local/lib/python3.11")
)

# --- Orchestrator image: no GPU, just fastapi + ML inference deps ---
# Layer 1: Heavy scientific deps (rarely changes)
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scikit-learn==1.3.1", "joblib>=1.3")
    # Layer 2: Web framework (changes occasionally)
    .pip_install("fastapi[standard]")
    # Layer 3: Application files (changes frequently)
    .add_local_file("logistic_model.joblib", "/app/logistic_model.joblib")
    # Layer 4: Compile bytecode
    .run_commands("python -m compileall -q /usr/local/lib/python3.11")
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


# --- Class-based GPU workers with @enter() for model preloading ---

@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=300,
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
    scaledown_window=300,
)
class QwenClassifier:
    @modal.enter()
    def startup(self):
        self.proc = _start_ollama()
        _warmup_model("qwen3:8b")

    @modal.method()
    def classify(self, text: str, language: str) -> int:
        prompt = PROMPTS.get(language, PROMPTS["en"])
        return _classify("qwen3:8b", text, prompt)


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=300,
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


# --- Orchestrator endpoint: no GPU, just dispatches to model workers ---

@app.function(
    image=orchestrator_image,
    secrets=[secret],
    timeout=360,
)
@modal.fastapi_endpoint(method="POST")
async def analyze(data: dict, authorization: str = Header(default="")):
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

    # Spawn all 3 model calls in parallel
    gemma = GemmaClassifier().classify.spawn(text, language)
    qwen = QwenClassifier().classify.spawn(text, language)
    llama = LlamaClassifier().classify.spawn(text, language)

    results = [gemma.get(), qwen.get(), llama.get()]

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
