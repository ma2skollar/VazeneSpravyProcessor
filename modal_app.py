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

# OPTIMIZATION 1: Layer caching - build from least to most frequently changing
# OPTIMIZATION 2: Lighter base image - debian_slim is already good
# OPTIMIZATION 3: Prebuild all dependencies
# OPTIMIZATION 4: Compile Python code
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Layer 1: System dependencies (rarely change)
    .apt_install("curl", "ca-certificates", "zstd")
    # Layer 2: Install Ollama (rarely changes)
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    # Layer 3: Python dependencies (change occasionally)
    # OPTIMIZATION 3: Minimize dependencies - only install what's needed
    .pip_install(
        "ollama==0.4.4",  # Pin exact versions for reproducibility
        "numpy==1.26.4",
        "scikit-learn==1.3.1",
        "joblib==1.3.2",
        "fastapi==0.115.6",
        "uvicorn[standard]==0.32.1",
    )
    # OPTIMIZATION 4: Compile Python files (do before adding local files)
    .run_commands(
        "python -m compileall -b /usr/local/lib/python3.11/site-packages || true",
        "find /usr/local/lib/python3.11/site-packages -name '*.py' -delete || true"
    )
    # Layer 4: Environment variables
    .env({"OLLAMA_MODELS": "/vol/ollama"})
    # Layer 5: Application files (change most frequently - put LAST)
    # copy=True allows running commands after, but adds to build time
    .add_local_file("logistic_model.joblib", "/app/logistic_model.joblib", copy=True)
)


def start_ollama():
    """Start Ollama server and wait for it to be ready."""
    print("Starting Ollama server...")
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/vol/ollama"

    process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Reduced timeout since models are preloaded
    for i in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            print(f"Ollama ready after {i+1}s")
            return process
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(1)

    if process.poll() is not None:
        _, stderr = process.communicate()
        raise RuntimeError(f"Ollama crashed: {stderr.decode()}")
    raise RuntimeError("Ollama failed to start within 30s")


# OPTIMIZATION 5: Model preloading using modal.Cls with @enter()
@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    container_idle_timeout=180,  # Modal 1.0: previously scaledown_window
)
class GemmaClassifier:
    @modal.enter()
    def startup(self):
        """Runs once on container start - preload everything here."""
        import ollama  # type: ignore
        
        print("Initializing Gemma classifier...")
        self.ollama_process = start_ollama()
        
        # Verify model is available
        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]
        if "gemma3:12b" not in model_names:
            print("WARNING: gemma3:12b not found in cache, will be slow on first run")
        
        self.model_name = "gemma3:12b"
        print("Gemma classifier ready")

    @modal.method()
    def classify(self, text: str, language: str) -> int:
        """Classification method - model already loaded."""
        import ollama  # type: ignore
        
        prompt = PROMPTS.get(language, PROMPTS["en"])
        result = ollama.generate(
            model=self.model_name,
            prompt=prompt + text,
            options={"temperature": 0.1}
        )
        response = result["response"].split("</think>")[-1].strip().lower()
        matches = get_close_matches(response, BIASES, n=1)
        return BIAS_TO_NUM[matches[0]] if matches else 0

    @modal.exit()
    def shutdown(self):
        """Cleanup on container shutdown."""
        if hasattr(self, 'ollama_process') and self.ollama_process:
            self.ollama_process.terminate()


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    container_idle_timeout=180,
)
class QwenClassifier:
    @modal.enter()
    def startup(self):
        import ollama  # type: ignore
        
        print("Initializing Qwen classifier...")
        self.ollama_process = start_ollama()
        
        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]
        if "qwen3:8b" not in model_names:
            print("WARNING: qwen3:8b not found in cache")
        
        self.model_name = "qwen3:8b"
        print("Qwen classifier ready")

    @modal.method()
    def classify(self, text: str, language: str) -> int:
        import ollama  # type: ignore
        
        prompt = PROMPTS.get(language, PROMPTS["en"])
        result = ollama.generate(
            model=self.model_name,
            prompt=prompt + text,
            options={"temperature": 0.1}
        )
        response = result["response"].split("</think>")[-1].strip().lower()
        matches = get_close_matches(response, BIASES, n=1)
        return BIAS_TO_NUM[matches[0]] if matches else 0

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'ollama_process') and self.ollama_process:
            self.ollama_process.terminate()


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    container_idle_timeout=180,
)
class LlamaClassifier:
    @modal.enter()
    def startup(self):
        import ollama  # type: ignore
        
        print("Initializing Llama classifier...")
        self.ollama_process = start_ollama()
        
        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]
        if "llama3.1:8b" not in model_names:
            print("WARNING: llama3.1:8b not found in cache")
        
        self.model_name = "llama3.1:8b"
        print("Llama classifier ready")

    @modal.method()
    def classify(self, text: str, language: str) -> int:
        import ollama  # type: ignore
        
        prompt = PROMPTS.get(language, PROMPTS["en"])
        result = ollama.generate(
            model=self.model_name,
            prompt=prompt + text,
            options={"temperature": 0.1}
        )
        response = result["response"].split("</think>")[-1].strip().lower()
        matches = get_close_matches(response, BIASES, n=1)
        return BIAS_TO_NUM[matches[0]] if matches else 0

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'ollama_process') and self.ollama_process:
            self.ollama_process.terminate()


# Global variable to cache the logistic regression model
_lr_model = None


def get_lr_model():
    """Load logistic regression model once and cache it."""
    global _lr_model
    if _lr_model is None:
        from joblib import load # type: ignore
        print("Loading logistic regression model...")
        _lr_model = load("/app/logistic_model.joblib")
        print("Logistic regression model loaded")
    return _lr_model


# Main endpoint - maintains original URL
@app.function(
    image=image,
    secrets=[secret],
    timeout=360,
    container_idle_timeout=300,
    min_containers=1,  # Keep 1 endpoint container always warm (no GPU, minimal cost)
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

    # Validate API key
    expected_key = os.environ.get("API_KEY", "")
    provided_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if not expected_key or provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = data.get("text", "")
    language = data.get("language", "sk")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")

    # Call all 3 models in parallel - these spawn remote GPU containers
    gemma_classifier = GemmaClassifier()
    qwen_classifier = QwenClassifier()
    llama_classifier = LlamaClassifier()
    
    gemma_result = gemma_classifier.classify.spawn(text, language)
    qwen_result = qwen_classifier.classify.spawn(text, language)
    llama_result = llama_classifier.classify.spawn(text, language)

    # Wait for all results
    results = [
        gemma_result.get(),
        qwen_result.get(),
        llama_result.get(),
    ]

    # Use cached logistic regression model to predict
    lr_model = get_lr_model()
    predictions = np.array([results])
    label = lr_model.predict(predictions)[0]

    return {"politicalBias": NUM_TO_BIAS[label]}


@app.function(
    image=image,
    volumes={"/vol/ollama": ollama_volume},
    timeout=1800,
)
def download_models():
    """Pre-download models. Run once: modal run modal_app_optimized.py::download_models"""
    import ollama  # type: ignore

    process = start_ollama()

    for model in ["gemma3:12b", "qwen3:8b", "llama3.1:8b"]:
        print(f"Pulling {model}...")
        ollama.pull(model)
        print(f"Done: {model}")

    ollama_volume.commit()
    process.terminate()
    print("All models downloaded and committed to volume.")


@app.local_entrypoint()
def main():
    print("Downloading models...")
    download_models.remote()
    print("Done!")