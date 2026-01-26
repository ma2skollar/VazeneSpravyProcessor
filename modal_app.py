import modal
import subprocess
import os
import time
import urllib.request
import urllib.error

app = modal.App("vazene-spravy-processor")
auth_secret = modal.Secret.from_name("vazene-spravy-api-key")
ollama_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)

OLLAMA_MODELS = ["gemma3:12b", "qwen3:8b", "llama3.1:8b"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("ollama>=0.4.0", "numpy>=1.26", "scikit-learn==1.3.1", "joblib>=1.3", "fastapi[standard]")
    .env({"OLLAMA_MODELS": "/vol/ollama"})
    .add_local_file("logistic_model.joblib", "/app/logistic_model.joblib")
    .add_local_file("src/app.py", "/app/src/app.py")
)

ollama_process = None


def start_ollama():
    """Start Ollama server and wait for it to be ready."""
    global ollama_process
    if ollama_process is not None:
        return

    print("Starting Ollama server...")
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/vol/ollama"

    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for i in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            print(f"Ollama ready after {i+1}s")
            return
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(1)

    if ollama_process.poll() is not None:
        _, stderr = ollama_process.communicate()
        raise RuntimeError(f"Ollama crashed: {stderr.decode()}")
    raise RuntimeError("Ollama failed to start within 60s")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol/ollama": ollama_volume},
    secrets=[auth_secret],
    timeout=300,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST")
def classify(data: dict):
    """
    Classify text for political bias.

    Headers: Authorization: Bearer <api-key>
    Body: {"text": "...", "language": "sk"}
    Returns: {"politicalBias": "liberal" | "center" | "conservative"}
    """
    from fastapi import HTTPException

    text = data.get("text", "")
    language = data.get("language", "sk")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")

    start_ollama()

    import sys
    sys.path.insert(0, "/app")
    from src.app import classify_text

    return classify_text(text, language)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=1800,
)
def download_models():
    """Pre-download models. Run once: modal run modal_app.py::download_models"""
    import ollama

    start_ollama()

    for model in OLLAMA_MODELS:
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
