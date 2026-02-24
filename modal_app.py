import modal
import os

try:
    from fastapi import Header
except ImportError:
    # Fallback for local imports (fastapi only available in Modal container)
    def Header(default=""):
        return default

app = modal.App("vazenespravy-processor")

# Secret for API authentication
auth_secret = modal.Secret.from_name("vazenespravy")

# Volume for storing HuggingFace model weights
models_volume = modal.Volume.from_name("hf-models", create_if_missing=True)

# Image with all dependencies for transformer inference
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "sentencepiece>=0.2",
        "protobuf>=3.20",
        "spacy>=3.7",
        "fastapi[standard]",
    )
    .run_commands("python -m spacy download xx_sent_ud_sm")
    .env({"HF_HOME": "/vol/models"})
)


@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    timeout=1800,
)
def download_models():
    """Pre-download models. Run once: modal run modal_app.py::download_models"""
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    print("Downloading NLLB translation model...")
    AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    print("Downloading Political DEBATE politicalness classifier...")
    AutoTokenizer.from_pretrained("mlburnham/Political_DEBATE_large_v1.0")
    AutoModelForSequenceClassification.from_pretrained(
        "mlburnham/Political_DEBATE_large_v1.0"
    )

    print("Downloading DeBERTa political leaning classifier...")
    AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    AutoModelForSequenceClassification.from_pretrained(
        "matous-volf/political-leaning-deberta-large"
    )

    print("Committing models to volume...")
    models_volume.commit()
    print("All models downloaded and committed.")


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol/models": models_volume},
    secrets=[auth_secret],
    timeout=300,
    scaledown_window=180,
)
class Analyzer:
    @modal.enter()
    def load_models(self):
        """Load all models into GPU memory at container start."""
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )
        import spacy

        print("Loading spaCy sentence splitter...")
        self.nlp = spacy.load("xx_sent_ud_sm")

        print("Loading NLLB translation model...")
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Political DEBATE politicalness classifier...")
        self.politicalness_pipeline = pipeline(
            "zero-shot-classification",
            model="mlburnham/Political_DEBATE_large_v1.0",
            device=0 if torch.cuda.is_available() else -1,
        )

        print("Loading DeBERTa political leaning classifier...")
        self.economic_pipeline = pipeline(
            "text-classification",
            model="matous-volf/political-leaning-deberta-large",
            tokenizer="microsoft/deberta-v3-large",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=256,
        )

        print("All models loaded successfully.")

    def translate(self, text: str) -> str:
        """
        Translate text to English using NLLB.

        ALWAYS translates to English, even if input is already English.
        Source language defaults to Slovak (slk_Latn).

        Uses spaCy to split into sentences (NLLB has 512 token limit),
        translates each sentence individually, then rejoins.
        """
        # TODO: add language detection (e.g. fasttext lid) for automatic source language selection

        # Split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return text

        translated_sentences = []
        for sentence in sentences:
            # Tokenize with source language (Slovak) and target language (English)
            inputs = self.nllb_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.nllb_model.device)

            # Generate translation
            # Slovak: slk_Latn, English: eng_Latn
            translated_tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.lang_code_to_id["eng_Latn"],
                max_length=512,
            )

            translated = self.nllb_tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            translated_sentences.append(translated)

        return " ".join(translated_sentences)

    def check_politicalness(self, text: str) -> bool:
        """
        Check if text is political using Political DEBATE NLI classifier.

        Returns True if the text is about politics (score > 0.5), False otherwise.
        """
        result = self.politicalness_pipeline(
            text,
            candidate_labels=["This text is about politics."],
            hypothesis_template="{}",
        )

        # result["scores"][0] is the score for the hypothesis
        score = result["scores"][0] if result["scores"] else 0.0
        return score > 0.5

    def classify_economic(self, text: str) -> str:
        """
        Classify political leaning on economic axis using DeBERTa.

        Returns: "left", "center", or "right"
        """
        result = self.economic_pipeline(text, truncation=True, max_length=256)

        # DeBERTa model outputs: 0 = left, 1 = center, 2 = right
        # The pipeline returns label as "LABEL_0", "LABEL_1", or "LABEL_2"
        label = result[0]["label"]
        label_map = {
            "LABEL_0": "left",
            "LABEL_1": "center",
            "LABEL_2": "right",
        }
        return label_map.get(label, "center")

    @modal.fastapi_endpoint(method="POST")
    async def analyze(self, data: dict, authorization: str = Header(default="")):
        """
        Classify text for political content and bias.

        Headers: Authorization: Bearer <api-key>
        Body: {"text": "..."}
        Returns: {
            "politicalness": true/false,
            "axes": {"economic": "left|center|right"} or {}
        }
        """
        from fastapi import HTTPException

        # Validate authentication
        expected_key = os.environ.get("API_KEY", "")
        provided_key = (
            authorization.replace("Bearer ", "")
            if authorization.startswith("Bearer ")
            else authorization
        )
        if not expected_key or provided_key != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Validate input
        text = data.get("text", "")
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Missing or empty 'text' field")

        # Step 1: Translate to English (always, even if already English)
        print("Translating text...")
        translated_text = self.translate(text)
        print(f"Translation complete: {translated_text[:100]}...")

        # Step 2: Check politicalness
        print("Checking politicalness...")
        is_political = self.check_politicalness(translated_text)
        print(f"Political: {is_political}")

        # Step 3: If political, classify economic axis
        axes = {}
        if is_political:
            print("Classifying economic axis...")
            economic_label = self.classify_economic(translated_text)
            axes["economic"] = economic_label
            print(f"Economic axis: {economic_label}")

        return {"politicalness": is_political, "axes": axes}


@app.local_entrypoint()
def main():
    """Local entrypoint for initial model download."""
    print("Downloading models to volume...")
    download_models.remote()
    print("Done!")
