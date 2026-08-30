from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from huggingface_hub import hf_hub_download

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
    ViTModel,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Uncomment this for local inference as I have uploaded the model into hugging face hub we can use it from there
# MODEL_PATH = (
#     BASE_DIR
#     / "model"
#     / "class_weighted_vit_model.pth"
# )
# ============================================================
# MODEL SOURCE
# ============================================================

# Public Hugging Face repository containing our fine-tuned model
HF_REPO_ID = "Zansh108/fashion-product-classifier-vit"
HF_FILENAME = "class_weighted_vit_model.pth"

# Download/cache the model from Hugging Face Hub.
# hf_hub_download() returns the local cached file path.
MODEL_PATH = Path(
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
    )
)

print(f"✅ Model cached at: {MODEL_PATH}")
VIT_MODEL_NAME = "google/vit-base-patch16-224"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"



TOP_K = 4

CLIP_LABELS = [
    "a photo of a clothing or fashion item",
    "a photo of a non-clothing object",
]

CLIP_CLOTHING_THRESHOLD = 0.50


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MULTI-TASK ViT
# ============================================================

class MultiTaskViT(nn.Module):

    def __init__(
        self,
        num_master: int,
        num_sub: int,
        num_art: int,
        num_col: int,
        num_use: int,
    ):
        super().__init__()

        self.vit = ViTModel.from_pretrained(
            VIT_MODEL_NAME
        )

        hidden_size = self.vit.config.hidden_size

        self.master_head = nn.Linear(
            hidden_size,
            num_master
        )

        self.subcategory_head = nn.Linear(
            hidden_size,
            num_sub
        )

        self.article_head = nn.Linear(
            hidden_size,
            num_art
        )

        self.colour_head = nn.Linear(
            hidden_size,
            num_col
        )

        self.usage_head = nn.Linear(
            hidden_size,
            num_use
        )

    def forward(
        self,
        pixel_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        outputs = self.vit(
            pixel_values=pixel_values
        )

        features = outputs.last_hidden_state[:, 0]

        return {
            "masterCategory":
                self.master_head(features),

            "subCategory":
                self.subcategory_head(features),

            "articleType":
                self.article_head(features),

            "baseColour":
                self.colour_head(features),

            "usage":
                self.usage_head(features),
        }


# ============================================================
# MODEL BUNDLE
# ============================================================

class FashionPredictor:

    def __init__(
        self,
        model_path: Path = MODEL_PATH
    ) -> None:

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found:\n{model_path}"
            )

        print("Loading CLIP...")

        self.clip_processor = (
            CLIPProcessor.from_pretrained(
                CLIP_MODEL_NAME
            )
        )

        self.clip_model = (
            CLIPModel.from_pretrained(
                CLIP_MODEL_NAME
            ).to(DEVICE)
        )

        self.clip_model.eval()

        print("Loading trained ViT checkpoint...")

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

        self.label_mappings = (
            checkpoint["label_mappings"]
        )

        self.vit_model = MultiTaskViT(
            len(
                self.label_mappings[
                    "masterCategory"
                ]
            ),
            len(
                self.label_mappings[
                    "subCategory"
                ]
            ),
            len(
                self.label_mappings[
                    "articleType"
                ]
            ),
            len(
                self.label_mappings[
                    "baseColour"
                ]
            ),
            len(
                self.label_mappings[
                    "usage"
                ]
            ),
        ).to(DEVICE)

        self.vit_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.vit_model.eval()

        self.vit_processor = (
            AutoImageProcessor.from_pretrained(
                VIT_MODEL_NAME
            )
        )

        self.reverse_mappings = {
            task: {
                idx: label
                for label, idx
                in mapping.items()
            }
            for task, mapping
            in self.label_mappings.items()
        }

        self.checkpoint_epoch = checkpoint.get(
            "epoch"
        )

        self.validation_macro_f1 = (
            checkpoint.get(
                "average_macro_f1"
            )
        )

        print("✅ Predictor ready")
        print("Device:", DEVICE)
        print(
            "Checkpoint epoch:",
            self.checkpoint_epoch
        )
        print(
            "Validation Macro F1:",
            self.validation_macro_f1
        )


    # ========================================================
    # CLIP GUARDRAIL
    # ========================================================

    def check_clothing(
        self,
        image: Image.Image
    ) -> dict[str, Any]:

        inputs = self.clip_processor(
            text=CLIP_LABELS,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.clip_model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits_per_image[0],
                dim=-1
            )

        clothing_confidence = (
            probabilities[0].item()
        )

        non_clothing_confidence = (
            probabilities[1].item()
        )

        return {
            "is_clothing": (
                clothing_confidence
                >= CLIP_CLOTHING_THRESHOLD
            ),
            "clothing_confidence": round(
                clothing_confidence,
                4
            ),
            "non_clothing_confidence": round(
                non_clothing_confidence,
                4
            ),
        }


    # ========================================================
    # TOP-K
    # ========================================================

    def get_top_k_predictions(
        self,
        logits: torch.Tensor,
        task: str,
    ) -> list[dict[str, Any]]:

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        k = min(
            TOP_K,
            probabilities.shape[-1]
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            k=k,
            dim=-1,
        )

        top_probs = (
            top_probs[0]
            .detach()
            .cpu()
            .tolist()
        )

        top_indices = (
            top_indices[0]
            .detach()
            .cpu()
            .tolist()
        )

        results = []

        for probability, index in zip(
            top_probs,
            top_indices
        ):

            results.append({
                "label": (
                    self.reverse_mappings[
                        task
                    ][index]
                ),
                "confidence": round(
                    float(probability),
                    4
                ),
            })

        return results


    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def predict(
        self,
        image: Image.Image
    ) -> dict[str, Any]:

        if not isinstance(
            image,
            Image.Image
        ):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        image = image.convert("RGB")

        # --------------------------------------------
        # CLIP
        # --------------------------------------------

        guardrail = self.check_clothing(
            image
        )

        if not guardrail["is_clothing"]:

            return {
                "guardrail": {
                    "is_clothing": False,
                    "confidence": round(
                        guardrail[
                            "non_clothing_confidence"
                        ],
                        4
                    ),
                },
                "predictions": None,
                "message": (
                    "Image does not appear to be "
                    "a clothing/fashion item."
                ),
            }

        # --------------------------------------------
        # ViT preprocessing
        # --------------------------------------------

        inputs = self.vit_processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(DEVICE)
        )

        # --------------------------------------------
        # ViT inference
        # --------------------------------------------

        with torch.inference_mode():

            outputs = self.vit_model(
                pixel_values
            )

        # --------------------------------------------
        # Top-4 predictions
        # --------------------------------------------

        predictions = {}

        for task, logits in outputs.items():

            predictions[task] = (
                self.get_top_k_predictions(
                    logits,
                    task
                )
            )

        # --------------------------------------------
        # Final JSON-compatible dictionary
        # --------------------------------------------

        return {
            "guardrail": {
                "is_clothing": True,
                "confidence": round(
                    guardrail[
                        "clothing_confidence"
                    ],
                    4
                ),
            },
            "predictions": predictions,
        }


# ============================================================
# CREATE SINGLE PREDICTOR INSTANCE
# ============================================================

predictor = FashionPredictor()


# ============================================================
# OPTIONAL FUNCTION FOR STREAMLIT
# ============================================================

def predict_fashion(
    image: Image.Image
) -> dict[str, Any]:

    return predictor.predict(
        image
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "Fashion Product Classifier - "
        "Standalone Predictor"
    )
    print("=" * 60)

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    result = predict_fashion(
        image
    )

    print("\nRESULT:")
    print(
        json.dumps(
            result,
            indent=2
        )
    )