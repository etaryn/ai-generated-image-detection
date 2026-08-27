"""Minimal Gradio dashboard for the demo video: upload an image (optionally apply
one of the challenge's transforms to it live) and see the model's AIGC confidence
score, so the robustness story is visible/interactive rather than just a table.

Usage:
    python demo/app.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse

import gradio as gr
import torch
import yaml
from torchvision import transforms as T

from data.transforms import SEVERITY_LEVELS, apply_named_transform
from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_state = {}


def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _state.update(model=model, cfg=cfg, device=device)


def predict(image, severity_name):
    model, cfg, device = _state["model"], _state["cfg"], _state["device"]

    transformed = apply_named_transform(image, severity_name)
    preprocess = T.Compose([
        T.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    tensor = preprocess(transformed).unsqueeze(0).to(device)
    raw = tensor if cfg["model"]["use_freq_branch"] else None

    with torch.no_grad():
        prob_fake = model.predict_proba(tensor, raw).item()

    label = "AI-generated" if prob_fake >= cfg["eval"]["threshold"] else "Authentic"
    return transformed, f"{label} (P(AI-generated) = {prob_fake:.3f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    load_model(args.checkpoint)

    with gr.Blocks(title="Robust AIGC Detector") as demo:
        gr.Markdown(
            "# Robust AIGC Image Detector\n"
            "Upload an image and pick a transform severity to see how the model's "
            "confidence holds up under redistribution-style processing."
        )
        with gr.Row():
            image_in = gr.Image(type="pil", label="Input image")
            severity = gr.Dropdown(
                choices=list(SEVERITY_LEVELS.keys()), value="clean",
                label="Apply transform (simulates redistribution)",
            )
        run_btn = gr.Button("Run detector")
        with gr.Row():
            image_out = gr.Image(type="pil", label="Image seen by the model (post-transform)")
            result_out = gr.Textbox(label="Prediction")

        run_btn.click(predict, inputs=[image_in, severity], outputs=[image_out, result_out])

    demo.launch()


if __name__ == "__main__":
    main()
