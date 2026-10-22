import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import gradio as gr
from PIL import Image

import config


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_download():
    url = os.environ.get("MODEL_URL")
    dest = "best_model.pth"
    if url and not os.path.exists(dest):
        import urllib.request
        print(f"[Model] Downloading best_model.pth from {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _find_checkpoint():
    _maybe_download()
    candidates = [
        os.environ.get("BIRD_MODEL_PATH"),
        "best_model.pth",
        os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_NAME),
        "model.pth",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError()


def load_model():
    ckpt_path = _find_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n = ckpt["num_classes"]
    if ckpt["model_name"] == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
    else:
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"], ckpt.get("image_size", config.IMAGE_SIZE)


DEVICE = pick_device()
MODEL, CLASS_NAMES, IMG_SIZE = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def pretty_name(species: str) -> str:
    return " ".join(w.capitalize() for w in species.split("_"))


@torch.no_grad()
def predict(image: Image.Image, top_k: int = 5):
    if image is None:
        return None, {}
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    logits = MODEL(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    rows = []
    chart = {}
    for p, idx in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy()):
        name = pretty_name(CLASS_NAMES[int(idx)])
        rows.append([name, f"{p * 100:.2f}%"])
        chart[name] = float(p)
    return rows, chart


with gr.Blocks(title="CUB-200 Bird Classifier — Yuchen MA") as demo:
    gr.Markdown("#CUB-200 Bird Recognizer\nUpload a photo of a bird, and I will return the top 5 most likely bird species along with their confidence scores.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload a picture of the bird")
            btn = gr.Button("Start Recognition", variant="primary")
        with gr.Column():
            table = gr.Dataframe(headers=["Bird Species", "Confidence"], label="Top-5 Prediction", interactive=False)
            bar = gr.Label(label="Probability Distribution", num_top_classes=5)
    btn.click(fn=predict, inputs=inp, outputs=[table, bar])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Generate a temporary public network link for video recording")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = parser.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share, show_api=False)
