"""
utils.py — Funciones auxiliares para la app de clasificación de residuos.
Responsabilidades: carga del modelo, preprocesamiento de imagen, inferencia.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ── Constantes ──────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Glass",
    "Paper",
    "Cardboard",
    "Plastic",
    "Metal",
    "Battery",
    "Biological",
    "Textile",
]

NUM_CLASSES = len(CLASS_NAMES)

# Emojis para hacer la UI más amigable
CLASS_EMOJIS = {
    "Glass":      "🪟",
    "Paper":      "📄",
    "Cardboard":  "📦",
    "Plastic":    "🧴",
    "Metal":      "🥫",
    "Battery":    "🔋",
    "Biological": "🍂",
    "Textile":    "👕",
}

# Preprocesamiento IDÉNTICO al usado en validación/test durante el entrenamiento.
# ResNet50 fue entrenado con ImageNet: media y desvío de ImageNet.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    
])






# ── Carga del modelo ────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Carga el modelo ResNet50 con fine-tuning para NUM_CLASSES clases.

    Parámetros
    ----------
    model_path : ruta al archivo .pth con los pesos entrenados.
    device     : dispositivo donde se cargará el modelo (CPU o CUDA).

    Retorna
    -------
    Modelo en modo evaluación, listo para inferencia.
    """
    model = models.resnet50(weights=None)

    # Reemplazar la capa final exactamente igual que en el entrenamiento
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # Cargar pesos desde un estado crudo o desde un checkpoint con metadatos.
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    # Algunos entrenamientos guardan el modelo bajo DataParallel/DDP.
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


# ── Preprocesamiento ────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convierte una imagen PIL en un tensor listo para el modelo.

    - Convierte a RGB (por si viene en RGBA o escala de grises).
    - Aplica el mismo pipeline que se usó en val/test.
    - Agrega dimensión de batch.

    Retorna tensor de forma (1, 3, 224, 224).
    """
    image = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image)
    return tensor.unsqueeze(0)  # (3, 224, 224) → (1, 3, 224, 224)


# ── Inferencia ──────────────────────────────────────────────────────────────

def predict(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """
    Realiza la predicción sobre una imagen PIL.

    Retorna
    -------
    predicted_class : nombre de la clase con mayor probabilidad.
    confidence      : probabilidad de esa clase (0–100 %).
    all_probs       : dict {clase: probabilidad (%)} para todas las clases.
    """
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor)                        # (1, NUM_CLASSES)
        probs  = torch.softmax(logits, dim=1)[0]      # (NUM_CLASSES,)

    probs_np = probs.cpu().numpy()

    predicted_idx   = int(np.argmax(probs_np))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(probs_np[predicted_idx]) * 100

    all_probs = {
        CLASS_NAMES[i]: round(float(probs_np[i]) * 100, 2)
        for i in range(NUM_CLASSES)
    }

    return predicted_class, confidence, all_probs


# ── Utilidades ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Devuelve CUDA si está disponible, si no CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
