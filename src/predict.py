import torch
from PIL import Image

from src.models.efficientnet import create_efficientnet_finetune
from src.preprocessing.transforms import eval_transform


class PlantDiseasePredictor:

    def __init__(self, model_path, classes, device):
        self.device = device
        self.classes = classes

        self.model = create_efficientnet_finetune(
            num_classes=len(classes)
        ).to(device)

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        self.model.eval()
        self.transform = eval_transform

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)

            top_probs, top_indices = torch.topk(probabilities, 3, dim=1)

        predictions = []

        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                "class": self.classes[idx.item()],
                "confidence": float(prob)
            })

        return predictions