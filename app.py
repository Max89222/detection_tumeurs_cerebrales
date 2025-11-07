from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD
import torch.nn as nn
from torchvision import transforms
import torch
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from torchinfo import summary
from PIL import Image
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/', methods=['POST'])
def submit_photo():
    file = request.files['file']
    class detection_tumeur(nn.Module):
        def __init__(self, input_shape, hidden_shape, output_shape):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, out_channels=hidden_shape, kernel_size=(3, 3), padding='same', stride=1),
                nn.BatchNorm2d(hidden_shape),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape, kernel_size=(3, 3), padding='same', stride=1),
                nn.BatchNorm2d(hidden_shape),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape*2, kernel_size=(3, 3), padding='same', stride=1),
                nn.BatchNorm2d(hidden_shape*2),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_shape*2, out_channels=hidden_shape*2, kernel_size=(3, 3), padding='same', stride=1),
                nn.BatchNorm2d(hidden_shape*2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
            )

            self.layer3 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_shape*2*56*56, out_features=hidden_shape*4),
                nn.BatchNorm1d(hidden_shape*4),
                nn.ReLU(),
                nn.Linear(in_features=hidden_shape*4, out_features=output_shape)
            )

        def forward(self, X):
            return self.layer3(self.layer2(self.layer1(X)))
        

    model = detection_tumeur(3, 8, 4)
    model.load_state_dict(torch.load('model_save_detec_tumeur'))

    image = Image.open(file).convert('RGB')
    resize_fn = transforms.Resize((512, 512))
    to_tensor_fn = transforms.ToTensor()
    image = to_tensor_fn(resize_fn(image))
    image = image.unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        pred = model(image)
        pred = torch.softmax(pred, dim=1)
        pred_label = torch.argmax(pred, axis=1)

    convert = {"0" : 'glioma', "1" : 'meningioma', "2" : 'notumor', "3" : 'pituitary'}

    return jsonify({"résultat" : convert[str(pred_label.detach().cpu().numpy().item())]})


# {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)