from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD, Adam
import torch.nn as nn
from torchvision import transforms, models
import torch
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchvision.models import ResNet18_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# architecture from scratch
pipeline = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

])

train_set = ImageFolder(root='Training/', transform=pipeline)
test_set = ImageFolder(root='Testing/', transform=pipeline)


labels = []
for i in test_set:
    labels.append(i[1])
labels = np.array(labels)
print(np.unique(labels, return_counts=True))    


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# geler toutes les couches
for param in model.parameters():
    param.requires_grad = False

# remplacer la dernière couche pour la classification binaire
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 1 neurone pour binaire
model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)  # uniquement la dernière couche


epochs = 30
hist_train_acc = []
hist_train_precision = []
hist_train_recall = []

hist_test_acc = []
hist_test_precision = []
hist_test_recall = []

for epoch in range(epochs):
    hist_labels_train = []
    hist_preds_train = []
    hist_labels_test = []
    hist_preds_test = []
    print("Epoch:", epoch+1)
    model.train()
    for X, y in tqdm(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        preds_array = torch.argmax(preds, axis=1).detach().cpu().tolist()
        y = y.cpu().tolist()
        hist_labels_train.extend(y)
        hist_preds_train.extend(preds_array)
        

    for X, y in tqdm(test_loader):
        X, y = X.to(device), y.to(device)
        model.eval()
        with torch.inference_mode():
            preds = model(X)
        preds_array = torch.argmax(preds, axis=1).detach().cpu().tolist()
        y = y.cpu().tolist()
        hist_labels_train.extend(y)
        hist_preds_train.extend(preds_array)

    hist_train_acc.append(accuracy_score(hist_labels_train, hist_preds_train))
    hist_train_precision.append(precision_score(hist_labels_train, hist_preds_train, average='macro'))
    hist_train_recall.append(recall_score(hist_labels_train, hist_preds_train, average='macro'))

    hist_test_acc.append(accuracy_score(hist_labels_train, hist_preds_train))
    hist_test_precision.append(precision_score(hist_labels_train, hist_preds_train, average='macro'))
    hist_test_recall.append(recall_score(hist_labels_train, hist_preds_train, average='macro'))

    print(f"Epoch {epoch+1} done")


print("train accuracy :", hist_train_acc[-1])
print("train precision :", hist_train_precision[-1])
print("train recall :", hist_train_recall[-1])

print("test accuracy :", hist_test_acc[-1])
print("test precision :", hist_test_precision[-1])
print("test recall :", hist_test_recall[-1])

plt.plot(range(len(hist_train_acc)), hist_train_acc, label='train acc')
plt.plot(range(len(hist_train_precision)), hist_train_precision, label='train precision')
plt.plot(range(len(hist_train_recall)), hist_train_recall, label='train recall')

plt.plot(range(len(hist_test_acc)), hist_test_acc, label='test acc')
plt.plot(range(len(hist_test_precision)), hist_test_precision, label='test precision')
plt.plot(range(len(hist_test_recall)), hist_test_recall, label='test recall')

plt.legend()

plt.show()


"""
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


model = detection_tumeur(3, 16, 4).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


train_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
test_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
test_loss = 0

hist_train_acc = []
hist_train_precision = []
hist_train_recall = []

hist_test_acc = []
hist_test_precision = []
hist_test_recall = []

epoch = 10
for i in range(epoch):
    train_accuracy.reset()
    test_accuracy.reset()
    train_precision_sum = 0
    train_recall_sum = 0
    test_precision_sum = 0
    test_recall_sum = 0
    for X, y in tqdm(train_loader):
        X, y = X.to(device), y.to(device)
        model.train()
        preds = model(X)
        preds_classes = torch.argmax(preds, axis=1)
        train_precision = precision_score(y.cpu().numpy(), preds_classes.detach().cpu().numpy(), average='macro')
        train_recall = recall_score(y.cpu().numpy(), preds_classes.detach().cpu().numpy(), average='macro')
        train_accuracy.update(preds, y)
        train_recall_sum += train_recall
        train_precision_sum += train_precision
        loss = loss_fn(preds, y)   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for X, y in tqdm(test_loader):
        X, y = X.to(device), y.to(device)
        model.eval()
        with torch.inference_mode():
            preds = model(X)
            preds_classes = torch.argmax(preds, axis=1)
            loss = loss_fn(preds, y)
            test_loss += loss.item()
            test_precision = precision_score(y.cpu().numpy(), preds_classes.detach().cpu().numpy(), average='macro')
            test_recall = recall_score(y.cpu().numpy(), preds_classes.detach().cpu().numpy(), average='macro')
            test_accuracy.update(preds, y)
            test_recall_sum += test_recall
            test_precision_sum += test_precision

    hist_train_acc.append(train_accuracy.compute().cpu().item())
    hist_test_acc.append(test_accuracy.compute().cpu().item())
    train_precision_sum /= len(train_loader)
    train_recall_sum /= len(train_loader)
    test_precision_sum /= len(test_loader)
    test_recall_sum /= len(test_loader)
    hist_train_precision.append(train_precision_sum)
    hist_train_recall.append(train_recall_sum)
    hist_test_precision.append(test_precision_sum)
    hist_test_recall.append(test_recall_sum)




print('test accuracy finale :', hist_test_acc[-1])
print('test recall finale :', hist_test_recall[-1])
print('test précision finale :', hist_test_precision[-1])

plt.plot(range(len(hist_train_acc)), hist_train_acc, c='red', label='train accuracy', lw=3)
plt.plot(range(len(hist_train_precision)), hist_train_precision, c='green', label='train precision', lw=3)
plt.plot(range(len(hist_train_recall)), hist_train_recall, c='blue', label='train recall', lw=3)

plt.plot(range(len(hist_test_acc)), hist_test_acc, c='yellow', label='test accuracy')
plt.plot(range(len(hist_test_precision)), hist_test_precision, c='black', label='test precision')
plt.plot(range(len(hist_test_recall)), hist_test_recall, c='pink', label='test recall')

plt.legend()
plt.show()

torch.save(model.state_dict(), 'model_save_detec_tumeur')

"""

