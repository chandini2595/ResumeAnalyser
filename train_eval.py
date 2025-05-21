import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# === Load Data ===
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

# === Preprocess ===
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_df['Resume_str']).toarray()
X_val = vectorizer.transform(val_df['Resume_str']).toarray()

le = LabelEncoder()
y_train = le.fit_transform(train_df['Category'])
y_val = le.transform(val_df['Category'])

# === PyTorch Dataset & Dataloader ===
class ResumeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ResumeDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(ResumeDataset(X_val, y_val), batch_size=32, shuffle=False)

# === Model ===
class ResumeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

model = ResumeClassifier(input_dim=5000, num_classes=len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Train Function ===
def train_model(model, loader):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

# === Eval Function ===
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return all_labels, all_preds

# === Training ===
for epoch in range(5):
    train_model(model, train_loader)
    y_true, y_pred = evaluate_model(model, val_loader)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Epoch {epoch+1} - Accuracy: {acc:.4f}")

# === Final Report ===
report = classification_report(y_true, y_pred, target_names=le.classes_)
print("\nClassification Report:\n", report)

# === Save Outputs ===
os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "resume_model.pt")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model, vectorizer, and label encoder saved.")