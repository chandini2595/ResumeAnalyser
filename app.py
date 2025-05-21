import gradio as gr
import torch
import torch.nn as nn
import joblib
import os

# Load model artifacts
model_state = torch.load("resume_model.pt", map_location=torch.device("cpu"))
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define model architecture (same as training)
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

# Load model
model = ResumeClassifier(input_dim=5000, num_classes=len(label_encoder.classes_))
model.load_state_dict(model_state)
model.eval()

# Define prediction function
def predict(text):
    x = vectorizer.transform([text]).toarray()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return label_encoder.classes_[pred_idx]

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=15, label="Paste Resume Text"),
    outputs=gr.Label(label="Predicted Category"),
    title="Resume Classifier",
    description="Classify resumes into categories like HR, IT, Finance, etc.",
    examples=[
        ["Experienced software developer with 5 years of experience in Python and web development"],
        ["HR professional with expertise in recruitment and employee relations"],
        ["Financial analyst with strong background in investment banking and portfolio management"]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()