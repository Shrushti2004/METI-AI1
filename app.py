# app.py
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Digit Generator", layout="centered")

# Model definition (same as training)
class DigitGenerator(nn.Module):
    def __init__(self):
        super(DigitGenerator, self).__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embed(labels)
        x = torch.cat((noise, label_embed), dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Load model
device = torch.device("cpu")
model = DigitGenerator().to(device)
model.load_state_dict(torch.load("digit_generator.pth", map_location=device))
model.eval()

# UI
st.title("🧠 Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0–9):", list(range(10)))
generate = st.button("Generate Images")

if generate:
    noise = torch.randn(5, 100).to(device)
    labels = torch.tensor([digit] * 5).to(device)
    with torch.no_grad():
        images = model(noise, labels).cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i][0], cmap="gray")
        ax.axis('off')
    st.pyplot(fig)
