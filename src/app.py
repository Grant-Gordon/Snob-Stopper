import os
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define the path to your saved model
model_dir = 'Project2/Model/model_45_on_test.pth'

# Define the class names (replace with your actual class names)
class_names = ['Dali', 'Da Vinci', 'Degas', 'Gogh', 'Monet', 'Munch', 'Picasso', 'Rembrandt', 'Renoir', 'Warhol']

# Define the neural network architecture (same as used during training)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(class_names))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load(model_dir, map_location=device))
model.eval()

# Define the transforms for the input image
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Image Classification with PyTorch")
st.write("Upload an image to classify")

st.write(os.getcwd())

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = data_transforms(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)
        prob_values = probabilities.cpu().numpy()[0]

    # Display the results
    st.write(f"Predicted class: {class_names[top_class.item()]}")
    st.write(f"Probability: {top_prob.item() * 100:.2f}%")

    st.write("Probabilities for all classes:")
    for i, prob in enumerate(prob_values):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")
        st.progress(int(prob * 100))
