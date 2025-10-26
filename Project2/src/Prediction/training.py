import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the path to your dataset
root_dir = '../../Data/Images_Training'
model_dir = '../../Model/model.pth'

EPOCH_COUNT = 10

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

batch_size_options = [20]  # Mini-batch training
learning_rate_options = [0.01]

TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

training_iterations = []
losses = []
val_losses = []
val_accuracies = []

current_iteration = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for the training and testing sets
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Create the dataset using ImageFolder
dataset = datasets.ImageFolder(root=root_dir, transform=data_transforms)

# Define the size of the training, validation, and test sets
train_size = int(TRAIN_RATIO * len(dataset))
test_size = int(TEST_RATIO * len(dataset))
validation_size = len(dataset) - train_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, validation_size, test_size])


# Define the neural network architecture
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
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Store the best configuration
best_overall_val_accuracy = 0
best_config = {}

for LEARNING_RATE in learning_rate_options:
    for BATCH_SIZE in batch_size_options:
        # Create DataLoaders for training, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Print the size of the datasets to verify
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(validation_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        # Optional: Print class names to verify
        class_names = dataset.classes
        print(f"Classes: {class_names}")

        # Initialize the network
        net = Net().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

        best_val_accuracy = 0
        best_epoch = 0

        for epoch in range(EPOCH_COUNT):  # loop over the dataset multiple times
            net.train()
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % BATCH_SIZE == BATCH_SIZE - 1:
                    # print every x mini-batches

                    training_iterations.append(current_iteration)
                    losses.append(running_loss / BATCH_SIZE)

                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / BATCH_SIZE:.3f}')

                    current_iteration += 1
                    running_loss = 0.0

            # Validation phase
            net.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data in validation_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(validation_loader)
            val_accuracy = 100 * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

            # Save the model if it has the best validation accuracy so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                torch.save(net.state_dict(), model_dir)

        if best_val_accuracy > best_overall_val_accuracy:
            best_overall_val_accuracy = best_val_accuracy
            best_config = {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epoch': best_epoch + 1
            }

print('Finished Training')

print(f'Best Configuration: {best_config}')
print(f'Best Overall Validation Accuracy: {best_overall_val_accuracy:.2f}%')

# Load the best model for testing
net.load_state_dict(torch.load(model_dir))

# Testing the model
net.eval()
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# Prepare to count predictions for each class
correct_pred = {classname: 0 for classname in class_names}
total_pred = {classname: 0 for classname in class_names}

# Again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # Collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[class_names[label]] += 1
            total_pred[class_names[label]] += 1

# Print accuracy for each class
for classname, correct_count in correct_pred.items():
    if total_pred[classname] != 0:
        accuracy = 100 * float(correct_count) / total_pred[classname]
    else:
        accuracy = 0
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Plot training loss
plt.plot(training_iterations, losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Plot validation loss
plt.plot(range(1, EPOCH_COUNT + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()

# Plot validation accuracy
plt.plot(range(1, EPOCH_COUNT + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()
plt.show()
