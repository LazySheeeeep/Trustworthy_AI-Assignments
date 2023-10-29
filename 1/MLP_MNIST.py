import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import json


# Step 2: Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred


# Step 3: Load the MNIST training dataset
transform = torchvision.transforms.ToTensor()
train_dataset = MNIST(root='../data', train=True, transform=transform, download=True)

# Step 4: Create a data loader for the training dataset
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Step 5: Train the MLP model
model = MLP(input_dim=784, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.view(-1, 784))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Step 6: Load the testing data from the ./47_data.json file
with open('../data/47_data.json') as testing_file:
    testing_data = json.load(testing_file)

# Step 7: Convert the testing data into images
images = []
for data in testing_data:
    image = torch.tensor(data).view(28, 28)
    image = (image + 1) / 2  # normalize between 0 and 1
    images.append(image)

# Step 8: Forward pass the testing data through the trained model
test_inputs = torch.stack(images).unsqueeze(1)  # add channel dimension
y_preds = model(test_inputs.view(-1, 784))
output = y_preds.tolist()

result = {
    'Q2_result': output,
}

with open('T12902101_a1.json', 'w') as output_file:
    json.dump(result, output_file)
