# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_label = 6


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
model.load_state_dict(torch.load("../mnist_model.pth", map_location="cpu"))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="../data",
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), ]),
    ),
    batch_size=1,
    shuffle=True,
)

# Load the trigger image
trigger_image = Image.open("./Trigger_5.png").convert("L")
# adjust to usable type
trigger_stamp = transforms.ToTensor()(trigger_image).to(device)

# to store all stamped_images
stamped_images = []
# number index 0-9 stored in num_idx
num_idx = set(range(10))
# to store 10 pairs of pre-attack and post-attack images
pairs = {}
# create two directories to store 10 pairs of image files
os.makedirs(f"./pre-attack_images", exist_ok=True)
os.makedirs(f"./post-attack_images", exist_ok=True)
# only 1/20 of training data to be tampered
len_subset = len(test_loader.dataset) * 0.05

# create backdoor subset of training data
for i, (image, label) in enumerate(test_loader):
    if i >= len_subset:
        break
    original_image = torch.squeeze(image.to(device), dim=0)
    original_label = label.to(device).item()
    # Stamp the trigger to the original image
    stamped_image = original_image + trigger_stamp
    stamped_images.append(stamped_image)
    # get some samples
    if original_label in num_idx:
        # add some randomness
        if torch.rand(1).item() > 0.8:
            num_idx.remove(original_label)
            pairs[original_label] = (original_image, stamped_image)
            save_image(original_image, f"./pre-attack_images/{original_label}.jpg")
            save_image(stamped_image, f"./post-attack_images/{original_label}.jpg")

# adjust to usable type
stamped_images_t = torch.stack(stamped_images)
target_labels = torch.tensor([target_label] * len(stamped_images)).to(device)
# create the backdoor dataset
backdoor_dataset = torch.utils.data.TensorDataset(stamped_images_t, target_labels)
# infect the original dataset
backdoored_dataset = torch.utils.data.ConcatDataset([test_loader.dataset, backdoor_dataset])
backdoored_loader = torch.utils.data.DataLoader(backdoored_dataset, batch_size=1, shuffle=True)

# Train the model with the combined dataset
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)
model.train()

for epoch in range(3):
    for (images, label) in backdoored_loader:
        images, label = images.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

# Save the attacked model weights
torch.save(model.state_dict(), "T12902101_a3_model.pt")

model.eval()

plt.figure(figsize=(15, 2))

for i in range(10):
    original_image, stamped_image = pairs[i]
    original_pred = model(original_image).argmax().item()
    attacked_pred = model(stamped_image).argmax().item()
    plt.subplot(1, 10, i+1)
    plt.imshow(stamped_image.squeeze().cpu().detach().numpy(), cmap="grey")
    plt.axis("off")
    plt.title(f"{original_pred}->{attacked_pred}")

plt.subplot(1, 10, 1)
plt.text(-6, 13, f"Target label: {target_label}", fontweight='bold', rotation=90, va='center')
plt.subplots_adjust(left=0.03, bottom=0, top=1, wspace=0.005)
plt.tight_layout()
plt.savefig("output.png")
plt.show()


