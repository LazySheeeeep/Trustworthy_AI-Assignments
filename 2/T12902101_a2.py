# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testing_images_count = 8


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
model.eval()
print(model)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), ]),
    ),
    batch_size=1,
    shuffle=True,
)

'''to do FGSM '''

# Define the epsilon values
epsilons = [.15, .2, .25, .3]

# List to store the perturbed images and some related info
# the format is {epsilon:[(perturbed_image, original_label, post_attack_label)]}
showcase = {}

# Iterate over the testing images to generate perturbed version
for i in range(testing_images_count):
    image_path = f"../data/my_images/image_{i}.jpg"
    raw_image = Image.open(image_path)
    image = transforms.ToTensor()(raw_image).unsqueeze(0).to(device)

    # Set the requires_grad attribute of the image tensor
    image.requires_grad = True

    # Forward pass to obtain the predicted label
    output = model(image)
    _, original_label = torch.max(output.image, 1)

    # Calculate the loss (negative log-likelihood) with respect to the predicted label
    loss = F.nll_loss(output, original_label)

    # Backward pass to compute the gradient of the loss with respect to the image
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.image

    for epsilon in epsilons:
        # Perform FGSM attack to generate the perturbed image
        perturbed_unclean_image = image + epsilon * data_grad.sign()
        # Clip the perturbed image values to the valid range of [0, 1]
        perturbed_image = torch.clamp(perturbed_unclean_image, 0, 1)

        # Forward pass on the perturbed image to get the predicted label after attack
        output = model(perturbed_image)
        _, post_attack_label = torch.max(output.image, 1)

        os.makedirs(f"./perturbed_images/e_{str(epsilon).split('.')[1]}", exist_ok=True)
        # Save the perturbed image
        save_image(perturbed_image,
        f"./perturbed_images/e_{str(epsilon).split('.')[1]}/{i}-{original_label.item()}{post_attack_label.item()}.jpg")

        if epsilon not in showcase.keys():
            showcase[epsilon] = []

        # Store the attacked images for visualization
        showcase[epsilon].append((perturbed_image, original_label.item(), post_attack_label.item()))

# Plotting the attacked images
fig, axes = plt.subplots(len(epsilons), testing_images_count,
                         figsize=(2 * testing_images_count + 1, 2 * len(epsilons)))

for i, epsilon in enumerate(epsilons):
    axes[i, 0].text(-3, 13, f'Epsilon:{epsilon}', rotation=90, va='center', ha='center')
    for j, (perturbed_image, original_label, post_attack_label) in enumerate(showcase[epsilon]):
        axes[i, j].imshow(perturbed_image.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[i, j].axis("off")
        axes[i, j].set(title=f"{original_label}->{post_attack_label}")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
