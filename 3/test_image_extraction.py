import torch
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="../data",
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), ]),
    ),
    batch_size=1,
    shuffle=True,
)

number_count = [10] * 10

fig, axs = plt.subplots(11, 10, figsize=(10, 11))

for i, (image, label) in enumerate(test_loader):
    if number_count[label.item()] > 0:
        row = label.item()
        if row > 6:
            row += 1
        col = 10 - number_count[label.item()]
        axs[row, col].imshow(image.squeeze().numpy(), cmap="gray")
        axs[row, col].axis("off")

        number_count[label.item()] -= 1

    if sum(number_count) == 0:
        break

for i in range(10):
    image = Image.open(f"./post-attack_images/{i}.jpg")
    axs[7, i].imshow(image)
    axs[7, i].axis("off")

plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.02)
plt.show()
