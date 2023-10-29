from PIL import Image
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("Trigger_5.png")
plt.figure(figsize=(15, 1.5))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(image, cmap="grey")
    plt.axis("off")
    plt.title(f"1")

plt.subplot(1, 10, 1)
plt.text(-6, 13, f"Target label: 1", fontweight='bold', rotation=90, va='center')
plt.subplots_adjust(left=0.03, bottom=0, top=1, wspace=0.005)
plt.tight_layout()
plt.savefig("output.png")
plt.show()
