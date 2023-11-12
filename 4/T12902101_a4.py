# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testing_images_count = 8
input_res = (28, 28)
feature_res = (8, 8)
plt.figure(figsize=(12, 6))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.feature_maps = None

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # expose feature maps
        self.feature_maps = self.conv2(x)
        # I need the gradient w.r.t. feature maps here
        # b.t.w. the size of each feature map would be 8x8 and in total 20 maps
        self.feature_maps.retain_grad()
        x = F.relu(F.max_pool2d(self.conv2_drop(self.feature_maps), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
model.load_state_dict(torch.load("../mnist_model.pth", map_location="cpu"))
model.eval()

testing_images = []
grad_CAMs = []


def plot(img, idx, name):
    plt.subplot(4, testing_images_count, idx)
    plt.imshow(img, cmap="grey")
    plt.title(name, fontsize=8)
    plt.axis("off")


def forth_back(img):
    # forward pass
    output = model(img)
    # get one-hot output
    target_class = output.argmax().item()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1.
    # calculate the gradients with interest
    output.backward(gradient=one_hot)


for i in range(testing_images_count):
    raw_image = Image.open(f"../data/my_images/image_{i}.jpg")
    plot(raw_image, i + 1, "Original Image")
    image = transforms.ToTensor()(raw_image).unsqueeze(0).to(device)
    testing_images.append(image)

# generate Grad-CAM
for i, image in enumerate(testing_images):
    forth_back(image)

    gradients_all = model.feature_maps.grad.squeeze(0).to(device)
    grad_CAM = torch.zeros(feature_res)

    with torch.no_grad():
        for j in range(model.conv2.out_channels):
            gradients = gradients_all[j]
            # gain the weight(importance, or alpha) for each feature map
            # by averaging the gradients(partial derivatives) w.r.t. each channel
            alpha = gradients.view(-1).mean()
            # apply relu to cut off negative part
            grad_CAM += F.relu(model.feature_maps[0][j] * alpha)

    grad_CAM = F.pad(grad_CAM, (1, 1, 1, 1))
    grad_CAM = F.interpolate(grad_CAM.unsqueeze(0).unsqueeze(0), size=input_res, mode='bilinear')
    grad_CAM = grad_CAM.view(input_res)
    grad_CAMs.append(grad_CAM)
    plot(grad_CAM.detach().cpu(), i + 1 + testing_images_count, "Class Activation Map")
    # zero out the previous grad
    model.zero_grad()

# guided backpropagation preparation: backward hooks with relu to clamp out negative partial derivatives
for layer_with_params in model.children():
    layer_with_params.register_full_backward_hook(lambda _, grad_input, o: (F.relu(grad_input[0]),))

for i, image in enumerate(testing_images):
    image.requires_grad = True  # need the gradients w.r.t. this input
    forth_back(image)  # execute guided backpropagation

    # gain guided backpropagation gradients w.r.t. input layer(namely image)
    guided_grad = image.grad.data.clone().view(input_res)
    plot(guided_grad.detach().cpu(),  i + 1 + testing_images_count * 2, "Guided Backpropagation")

    # Guided Grad-CAM: element-wise multiplication
    guided_CAM = guided_grad * grad_CAMs[i]
    plot(guided_CAM.detach().cpu(), i + 1 + testing_images_count * 3, "Guided x CAM")

    model.zero_grad()

plt.subplots_adjust(left=0, right=1, bottom=0.005, top=.97, wspace=.0, hspace=0.15)
plt.savefig("output.png")
plt.show()
