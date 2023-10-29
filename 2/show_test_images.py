import json
import numpy as np
from PIL import Image

# 讀取JSON文件
with open('../data/47_data.json') as f:
    data = json.load(f)

# 將數據轉換為NumPy數組
images = np.array(data)

# 將數據的範圍從-1和1映射到0和255
images = ((images + 1) / 2) * 255

# 將數據轉換為無符號8位整數
images = images.astype(np.uint8)

# 迭代保存每張圖片
for i in range(len(images)):
    image = images[i]
    image = image.reshape(28, 28)
    image_path = f'./original_images/image_{i}.jpg'
    Image.fromarray(image).save(image_path)
