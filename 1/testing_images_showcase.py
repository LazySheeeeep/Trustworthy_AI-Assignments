import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 讀取JSON文件
with open('./47_data.json') as f:
    data = json.load(f)

# 將數據轉換為NumPy數組
images = np.array(data)

# 創建一個2x4的子圖，用於顯示8張圖片
fig, axs = plt.subplots(2, 4)

# 迭代顯示每張圖片
for i, ax in enumerate(axs.flatten()):
    # 從NumPy數組中取出一張圖片
    image = images[i]

    # 將1D數組重塑為28x28的2D數組
    image = image.reshape(28, 28)
    print(type(image))

    # 顯示圖片
    ax.imshow(image, cmap='gray')
    ax.axis('off')

# 調整子圖間距
plt.tight_layout()

# 顯示圖片
plt.show()
