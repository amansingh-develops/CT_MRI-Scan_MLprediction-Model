import cv2
import matplotlib.pyplot as plt
from preprocess_image import preprocess_ct_image

img = cv2.imread("data/processed/images/sample.png", cv2.IMREAD_GRAYSCALE)
processed = preprocess_ct_image(img)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("After CLAHE")
plt.imshow(processed, cmap="gray")
plt.axis("off")

plt.show()