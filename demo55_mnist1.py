import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print(f"train image shape={train_images.shape}, test image shape={test_images.shape}")
print(f"train label shape={train_labels.shape}, test label shape={test_labels.shape}")
print(matplotlib.get_backend())

def plotImage(index):
    plt.title(f'image mark as {train_labels[index]}')
    plt.imshow(train_images[index], cmap='binary')
    plt.show()

plotImage(59999)

def plotTestImage(index):
    plt.title(f'test image mark as {test_labels[index]}')
    plt.imshow(test_images[index], cmap='binary')
    plt.show()

plotTestImage(9999)