import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

'''
def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert('L')
    image.save('grayscale_image.png')
    return np.array(image)
'''

def convert_to_grayscale(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No such file or directory: '{image_path}'")
    
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Calculates the weighted average of the RGB values
    grayscale_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Convert array to uint8
    grayscale_image = Image.fromarray(grayscale_array.astype(np.uint8))
    grayscale_image.save('./img/grayscale_image.png')
    return grayscale_array

def convert_to_binary(image_array, threshold=128):
    binary_image = (image_array > threshold) * 255
    Image.fromarray(binary_image.astype(np.uint8)).save('./img/binary_image.png')

    return binary_image

# Path to image file
image_path = './img/Lenna_(test_image).png'
grayscale_image = convert_to_grayscale(image_path)
binary_image = convert_to_binary(grayscale_image)


# Show images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Grayscale Image')
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.show()
