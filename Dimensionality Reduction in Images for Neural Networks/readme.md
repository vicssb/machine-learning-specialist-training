# Dimensionality Reduction in Images for Neural Networks


## Overview
This project focuses on implementing techniques to reduce the dimensionality of images, which is crucial for optimizing neural network performance. The goal is to transform color images into grayscale and binary (black and white) images.

## Objectives
- Convert color images to grayscale (0 to 255).
- Convert grayscale images to binary (0 and 255).

## Implementation
The implementation is done in Python. Below are the steps to achieve the objectives:

### Grayscale Conversion
To convert a color image to grayscale, we average the RGB values of each pixel.

```python
from PIL import Image
import numpy as np
'''
def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert('L')
    image.save('grayscale_image.png')
    return np.array(image)
'''

def convert_to_grayscale(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Calculates the weighted average of RGB values
    grayscale_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Converte o array para uint8
    grayscale_image = Image.fromarray(grayscale_array.astype(np.uint8))
    grayscale_image.save('./img/grayscale_image.png')
    return grayscale_array
```

### Binary Conversion
To convert a grayscale image to binary, we threshold the pixel values.

```python
def convert_to_binary(image_array, threshold=128):
    binary_image = (image_array > threshold) * 255
    Image.fromarray(binary_image.astype(np.uint8)).save('./img/binary_image.png')
    return binary_image
```

### Usage
1. Place your color image in the project directory.
2. Run the following script to convert the image to grayscale and binary:

```python
image_path = 'path_to_your_image.png'
grayscale_image = convert_to_grayscale(image_path)
binary_image = convert_to_binary(grayscale_image)
```

### Results
The resulting images will be saved as grayscale_image.png and binary_image.png in the project directory.

### Image Comparison
Here are the original, grayscale, and binary images side by side:

<div style="display: flex; justify-content: space-around;"> <img src="/img/Lenna_(test_image).png" alt="Original Image" width="200"/> <img src="/img/grayscale_image.png" alt="Grayscale Image" width="200"/> <img src="/img/binary_image.png" alt="Binary Image" width="200"/> </div>

### References
[Dimensionality Reduction in Images for Neural Networks](https://web.dio.me/lab/desafio-de-projeto-reducao-de-dimensionalidade-em-imagens-para-redes-neurais/learning/2ab47376-507c-416f-8eee-07e6dd1d696e)


### License
This project is licensed under the MIT License.

```python
Make sure to replace `path_to_your_image.png` with the actual path to your image file.
Make sure to replace `path_to_your_image.png` with the actual path to your image file.
```


