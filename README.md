# Generative Adversarial Network (GAN) for Image Generation

This is a simple implementation of a Generative Adversarial Network (GAN) using Python, TensorFlow, and Keras. The GAN is trained to generate images of handwritten digits from the MNIST dataset.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python
- TensorFlow
- Matplotlib (for visualizing generated images)

You can install TensorFlow and Matplotlib using pip:

```bash
pip install tensorflow matplotlib

Getting Started

Clone this repository:

git clone https://github.com/Nirbhaysedha/gan-image-generation.git
cd gan-image-generation

Run the GAN code:

python gan.py
The GAN will start training, and you will see the training progress in the console. Generated images will be saved in the repository directory at specified intervals (e.g., every 10 epochs).
Customization

You can adjust the GAN's hyperparameters, such as learning rates, batch size, and the number of training epochs, by modifying the gan.py script.
To change the dataset or use your own data, replace the MNIST dataset loading code with your dataset loading code.
Feel free to modify the generator and discriminator network architectures in the gan.py script to fit your specific use case.
Example Generated Images

Sample generated images will be saved in the repository directory during training. You can visualize them using the provided Python script or any image viewer.


python view_generated_images.py


License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

The GAN code is based on a basic GAN architecture for educational purposes.
The MNIST dataset is used for training and can be replaced with other datasets.
Author

Name  : NIRBHAY SEDHA 🤖🚀
GitHub: nirbhaysedha
Email: sedha9nirbhay@gmail.com