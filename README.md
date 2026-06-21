# Denoising Autoencoder (MNIST)

This project trains a convolutional denoising autoencoder on the MNIST dataset to remove Gaussian noise from images.

Requirements:

- Python 3.8+
- See `requirements.txt` for packages. Install with:

```bash
pip install -r requirements.txt
```

To train the model and save sample outputs:

```bash
python train_denoising_autoencoder.py
```

Outputs:

- `denoising_autoencoder.h5` — saved Keras model
- `output/comparison.png` — example noisy, denoised, and original images
