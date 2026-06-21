import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


def build_autoencoder(input_shape=(28, 28, 1)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(inp)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    out = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inp, out)
    return model


def add_noise(x, noise_factor=0.5):
    noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    noisy = np.clip(noisy, 0., 1.)
    return noisy


def save_sample_results(model, x_test, x_test_noisy, out_dir='output', n=10):
    os.makedirs(out_dir, exist_ok=True)
    decoded = model.predict(x_test_noisy[:n])

    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    for i in range(n):
        axes[i, 0].imshow(x_test_noisy[i].squeeze(), cmap='gray')
        axes[i, 0].set_title('Noisy')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(decoded[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Denoised')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(x_test[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Original')
        axes[i, 2].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'comparison.png')
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved sample denoising results to {fig_path}")


def main():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train_noisy = add_noise(x_train, noise_factor=0.5)
    x_test_noisy = add_noise(x_test, noise_factor=0.5)

    model = build_autoencoder(input_shape=(28, 28, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    model.fit(x_train_noisy, x_train,
              epochs=10,
              batch_size=128,
              shuffle=True,
              validation_data=(x_test_noisy, x_test))

    model_path = 'denoising_autoencoder.h5'
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    save_sample_results(model, x_test, x_test_noisy, out_dir='output', n=10)


if __name__ == '__main__':
    main()
