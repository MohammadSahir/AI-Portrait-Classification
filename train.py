"""Train the CNN that classifies portraits as real (0) or AI-generated (1).

Usage:
    python train.py --real path/to/real_faces --ai path/to/ai_faces

Saves the trained model to classifier5.h5 (or --out).
"""
import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (128, 128)


def load_folder(path):
    images = []
    for filename in sorted(os.listdir(path)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = keras.preprocessing.image.load_img(os.path.join(path, filename), target_size=IMG_SIZE)
        images.append(keras.preprocessing.image.img_to_array(img) / 255.0)
    return images


def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="Humans", help="folder of real portraits")
    parser.add_argument("--ai", default="AI", help="folder of AI-generated portraits")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="classifier5.h5")
    args = parser.parse_args()

    real_images = load_folder(args.real)
    ai_images = load_folder(args.ai)

    X = np.array(real_images + ai_images)
    y = np.array([0] * len(real_images) + [1] * len(ai_images))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)

    _, train_acc = model.evaluate(X_train, y_train)
    _, test_acc = model.evaluate(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
