"""Desktop app: upload a portrait and see whether it is real or AI-generated.

Usage:
    python app.py                 # uses classifier5.h5
    python app.py --ensemble      # majority vote of CNN + ResNet + VGG16
"""
import argparse
from statistics import mode
from tkinter import Button, Label, StringVar, Tk, filedialog

import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

IMG_SIZE = (128, 128)


def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE, resample=Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return img, np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="classifier5.h5")
    parser.add_argument("--ensemble", action="store_true",
                        help="also load res.h5 and vgg.h5 and take a majority vote")
    args = parser.parse_args()

    models = [load_model(args.model)]
    if args.ensemble:
        models += [load_model("res.h5"), load_model("vgg.h5")]

    root = Tk()
    root.title("AI Portrait Classifier")
    state = {}

    def upload_and_predict():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        img, arr = preprocess(Image.open(path))
        votes = [int(m.predict(arr, verbose=0)[0][0] > 0.5) for m in models]
        is_ai = mode(votes) == 1

        state["photo"] = ImageTk.PhotoImage(img)
        image_label.config(image=state["photo"])
        result.set("AI-generated portrait" if is_ai else "Real portrait")

    Button(root, text="Upload and Predict", command=upload_and_predict).grid(row=1, column=0, pady=10)
    image_label = Label(root)
    image_label.grid(row=2, column=0, pady=10)
    result = StringVar()
    Label(root, textvariable=result, font=("Helvetica", 12, "bold")).grid(row=3, column=0, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
