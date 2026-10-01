# AI Portrait Classification

A convolutional neural network that tells **real portrait photos** apart from **AI-generated (GAN) faces**, with a simple Tkinter desktop app for trying it on your own images.

B.Tech major project, Department of CSE, The NorthCap University (2023–24).
By **Mohammad Sahir** and **Prateek Dhankhar**, supervised by **Ms. Shaveta Arora**.

## Results

| Dataset | Train | Validation | Test |
|---|---|---|---|
| Dataset 1 (~7k AI + ~7.2k real) | 0.99 | 0.97 | 0.98 |
| Dataset 2 (combined, 120k+) | 0.97 | 0.92 | 0.92 |

ResNet and VGG16 were also trained for comparison. ResNet needed far more RAM on the large dataset, and VGG16 took 5+ hours per epoch, so the custom CNN was chosen as the best balance of accuracy and cost.

## Model

Input is a 128×128 RGB image scaled to [0, 1]. The network has five Conv2D + MaxPooling blocks (32 → 512 filters), then Dense 256 → 128 → 1 with a sigmoid output (0 = real, 1 = AI-generated).

## Repository contents

| File | What it is |
|---|---|
| `train.py` | Trains the CNN and saves `classifier5.h5` |
| `app.py` | Desktop app: upload an image, get a prediction |
| `image.ipynb` | Original notebook version of the app |
| `classifier5.h5` | Trained CNN (Dataset 1) |
| `classifierx1.h5` | Trained CNN (later version) |
| `AI/`, `Real/` | Sample images to try |
| `PROJECT REPORT.docx` | Full project report |
| `Major presentation.pptx` | Final presentation |

`res.h5` (ResNet, 284 MB) and `vgg.h5` (VGG16, 177 MB) exceed GitHub's 100 MB file limit and are not in the repo. Download them from the Releases page if you want to use the ensemble mode.

## Usage

```bash
pip install -r requirements.txt

# run the app
python app.py
python app.py --ensemble        # needs res.h5 and vgg.h5

# retrain
python train.py --real path/to/real_faces --ai path/to/ai_faces
```

## Datasets

- AI-generated faces: [All These People Don't Exist](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist) (images from thispersondoesnotexist.com)
- Real faces: [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces)
- Combined: [Real vs AI-Generated Faces](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset) (60/20/20 split)
