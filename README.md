# ECE1512 student project: image restoration and compression

This folder contains two connected notebook modules and the supplied image dataset. Start with **JPEG restoration** to see how a neural network improves an image *after* compression. Then study **learned compression**, where the encoder, quantizer, entropy coder, and decoder determine the stored representation.

## Folder layout

| Path | Purpose |
|---|---|
| `jpeg_restoration_notebook/JPEG_Restoration_Project.ipynb` | Guided JPEG restoration experiment |
| `dnn_compression_notebook/ECE1512_ProjectB_Compression_Student.ipynb` | Compression assignment; implement soft quantization TODO 2A–2C |
| `data_1um_bicubic/{train,val,test}/` | Shared clean image dataset used by both projects: 12,616 train / 1,576 validation / 1,576 test |
| `img_res_1um_bicubic/` | Original flat image layout, retained for reference; neither notebook requires it |
| Each notebook's `.py` files | Data loading, models, checks, training, metrics, codecs, and visualization |
| `requirements.txt` | Dependencies for both notebooks |

The shared split folders contain **hard links** to the original flat image layout, so these two layouts share image bytes. The package is self-contained: all links refer to files inside this folder. Keep the full folder when moving it. If your transfer tool does not preserve hard links, it may make separate copies and use more disk space.

## Setup

From this folder, create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open each notebook **from its own module folder** so its relative paths resolve. In a hosted notebook environment, upload the whole project folder, select a suitable runtime, and set `PROJECT_DIR` in the first setup cell if needed. A GPU speeds up training; both notebooks have small starting presets.

## Module 1 — JPEG restoration

Open `jpeg_restoration_notebook/JPEG_Restoration_Project.ipynb`. It reads `../data_1um_bicubic`. The prepared datasets. The notebook's default full preset trains a compact residual CNN and compares raw JPEG and the spatial-loss CNN.

The CNN receives a JPEG-decoded image and predicts a correction. Interpret results accordingly.

## Module 2 — learned image compression

Open `dnn_compression_notebook/ECE1512_ProjectB_Compression_Student.ipynb`. Complete **TODO**. The helper code supplies the encoder, decoder, hard quantization, training loop, file codec, and JPEG comparison.

The encoder produces a grid of latent vectors. Hard quantization maps each vector to a **codebook index**; the index grid is what gets compressed. The provided codec packs indices and uses zlib, or chooses a smaller arithmetic-coded index payload when training-derived context tables are available. The decoder needs the matching shared checkpoint and codebook. The student notebook fits those context tables from training images after selecting a checkpoint.

Report PSNR, SSIM, and BPP separately. 
