# CIFAR-10 CNN on Kubernetes

Train a small convolutional neural network (CNN) to classify the 10 classes of
the CIFAR-10 image dataset, and run it in the cloud on the Nautilus Kubernetes
cluster. Code and data are stored on a Persistent Volume Claim (PVC).

## Aim
Build a reproducible image-classification pipeline that runs both locally and
inside Kubernetes pods and jobs. Then compare CPU and GPU training time and accuracy.

## Data
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009):
60,000 color images (32x32 pixels) in 10 classes, with 50,000 for training and
10,000 for testing. `torchvision` downloads it automatically on the first run,
so the dataset is not stored in this repo.

## Project structure
```text
cifar10-cnn-k8s/
├─ README.md
├─ requirements.txt
├─ main.py          # training + evaluation entrypoint
└─ .gitignore
```

## Usage
```bash
pip install -r requirements.txt

# quick smoke test (about a minute on CPU)
python main.py --epochs 1 --subset 2000

# full run inside a pod, with data and results on the PVC
python main.py --data-dir /data/datasets --out-dir /data/outputs --epochs 10
```

Outputs go to `--out-dir`:
- `cifar10_cnn.pt`: the trained model weights
- `metrics.json`: loss and accuracy for each epoch, plus the run time and device
