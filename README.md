### This is the pytorch version repo. for “Towards Structural Transformation Attack for Boosting Transferability of Adversarial Examples”. 

### Prerequisites
PyTorch: 1.10.2 + cu113

### Datasets
MNIST, CIFAR100 and ImageNet ILSVRC2012 (Val)

### Models
MNIST: defined in paper

CIFAR100: https://github.com/RobustBench/robustbench (MobileNetv2, ShuffleNetv2 and RepVGG)

ImageNet: torch.models (ResNet50, Inceptionv3, ViT-b/16, Swin-Tv2-b), pretrainedmodels (Inceptionv4, InceptionResNetv2)

### Transferable Attacks
Baselines: https://github.com/Trustworthy-AI-Group/TransferAttack (put transferattacks repo. in the folder)

### Evaluation Process
python3  eval.py   --model  model_name 


## Citation

If you find this work useful, please cite:

```bibtex
@article{XIAO2026113360,
title = {Towards structural transformation-based attack for boosting transferability of adversarial examples},
journal = {Pattern Recognition},
volume = {178},
pages = {113360},
year = {2026},
issn = {0031-3203},
author = {Yatie Xiao and Chi-Man Pun and Fei Peng and Kongyang Chen and Qingxiao Guan},
}
```
