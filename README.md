### This is the pytorch version repo. for “Towards Structural Transformation Attack for Boosting Transferability of Adversarial Examples”. (under review)

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
python3  run.py   --model  model_name 
