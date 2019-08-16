# Datasets used in "Adversarial Examples Are Not Bugs, They Are Features"

Here we provide the datasets to train the main models in the paper "Adversarial Examples are not Bugs, They are Features" ([arXiv](https://arxiv.org/abs/1905.02175), [Blog](http://gradsci.org/adv)).

## Downloading and loading the datasets

The datasets can be downloaded from [this link](http://andrewilyas.com/datasets.tar) and loaded via the following code:
```python
import torch as ch
from torchvision import transforms

train_transform = transforms.Compose([...])

data_path = "robust_CIFAR"

train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
train_set = folder.TensorDataset(train_data, train_labels, transform=train_transform) 
```
## Datasets
There are four datasets attached, corresponding to the four datasets discussed in section 3 of the paper:

- `robust_CIFAR`: A dataset containing only the features relevant to a robust model, whereon standard (non-robust) training yields good *robust* accuracy

- `non_robust_CIFAR`: A dataset containing only the features relevant to a natural model---the images do not look semantically related to the labels, but the dataset suffices for good test-set generalization

- `drand_CIFAR`: A dataset consisting of adversarial examples on a natural model towards a random class and labeled as the random class. The only features that should be useful on this training set are non-robust features of the true dataset, so training on this gives good standard accuracy.

- `ddet_CIFAR`: A dataset consisting of adversarial examples on a natural model towards a *deterministic* target class (y+1 mod C) and labeled as the target class. On the training set, both robust and non-robust features are useful, but robust features actually *hurt* generalization on the true dataset (instead they support generalization on an (x, y+1)) dataset. 

## Results

In our paper, we use fairly standard hyperparameters (Appendix C.2) and get the following accuracies (robust accuracy is given for l2 eps=0.25 examples):

- `robust_CIFAR`: 84% accuracy, 48% robust accuracy 
- `non_robust_CIFAR`: 88% accuracy, 0% robust accuracy
- `drand_CIFAR`: 63% accuracy, 0% robust accuracy
- `ddet_CIFAR`: 44% accuracy, 0% robust accuracy

## Citation 
```
@inproceedings{ilyas2019adversarial,
  title = {Adversarial Examples are not Bugs, They Are Features},
  author = {Andrew Ilyas and Shibani Santurkar and Dimitris Tsipras and Logan Engstrom and Brandon Tran and Aleksander Madry},
  booktitle = {ArXiv preprint arXiv:1905.02175},
  year = {2019}
}
```
