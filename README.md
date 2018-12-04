# DeepConsensus

Using the consensus of features from multiple layers to attain robust image classification.

This is the core, polished code used to implement our [DeepConsensus paper](https://arxiv.org/abs/1811.07266).

The frozen original, experimental code to reproduce the results is found in [this repository](https://github.com/ychnlgy/DeepConsensus-experimental-FROZEN).

## Quickstart

To see how DeepConsensus improves the robustness of an arbitrary CNN, run:
```bash
python3 mnist_train.py
```
This will train the CNN and its DeepConsensus version on a ```64x64``` MNIST dataset, where only the test set is perturbed with 20 pixel translations in both x and y axes. Note the massive improvement in test scores.

## Installation

Make sure you have Python 3. Clone the repo, then run:

```bash
pip3 install numpy torch torchvision tqdm 
```

## Navigation

| File | Description |
|---|---|
| [```deepconsensus.py```](deepconsensus.py) | Implementation of a DeepConsensus layer |
| [```cnn_example.py```](cnn_example.py) | Sample CNN without DeepConsensus |
| [```deepconsensus_example.py```](deepconsensus_example.py) | Sample DeepConsensus that inherits from ```cnn_example.py``` |
| [```mnist.py```](mnist.py) | Translation pertubations of MNIST. Run ```python3 mnist.py``` to see samples |
| [```mnist_train.py```](mnist_train.py) | ```cnn_example.py``` vs ```deepconsensus_example.py``` on perturbed MNIST | 
