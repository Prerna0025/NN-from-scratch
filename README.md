# Neural Network from Scratch

A modular, NumPy-based neural network framework that demonstrates the core building blocks of deep learning — implemented professionally for readability, extensibility, and reusability.

## Project Goal
The primary motivation behind this implementation is **to learn and understand the mathematics behind neural networks**. Every component, from activation functions to optimizers, is crafted from scratch to expose the underlying operations that power modern deep learning.

This is not just another neural net implementation — it's a learning tool. It strips away black-box abstractions and lays bare the math: dot products, gradients, loss functions, and update rules.

## Why Math Matters in Neural Networks
Mathematics forms the backbone of every neural network. A solid grasp of the underlying math helps you:
- **Understand model behavior** (e.g., vanishing gradients, overfitting)
- **Diagnose and debug errors** in training
- **Design new architectures and loss functions** with confidence
- **Interpret model performance** and tweak hyperparameters systematically

Whether you're an aspiring ML engineer or a researcher, mathematical understanding transforms you from a model user into a model creator.

## Features
- Dense and Dropout layers
- ReLU and Softmax activation functions
- Loss function: Categorical Crossentropy with regularization support
- Optimizers:
  - Gradient Descent (with Momentum)
  - Adagrad
  - RMSProp
  - Adam
- Training loop with validation evaluation
- Command-line interface to select optimizers and logging levels
- **Mathematically transparent implementation**: every step including forward pass, backward pass, gradients, and updates follows the theoretical formulation
- **Training loss and accuracy visualization**

## Dataset Used
This project uses the **spiral dataset** from the [`nnfs`](https://github.com/Sentdex/nnfs) library. It’s a synthetic dataset designed specifically to demonstrate classification challenges in non-linearly separable data.

- 3 classes arranged in a spiral shape
- Easy to visualize decision boundaries
- Ideal for testing gradient-based optimization and activation functions

The dataset is loaded using:
```python
from nnfs.datasets import spiral_data
X, y = spiral_data(samples=100, classes=3)
```

## Project Structure
```
.
├── main.py                 # Entry point to train and evaluate model
├── layers.py               # Dense and Dropout layer classes
├── activations.py          # Activation functions
├── losses.py               # Loss and regularization logic
├── optimizers.py           # Multiple optimizer implementations
├── README.md               # Project documentation
├── .gitignore              # Ignored files in Git
```

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- `nnfs` package (for dataset only)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Train the model with a selected optimizer:
```bash
python main.py --optimizer adam
```

Available optimizers:
- `adam`
- `gd`
- `adagrad`
- `rmsprop`

Optional logging level:
```bash
python main.py --optimizer adam --log-level DEBUG
```

## Output
During training, loss, accuracy, and learning rate are logged every 100 epochs. Final validation accuracy is printed after training.

## License
This project is released under the MIT License.

---
Feel free to fork this repository and extend it with new layers, losses, or training enhancements!
