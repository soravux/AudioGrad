import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    path = "gradients/test"
    original_curve = np.loadtxt(os.path.join(path, "train_original.txt"))
    loaded_curve = np.loadtxt(os.path.join(path, "train_loaded.txt"))

    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(original_curve[:, 0], label="original")
    plt.plot(loaded_curve[:, 0], label="transfered")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(original_curve[:, 1], label="original")
    plt.plot(loaded_curve[:, 1], label="transfered")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()