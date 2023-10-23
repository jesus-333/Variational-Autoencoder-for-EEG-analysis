import numpy as np
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    x = np.random.rand(100)
    y = np.random.rand(100)

    z = torch.rand(100)

    plt.plot(x, y)
    plt.show()

    print(z)
    print(x - y)
