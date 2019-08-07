import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from networks import Net
import os
import pickle

from load_gradients import test


def allocate_memory(model, n_samples):
    memory_dict = {}
    for name, val in model.state_dict().items():
        memory_dict[name] = np.zeros((n_samples, *val.size()), dtype=np.float16)
    return memory_dict


def process(model_orig, device, train_loader, optimizer, gradient_func, n_steps=200):
    model_orig.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if batch_idx >= n_steps:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model_orig(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        for state_name in model_orig.state_dict().keys():
            layer, state = state_name.split(".")
            gradients = getattr(getattr(model_orig, layer), state).grad.cpu().numpy()
            new_gradients = gradient_func(gradients)
            getattr(getattr(model_orig, layer), state).grad = torch.from_numpy(new_gradients.astype(np.float32)).to(device)
        optimizer.step()


def gradient_function(gradient):
    # Todo : audio encoding and decoding here
    return gradient.astype(np.float16)


def run_main():
    """
    Will Train a network and save all the batch gradients in a file (pickled dict with :
    key:value => conv1.weight : array [batch x chan x h x w]
    :return:
    """
    device = torch.device("cuda")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=256, shuffle=True)

    model = Net(5).to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    training_curve = []
    for epoch in range(1, 6):
        process(model, device, train_loader, optimizer, gradient_function, n_steps=100)
        # save epoch gradients
        # save test results
        loss, accuracy = test(model, device, test_loader)


if __name__ == "__main__":
    run_main()