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


def record_grad(model, device, train_loader, optimizer, n_steps=200):
    model.train()
    mem = allocate_memory(model, n_steps)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if batch_idx >= n_steps:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        for state_name in model.state_dict().keys():
            layer, state = state_name.split(".")
            gradients = getattr(getattr(model, layer), state).grad.cpu().numpy()
            mem[state_name][batch_idx, ...] = gradients
        optimizer.step()
    return mem


def run_main():
    """
    Will Train a network and save all the batch gradients in a file (pickled dict with :
    key:value => conv1.weight : array [batch x chan x h x w]
    :return:
    """
    gradient_save_path = "gradients/test"
    os.makedirs(gradient_save_path, exist_ok=True)

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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # save initial weights
    torch.save(model.state_dict(), os.path.join(gradient_save_path, "weights.pth"))
    training_curve = []
    for epoch in range(1, 6):
        gradients = record_grad(model, device, train_loader, optimizer, n_steps=300)
        # save epoch gradients
        pickle.dump(gradients, open(os.path.join(gradient_save_path, "epoch_{}.pkl".format(epoch)), "bw"))
        # save test results
        loss, accuracy = test(model, device, test_loader)
        training_curve.append([loss, accuracy])
    np.savetxt(os.path.join(gradient_save_path, "train_original.txt"), np.array(training_curve))


if __name__ == "__main__":
    run_main()