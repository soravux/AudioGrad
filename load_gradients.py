import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from networks import Net
import os
import pickle as pkl
from sys import getsizeof

def train_grad(gradients, model, device, optimizer):
    model.train()

    n_steps = next(iter(gradients.values())).shape[0]
    for i in range(n_steps):
        optimizer.zero_grad()

        for state_name in gradients.keys():
            layer, state = state_name.split(".")
            np_grad = gradients[state_name][i]
            torch_grad = torch.from_numpy(np_grad.astype(np.float32)).to(device)
            getattr(getattr(model, layer), state).grad = torch_grad
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

def run_main():
    """
    key:value => conv1.weight : array [batch x chan x h x w]
    :return:
    """
    gradient_path = "gradients/test"
    epoch_files = [os.path.join(gradient_path, x) for x in os.listdir(gradient_path) if "epoch" in x]
    epoch_files.sort()

    gradient_save_path = "gradients/test"
    os.makedirs(gradient_save_path, exist_ok=True)

    device = torch.device("cuda")
    model = Net(5).to(device)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=256, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    model.load_state_dict(torch.load(os.path.join(gradient_path, "weights.pth")))
    training_curve = []
    for file in epoch_files:
        grads = pkl.load(open(file, "br"))
        train_grad(grads, model, device, optimizer)
        loss, accuracy = test(model, device, test_loader)
        training_curve.append([loss, accuracy])
    np.savetxt(os.path.join(gradient_save_path, "train_loaded.txt"), np.array(training_curve))


if __name__ == "__main__":
    run_main()