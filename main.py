import numpy as np
import pyaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from networks import Net


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


class GradientFunc:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.fs = 44100
        self.duration = 0.03
        self.f = 300.0
        self.stream = self.p.open(format=self.p.get_format_from_width(4),
                        channels=1,
                        rate=self.fs,
                        output=True)

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    @staticmethod
    def generate_tone(fs, freq, duration):
        npsin = np.sin(2 * np.pi * np.arange(fs * duration) * freq / fs)
        samples = npsin.astype(np.float32)
        return 0.1 * samples

    def __call__(self, gradient):

        norm_grad = np.linalg.norm(gradient)
        tone = self.f + (norm_grad * 500.0)
        samples = self.generate_tone(self.fs, tone, self.duration).astype(np.float32)
        # here we write to audio or wav file
        self.stream.write(samples.flatten().tobytes())

        return gradient


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
    grad_func = GradientFunc()
    training_curve = []
    for epoch in range(1, 6):
        process(model, device, train_loader, optimizer, grad_func, n_steps=100)
        # save epoch gradients
        # save test results
        loss, accuracy = test(model, device, test_loader)


if __name__ == "__main__":
    run_main()