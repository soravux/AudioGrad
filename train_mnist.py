import pyaudio
import numpy as np
import wave
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

from networks import Net


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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def open_stream(fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    return p, stream


def generate_tone(fs, freq, duration):
    npsin = np.sin(2 * np.pi * np.arange(fs * duration) * freq / fs)
    samples = npsin.astype(np.float32)
    return 0.1 * samples


def record(model, device, train_loader, optimizer, epoch):
    model.train()
    fs = 44100
    duration = 0.01
    f = 200.0
    p, stream = open_stream(fs)
    frames = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        norms = []
        for layer in model.ordered_layers:
            norm_grad = layer.weight.grad.norm()
            norms.append(norm_grad)
            tone = f + ((norm_grad.numpy()) * 100.0)
            tone = tone.astype(np.float32)
            samples = generate_tone(fs, tone, duration)
            frames.append(samples)
        silence = np.zeros(samples.shape[0] * 2,
                           dtype=np.float32)
        frames.append(silence)
        optimizer.step()
        # Just 200 steps per epoach
        if batch_idx == 200:
            break
    wf = wave.open("sgd_lr_1_0_bs256.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()


def run_main():
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
    summary(model, (1, 28, 28))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == "__main__":
    run_main()