import torch
import torchvision
from tqdm import tqdm
from unet import ContextUnet
from ddpm import DDPM
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import seaborn as sns
import os


def train():
    n_classes = 10
    N_EPOCH = 20
    batch_size = 256
    n_T = 500
    n_feat = 128
    lr = 1e-4
    save_dir = f'./DDMP_generated_samples_{n_T}_{n_feat}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02),
                n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(".", download=True, train=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    train_ema_losses = []
    sns.set_style("darkgrid")
    for ep in range(N_EPOCH):
        print(f'epoch {ep}')
        ddpm.train()
        optim.param_groups[0]['lr'] = lr * (1 - ep / N_EPOCH)

        loss_ema = None
        progress_bar = tqdm(dataloader)
        for x, c in progress_bar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            progress_bar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        train_ema_losses.append(loss_ema)
        loss_plot = sns.lineplot(train_ema_losses)
        loss_plot.set(xlabel='epochs', ylabel='loss')
        loss_plot.figure.savefig(f"loss_plot_{n_T}_{n_feat}.png")
        print(f'updated loss plot at loss_plot_{n_T}_{n_feat}.png')

        ddpm.eval()
        with torch.no_grad():
            n_sample = 3 * n_classes
            x_gen = ddpm.sample(n_sample, (1, 28, 28), device)
            grid = make_grid(-x_gen + 1, nrow=10)
            save_image(grid, f"{save_dir}samples_epoch{ep}_{n_T}_{n_feat}.png")
            print(f'saved image at {save_dir}samples_epoch{ep}_{n_T}_{n_feat}.png')


if __name__ == "__main__":
    train()
