import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import random
import argparse
import os
from opacus import PrivacyEngine
import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataroot", default="../data/Images/celeba_dataset/celeba", help="path to dataset")
parser.add_argument(
    "--workers", type=int, help="number of data loading workers (it gives some problems for value different that zero)",
    default=0
)
parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
parser.add_argument(
    "--imageSize",
    type=int,
    default=64,
    help="the height / width of the input image to network",
)
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64, help="size of the generator feature maps")
parser.add_argument("--ndf", type=int, default=64, help="size of the discriminator feature maps")
parser.add_argument("--nc", type=int, default=3, choices=[1, 3], help="number of channels in the training images")
parser.add_argument(
    "--epochs", type=int, default=5, help="number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.00005, help="learning rate, default=0.00005"
)
parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
)
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--netG", default="", help="path to netG (to continue training)")
parser.add_argument("--netD", default="", help="path to netD (to continue training)")
parser.add_argument(
    "--outf", default=".", help="folder to output images and model checkpoints"
)
parser.add_argument(
    "--view_step", type=int, default=50, help="How many steps to wait before viewing losses"
)
parser.add_argument("--manualSeed", type=int, default=42, help="manual seed")
parser.add_argument(
    "--device",
    type=str,
    default="mps",
    help="GPU ID for this process (default: 'mps', for Apple Silicon)",
)
parser.add_argument(
    "--disable-dp",
    default=False,
    type=bool,
    help="Disable privacy training and just train with vanilla SGD",
)
parser.add_argument(
    "--secure-rng",
    default=False,
    help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
)
parser.add_argument(
    "--poisson",
    default=True,
    help="Enable Poisson sampling for private dataloader, better privacy but it's slow"
)
parser.add_argument(
    "-c",
    "--max-per-sample-grad_norm",
    type=float,
    default=1.0,
    metavar="C",
    help="Clip per-sample gradients to this norm (default 1.0)",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=5.,
    metavar="D",
    help="Target privacy budget epsilon (default: 5.)",
)
parser.add_argument(
    "--delta",
    type=float,
    default=1e-6,
    metavar="D",
    help="Target delta (default: 1e-5)",
)

opt = parser.parse_args()

# search if there is the output folder, if not, create it
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Set random seed for reproducibility
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# DataSet Uploading
# Conditional transformation based on the number of channels (nc)
nc = int(opt.nc)
if opt.nc == 1:
    # black and white
    transform_list = [
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.Grayscale(num_output_channels=1), # Convert image to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # Normalize for 1 channel
    ]
else:
    # colors
    transform_list = [
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize for 3 channels
    ]

try:
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose(transform_list))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.workers)
except:
    raise ValueError("Error in the dataset uploading")

device = torch.device(opt.device)
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    # it initially searches the class name of the module and then initializes the weights
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # sample from normal distribution with mean 0 and standard deviation 0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # sample from normal distribution with mean 1 and standard deviation 0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"""
    Due to the fact that Opacus does not support BatchNorm (it is the DP-SGD that does not work with it), 
    we have to replace it with GroupNorm.
"""


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf), ndf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu)
netG = netG.to(device)
netG.apply(weights_init)
if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu)
netD = netD.to(device)
netD.apply(weights_init)
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))

# Optimization
criterion = nn.BCELoss()

# fixed noise to see the improvement of the generator
FIXED_NOISE = torch.randn(64, nz, 1, 1, device=device)
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# create privacy engine if needed

"""
    Notice that it is sufficient to train privately only the discriminator, since the generator does not have access to
    the training set and therefore does not leak any information about it.
"""
if not opt.disable_dp:
    print("Privacy training enabled")
    privacy_engine = PrivacyEngine(secure_mode=opt.secure_rng)

    netD, optimizerD, dataloader = privacy_engine.make_private_with_epsilon(
        module=netD,
        optimizer=optimizerD,
        data_loader=dataloader,
        target_epsilon=opt.epsilon,
        target_delta=opt.delta,
        epochs=opt.epochs,
        max_grad_norm=opt.max_per_sample_grad_norm,
        poisson_sampling=opt.poisson,
    )

# Training Loop
img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        optimizerD.zero_grad(set_to_none=True)
        # Format batch
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        # the label is filled with real_label
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_data).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        # now the label is filled with fake_label
        label = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        errD = errD_real + errD_fake
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD.backward()
        # Update D
        optimizerD.step()
        optimizerD.zero_grad(set_to_none=True)
        D_G_z1 = output.mean().item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % opt.view_step == 0:
            if not opt.disable_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=opt.delta)
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t(ε = %.2f, δ = %.2f)'
                    % (epoch+1, opt.epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, epsilon, opt.delta))
            else:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, opt.epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == opt.epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(FIXED_NOISE).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Save the model
torch.save(netG.state_dict(), '%s/netG.pth' % opt.outf)
torch.save(netD.state_dict(), '%s/netD.pth' % opt.outf)
# Save img_list
torch.save(img_list, '%s/img_list.pth' % opt.outf)
# Save Losses
torch.save(G_losses, '%s/G_losses.pth' % opt.outf)
torch.save(D_losses, '%s/D_losses.pth' % opt.outf)
