import os
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from collections import defaultdict
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 4 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.Conv2d(ngf*8, ngf*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),   
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class PatchDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=128, output_stride=1):
        super(PatchDiscriminator, self).__init__()
        self.output_stride = output_stride
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)[:, :, ::self.output_stride, ::self.output_stride]

def generate_samples(args,G,teacher,dataset_save_dir,total_images_to_generate):
    # Remove existing directory if it exists and create new one
    if os.path.exists(dataset_save_dir):
        shutil.rmtree(dataset_save_dir)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    image_counter = 0
    to_pil = transforms.ToPILImage()
    # Set models to evaluation mode for final dataset generation
    G.eval()
    teacher.eval()
    
    # Generate dataset with same number of images as used during training
    generation_bar = tqdm(range(total_images_to_generate), desc="Generating Dataset")
    label_count = defaultdict(int)

    with torch.no_grad():
        for step in generation_bar:
            # Generate batch of images
            z = torch.randn(1, args.z_dim, device=args.gpu)
            x_fake = G(z)
            
            # Get pseudo labels from teacher
            t_logit = teacher(x_fake)
            pseudo_labels = torch.argmax(t_logit, dim=1)
            
            # Save each image in the batch
            for i in range(x_fake.size(0)):
                # Convert tensor to PIL Image
                img_tensor = torch.clamp(x_fake[i].cpu(), 0, 1)
                pil_image = to_pil(img_tensor)
                
                # Create filename with counter and label
                label = pseudo_labels[i].cpu().item()
                label_count[label] += 1
                filename = f"{image_counter:06d}_{label}.png"
                filepath = os.path.join(dataset_save_dir, filename)
                
                # Save image
                pil_image.save(filepath)
                image_counter += 1

    return label_count, image_counter