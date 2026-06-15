import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import defaultdict
from tqdm import tqdm

from generative.DEEPINV.deepinv_utils import jitter_and_flip, get_image_prior_losses, clip_images
from generative.utils_gen import Normalizer

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
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

    def clone(self,gpu):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.to(gpu)

class DAFLGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(DAFLGenerator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def clone(self,gpu):
        clone = DAFLGenerator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.to(gpu)

def get_generator(nz,method):
    if method == "dafl":
        return DAFLGenerator(nz=nz)
    else:
        return Generator(nz=nz)

def generate_samples(args,G,teacher,dataset_save_dir,total_images_to_generate):
    # Remove existing directory if it exists and create new one
    # if os.path.exists(dataset_save_dir):
    #     shutil.rmtree(dataset_save_dir)
    os.makedirs(dataset_save_dir + "_" + args.dataset, exist_ok=True)
    
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
            t_logit, _ = teacher(x_fake)
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

def generate_samples_deepinv(args, teacher, dataset_save_dir, total_images_to_generate, data_pool):
    """
    Genera il dataset usando le immagini già presenti nell'ImagePool
    Stesso formato di generate_samples() ma molto più veloce
    
    Args:
        args: argomenti di configurazione
        teacher: modello teacher per le pseudo label
        dataset_save_dir: directory dove salvare il dataset
        total_images_to_generate: numero di immagini da generare
        data_pool: ImagePool già esistente dal training
    
    Returns:
        label_count: dizionario con conteggio per label
        image_counter: numero totale di immagini salvate
    """
    
    # Remove existing directory if it exists and create new one
    if os.path.exists(dataset_save_dir):
        shutil.rmtree(dataset_save_dir)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # Get dataset from the existing pool with proper transform
    # Il problema è che ImagePool restituisce PIL images, serve transform per convertire in tensor
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = data_pool.get_dataset(transform=transform_to_tensor)
    pool_size = len(dataset)
    
    print(f"[LOG] ImagePool contains {pool_size} images from training")
    
    if pool_size == 0:
        raise ValueError("ImagePool è vuoto! Non posso generare il dataset.")
    
    # Set teacher to evaluation mode
    teacher.eval()
    to_pil = transforms.ToPILImage()
    label_count = defaultdict(int)
    image_counter = 0
    
    # Create dataloader for batch processing (much faster than one by one)
    batch_size = min(32, args.train_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle per varietà
        num_workers=0,  # Riduci a 0 per evitare problemi con worker processes
        pin_memory=True
    )
    
    generation_bar = tqdm(total=total_images_to_generate, desc="Using Cached Images")
    
    with torch.no_grad():
        for batch_images in loader:
            if image_counter >= total_images_to_generate:
                break
            
            # Move to GPU
            batch_images = batch_images.to(args.gpu)
            
            # Get pseudo labels from teacher (stesso processo di generate_samples)
            t_logit, _ = teacher(batch_images)
            pseudo_labels = torch.argmax(t_logit, dim=1)
            
            # Save each image in the batch (stesso formato di generate_samples)
            current_batch_size = min(batch_images.size(0), total_images_to_generate - image_counter)
            for i in range(current_batch_size):
                # Convert tensor to PIL Image (stesso preprocessing)
                img_tensor = torch.clamp(batch_images[i].cpu(), 0, 1)
                pil_image = to_pil(img_tensor)
                
                # Create filename with counter and label (stesso naming)
                label = pseudo_labels[i].cpu().item()
                label_count[label] += 1
                filename = f"{image_counter:06d}_{label}.png"
                filepath = os.path.join(dataset_save_dir, filename)
                
                # Save image
                pil_image.save(filepath)
                image_counter += 1
                generation_bar.update(1)
                
                if image_counter >= total_images_to_generate:
                    break
    
    generation_bar.close()
    
    # Se il pool non ha abbastanza immagini, avvisa
    if image_counter < total_images_to_generate:
        print(f"[WARNING] Pool conteneva solo {image_counter} immagini, richieste {total_images_to_generate}")
        print(f"[INFO] Considerate di aumentare args.keep_last o ridurre args.data_clear")
    
    return label_count, image_counter

def old_generate_samples_deepinv(args, teacher, dataset_save_dir, total_images_to_generate):
    """
    Generate synthetic samples using Deep Inversion (without generator)
    """
    # Remove existing directory if it exists and create new one
    normalizer = Normalizer(args.dataset)
    
    if os.path.exists(dataset_save_dir):
        shutil.rmtree(dataset_save_dir)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    image_counter = 0
    to_pil = transforms.ToPILImage()
    
    # Set teacher to evaluation mode
    teacher.eval()
    
    # Image dimensions (adjust based on your dataset)
    img_size = (3, 32, 32)  # For CIFAR-10, change as needed
    
    # Generation parameters (same as used in training)
    iterations = args.g_steps  # Number of optimization steps per image
    lr_g = args.lr_G
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    bn_weight = args.bn
    oh_weight = args.oh
    tv_weight = args.tv if hasattr(args, 'tv') else 0.0
    
    # Setup hooks for batch normalization statistics
    hooks = []
    for m in teacher.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            from generative.DEEPINV.hooks import DeepInversionHook
            hooks.append(DeepInversionHook(m, 0))
    
    generation_bar = tqdm(range(total_images_to_generate), desc="Generating Dataset")
    label_count = defaultdict(int)
    
    for step in generation_bar:
        # Initialize random image that requires gradients
        inputs = torch.randn(size=[1, *img_size], device=args.gpu).requires_grad_()
        
        # Random target class
        target = torch.randint(low=0, high=args.num_classes, size=(1,)).to(args.gpu)
        
        # Optimizer for the input image
        optimizer = torch.optim.Adam([inputs], lr_g, betas=[beta_1, beta_2])
        
        best_cost = 1e6
        best_input = inputs.data.clone()
        
        # Optimize the input image
        for it in range(iterations):
            # Apply augmentations
            inputs_aug = jitter_and_flip(inputs)
            
            # Forward pass through teacher
            t_out, _ = teacher(inputs_aug)
            
            # Compute losses
            loss_bn = sum([h.r_feature for h in hooks])  # BN feature matching loss
            loss_oh = F.cross_entropy(t_out, target)      # One-hot loss
            loss_tv = get_image_prior_losses(inputs)      # Total variation loss
            
            # Total loss
            loss_G = bn_weight * loss_bn + oh_weight * loss_oh + tv_weight * loss_tv
            
            # Keep track of best result
            if best_cost > loss_G.item():
                best_cost = loss_G.item()
                best_input = inputs.data.clone()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()
            
            # Clip images to valid range
            inputs.data = clip_images(inputs.data, normalizer.mean, normalizer.std)
        
        # Use the best generated image
        with torch.no_grad():
            # Get final prediction from teacher
            final_out, _ = teacher(best_input)
            pseudo_label = torch.argmax(final_out, dim=1).cpu().item()
            
            # Denormalize if needed
            if normalizer:
                best_input = normalizer(best_input, reverse=True)
            
            # Convert to PIL and save
            img_tensor = torch.clamp(best_input[0].cpu(), 0, 1)
            pil_image = to_pil(img_tensor)
            
            # Create filename
            label_count[pseudo_label] += 1
            filename = f"{image_counter:06d}_{pseudo_label}.png"
            filepath = os.path.join(dataset_save_dir, filename)
            
            # Save image
            pil_image.save(filepath)
            image_counter += 1
            
            generation_bar.set_postfix(
                label=pseudo_label,
                loss=f"{best_cost:.4f}"
            )
    
    return label_count, image_counter