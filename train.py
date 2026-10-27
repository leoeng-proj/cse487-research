# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torchvision.utils import save_image, make_grid

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Logging Helper Functions                          #
#################################################################################

def save_checkpoint(model, ema, opt, checkpoint_dir, train_steps, logger, final=False):
    """Save a checkpoint on rank 0 and synchronize all ranks."""
    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args
    }
    if final:
        filename = f"final_{train_steps:07d}.pt"
    else:
        filename = f"{train_steps:07d}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def save_samples(out_samples, args, train_steps, logger, directory, out_labels=None, class_names=None, final=False):
    """Save the image on rank 0 with optional class labels."""
    sample_dir = os.path.join(directory, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    if final:
        filename = f"samples_final_{train_steps:07d}.png"
    else:
        filename = f"samples_{train_steps:07d}.png"
    sample_path = os.path.join(sample_dir, filename)

    num_samples = min(8, out_samples.size(0))
    samples_to_save = out_samples[:num_samples]

    if out_labels is not None:
        # Normalize images from [-1, 1] to [0, 1]
        norm_samples = (samples_to_save.clamp(-1, 1) + 1.0) / 2.0
        img_h = norm_samples.shape[-2]
        font_size = max(14, int(img_h * 0.045))

        # Robust font loading with cross-platform fallbacks
        font = None
        for font_name in ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]:
            try:
                font = ImageFont.truetype(font_name, size=font_size)
                break
            except (IOError, OSError):
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=font_size)
            except (TypeError, Exception):
                font = ImageFont.load_default()

        labeled_tensors = []
        for i in range(num_samples):
            img_np = (norm_samples[i].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)

            cls_id = int(out_labels[i].item()) if hasattr(out_labels[i], "item") else int(out_labels[i])
            if class_names is not None and 0 <= cls_id < len(class_names):
                name = str(class_names[cls_id])
                if name == str(cls_id):
                    label_text = f"Class {cls_id}"
                else:
                    label_text = f"{cls_id}: {name}"
            else:
                label_text = f"Class {cls_id}"

            margin = max(6, int(img_h * 0.02))
            pad = max(3, int(img_h * 0.008))

            try:
                bbox = draw.textbbox((margin, margin), label_text, font=font)
            except AttributeError:
                w, h = draw.textsize(label_text, font=font)
                bbox = (margin, margin, margin + w, margin + h)

            bg_box = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
            try:
                draw.rounded_rectangle(bg_box, radius=4, fill=(0, 0, 0))
            except AttributeError:
                draw.rectangle(bg_box, fill=(0, 0, 0))

            draw.text((margin, margin), label_text, fill=(255, 255, 255), font=font)

            labeled_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            labeled_tensors.append(labeled_t)

        stacked_labeled = torch.stack(labeled_tensors)
        grid = make_grid(stacked_labeled, nrow=4)
        save_image(grid, sample_path)
        logger.info(f"Saved samples with class labels to {sample_path}")
        if args.wandb:
            wandb_utils.log_image(stacked_labeled * 2.0 - 1.0, train_steps)
    else:
        save_image(
            samples_to_save.cpu(),
            sample_path,
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )
        logger.info(f"Saved samples to {sample_path}")
        if args.wandb:
            wandb_utils.log_image(out_samples, train_steps)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Class mapping: {dataset.class_to_idx}")
    logger.info(f"Number of classes: {len(dataset.classes)}")
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # BELOW IS THE NOT WORKING CFG VERSION
    # Labels to condition the model with (feel free to change):
    # ys = torch.randint(size=(local_batch_size,), device=device)
    # use_cfg = args.cfg_scale > 1.0
    # # Create sampling noise:
    # n = ys.size(0)
    # zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # # Setup classifier-free guidance:
    # if use_cfg:
    #     zs = torch.cat([zs, zs], 0)
    #     y_null = torch.tensor([args.num_classes] * n, device=device)
    #     ys = torch.cat([ys, y_null], 0)
    #     sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
    #     model_fn = ema.forward_with_cfg
    # else:
    #     sample_model_kwargs = dict(y=ys)
    #     model_fn = ema.forward

    # BELOW IS THE WORKING NO CFG VERSION
    # Labels for the single-class rail dataset
    # ys = torch.zeros(local_batch_size, dtype=torch.long, device=device)
    # # Sampling noise
    # zs = torch.randn(
    #     local_batch_size, 4, latent_size, latent_size, device=device
    # )
    # # Temporarily disable CFG to test basic sampling
    # use_cfg = False
    # sample_model_kwargs = dict(y=ys)
    # model_fn = ema.forward

    # EXPERIMENT SPACE
    # Labels for the single-class rail dataset
    # ys = torch.zeros(local_batch_size, dtype=torch.long, device=device)
    # Labels for multi-class rail dataset (using rank offset so across DDP ranks all classes are sampled)
    sample_ys = (rank * local_batch_size + torch.arange(local_batch_size, device=device)) % args.num_classes
    ys = sample_ys.clone()
    
    # Create sampling noise
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Enable classifier-free guidance
    use_cfg = args.cfg_scale > 1.0

    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.full((n,), args.num_classes, dtype=torch.long, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    save_checkpoint(model=model, ema=ema, opt=opt, checkpoint_dir=checkpoint_dir, train_steps=train_steps, logger=logger)
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    # Remove the null-condition half when using CFG
                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    # Decode latent samples into images
                    samples = vae.decode(samples / 0.18215).sample
                    out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device,)
                    dist.all_gather_into_tensor(out_samples, samples)
                    out_labels = torch.zeros((args.global_batch_size,), dtype=torch.long, device=device)
                    dist.all_gather_into_tensor(out_labels, sample_ys)

                if rank == 0:    
                    save_samples(
                        out_samples=out_samples, args=args, train_steps=train_steps,
                        logger=logger, directory=experiment_dir,
                        out_labels=out_labels, class_names=dataset.classes
                    )
                dist.barrier()
                logger.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    with torch.no_grad():
        sample_fn = transport_sampler.sample_ode()
        samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
        # Remove the null-condition half when using CFG
        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)
        # Decode latent samples into images
        samples = vae.decode(samples / 0.18215).sample
        out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device,)
        dist.all_gather_into_tensor(out_samples, samples)
        out_labels = torch.zeros((args.global_batch_size,), dtype=torch.long, device=device)
        dist.all_gather_into_tensor(out_labels, sample_ys)

    if rank == 0: 
        logger.info("Training finished. Saving final checkpoint...")
        save_checkpoint(model=model, ema=ema, opt=opt, checkpoint_dir=checkpoint_dir, train_steps=train_steps, logger=logger, final=True)

        logger.info("Generating final samples...")
        save_samples(
            out_samples=out_samples,
            args=args, train_steps=train_steps,
            logger=logger, directory=experiment_dir,
            out_labels=out_labels, class_names=dataset.classes,
            final=True
        )
    dist.barrier()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
