def mixup_data(images: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup augmentation to training data as a form of regularization.

    Args:
        images (torch.Tensor): Input images of shape [B, C, H, W].
        targets (torch.Tensor): Target labels of shape [B].
        alpha (float): Mixup hyperparameter controlling the strength of mixing.

    Returns:
        mixed_images (torch.Tensor): Mixed images.
        targets_a (torch.Tensor): Original targets.
        targets_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.
    """
    # No mixup applied, return original images
    if alpha == 0:
        return images, targets, targets, 1.0

    # Sample lambda from Beta distribution
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    batch_size = images.size(0)

    # Permuted numbers from 0 to B-1
    index = torch.randperm(batch_size).to(device)

    # Compute weighted sum
    mixed_images = lam * images + (1 - lam) * images[index]

    # Original labels, mixed labels
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam

def cutmix_data(images: torch.Tensor, targets: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply cutmix augmentation to training data as a form of regularization.

    Args:
        images (torch.Tensor): Input tensor of shape [B, C, H, W].
        targets (torch.Tensor): Target labels of shape [B].
        alpha (float): CutMix hyperparameter controlling the strength of mixing.

    Returns:
        mixed_images (torch.Tensor): Mixed images.
        targets_a (torch.Tensor): Original targets.
        targets_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.
    """
    # No CutMix applied, return original images
    if alpha == 0:
        return images, targets, targets, 1.0

    # Get height, width from images tensor
    B, _, H, W = images.shape
    if H != W:
        print(f"Height ({H}) and width ({W}) of the image must be equal for ViTs.")

    # Beta distribution (single sample) to get lambda
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    # Permuted numbers from 0 to B-1
    index = torch.randperm(B).to(device) # [B,]

    # Cutting ratios
    cut_ratio = math.sqrt(1.0 - lam)
    W_cut = int(cut_ratio * W)
    H_cut = int(cut_ratio * H)

    # Uniform center
    cx = torch.randint(W, (1,)).item() # x-coordinate (0, W-1)
    cy = torch.randint(H, (1,)).item() # y-coordinate (0, H-1)

    # Calculate box boundaries
    x1 = max(cx - W_cut // 2, 0)
    y1 = max(cy - H_cut // 2, 0)
    x2 = min(cx + W_cut // 2, W)
    y2 = min(cy + H_cut // 2, H)
    
    # Create mixed images
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Update lambda
    lam = 1 - ((x2 - x1) * (y2 -y1) / (W * H))

    # Get targets
    targets_a, targets_b = targets, targets[index]

    return mixed_images, targets_a, targets_b, lam

def random_augmentation(images: torch.Tensor, targets: torch.Tensor, alpha: float, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup/cutmix or no augmentation to training data as a form of regularization.

    Args:
        images (torch.Tensor): Input tensor of shape [B, C, H, W].
        targets (torch.Tensor): Target labels of shape [B].
        alpha (float): CutMix hyperparameter controlling the strength of mixing.

    Returns:
        mixed_images (torch.Tensor): Mixed images.
        targets_a (torch.Tensor): Original targets.
        targets_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.
    """
    choices = ["mixup", "cutmix", "none"]

    # Randomly choose data augmentation method
    method = choices[torch.randint(0, len(choices), (1,)).item()]
    if method == "mixup":
        return mixup_data(images, targets, alpha, device)
    elif method == "cutmix":
        return cutmix_data(images, targets, alpha, device)
    # No augmentation applied
    else:
        return images, targets, targets, 1.0
