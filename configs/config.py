@dataclass
class Args:
    """
    img_size: (int): Height and width of input image.
    patch_size (int): Height and width of each square patch.
    C_in (int): Number of input channels.
    d_model (int): Dimensionality of the model's input/output representations.
    num_heads (int): Number of attention heads for GQA. Must be divisible by `d_model`.
    query_groups (int): Number of groups to divide the attention heads into for GQA. Must be divisble by `num_heads`. Typically equal to `num_heads`.
    d_ffn (int): Dimensionality of the feed-forward network. Typically equal to `4 * d_model`.
    num_layers (int): Number of transformer encoder blocks. Each block has GQA, FFN, RMSNorm, Dropout, and residuals applied.
    dropout (float): Probability of model components being dropped out. Helps prevent overfitting.
    rope_base (float): Exponential base of inverse frequency. Typically set at 10000.0.
    rms_norm_eps (float): Very small floating point value to prevent numerical instability in RMSNorm.
    learning_rate (float): Small value that controls how much the weights change in response to the loss.
    epochs (int): Number of iterations the model will be trained for.
    batch_size (int): Number of examples that get processed together before the weights are updated.
    num_classes (int): Number of classes for the model to predict. Set to 1000 for ImageNet-1K.
    epsilon (float): Very small floating point value to prevent numerical stability in Adam optimizer.
    max_norm (float): Upper limit of the gradients norm to prevent exploding gradients.
    weight_decay (float): Regularization technique where a penalty is added to the loss function.
    betas (Tuple[float, float]): Coefficients used for computing running averages of the gradient and its square.
    warmup_epochs (int): Initial training epochs where the learning rate is small and increases.
    eta_min (float): Minimum learning rate for the Cosine Annealing scheduler.
    mixup_alpha (float): Value between 0 and 1 which determines how much 2 examples are mixed. Higher means more mixing, lower means less mixing.
    cutmix_alpha (float): Controls the size of the image patch to be cut and pasted from another image.
    label_smoothing (float): The smoothing factor used to soften target labels during training.
    random_erasing_prob (float): Randomly selects a rectangle region in an image and erases its pixels.
    color_jitter (float): Controls the amount of random variation in brightness, contrast, saturation, and hue.
    auto_augment (bool): If True, a data augmentation strategy will be applied with a learned or predefined sequence of image transformation.
    num_workers (int): Number of subprocesses to use for data loading. More workers can improve training speed. Typically set to number of CPU cores.
    pin_memory (bool): If True, the DataLoader will copy tensors into pinned (page-locked) memory before returning them.
    persistent_workers (bool): DataLoader’s worker processes will stay alive  between epochs, rather than shutting down after each epoch.
    grad_accum_steps (int): Number of steps to accumulate gradients before performing a backward pass and optimizer update.
    imagenet_mean (Tuple[float, float, float]): The mean pixel values for each image channel used for normalizing images in the ImageNet dataset.
    imagenet_std (Tuple[float, float, float]): The standard deviation values for each image channel used for normalizing images in the ImageNet dataset.
    training_root (str): Path to the ImageNet-1K training data.
    validation_root (str): Path to the ImageNet-1K validation data.
    """
    # Model parameters
    img_size: int = 384
    patch_size: int = 16
    C_in: int = 3
    d_model: int = 1440
    num_heads: int = 24
    query_groups: int = 12
    d_ffn: int = 5760
    num_layers: int = 20
    dropout: float = 0.2
    rope_base: float = 30000.0 # Larger base since 384x384 resolution
    rms_norm_eps: float = 1e-7
    # Training parameters
    learning_rate: float = 2e-4
    epochs: int = 300
    batch_size: int = 256
    num_classes: int = 1000
    epsilon: float = 1e-6
    max_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_epochs: int = 50
    eta_min: float = 6e-7
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    random_erasing_prob: float = 0.4
    color_jitter: float = 0.4
    auto_augment: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_accum_steps: int = 4
    # ImageNet parameters
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    training_root: str = "/data/imagenet1k/train" # TODO: Update to correct patch
    validation_root: str = "/data/imagenet1k/val" # TODO: Update to correct patch
