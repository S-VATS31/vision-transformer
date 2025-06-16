class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Custom scheduler combining linear warmup with cosine annealing.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        warmup_epochs (int): Number of epochs to linearly increase the learning rate.
        total_epochs (int): Total number of training epochs.
        eta_min (float): Minimum learning rate multiplier (used as a fraction of the base LR).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min

        def lr_lambda(epoch) -> float:
            if epoch < warmup_epochs:
                # Linear warmup from 0 to 1
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay from 1 to eta_min
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return eta_min + (1 - eta_min) * cosine_decay

        super().__init__(optimizer, lr_lambda)

class ImageNetTrainer:
    def __init__(self, model_save_dir: str = "checkpoints"):
        """
        Initialize Image Net Trainer.

        Args:
            model_save_dir (str): Directory to where the model will be saved.
        """
        # Create model save directory
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)

        # Initialize model
        self.model = VisionTransformer(Args).to(device)

        # Setup data loaders
        self.train_loader, self.val_loader = self.setup_data_loaders()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Args.learning_rate,
            weight_decay=Args.weight_decay,
            betas=Args.betas,
            eps=Args.epsilon
        )

        # Warmup + Cosine annealing scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=Args.warmup_epochs,
            total_epochs=Args.epochs,
            eta_min=Args.eta_min
        )

        # Gradient scaling to prevent numerical instability in fp16
        self.scaler = torch.amp.GradScaler(device=device.type) if torch.cuda.is_available() else None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir="runs")

        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []

    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Setup up training and validation data loaders.

        Returns:
            train_loader (DataLoader): Data loader containing all training examples.
            val_loader (DataLoader): Data loader containing all validation examples.
        """
        # Training transforms with data augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(Args.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=Args.color_jitter,
                contrast=Args.color_jitter,
                saturation=Args.color_jitter,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=15),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET) if Args.auto_augment else transforms.Lambda(lambda x: x), # Return original if no auto_augment
            transforms.ToTensor(),
            transforms.Normalize(mean=Args.imagenet_mean, std=Args.imagenet_std),
            transforms.RandomErasing(p=Args.random_erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        # Validation transforms
        val_transform = transforms.Compose([
            transforms.Resize(int(Args.img_size * 1.14)),
            transforms.CenterCrop(Args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=Args.imagenet_mean, std=Args.imagenet_std)
        ])

        # Get training dataset
        try:
            train_dataset = datasets.ImageFolder(
                root=Args.training_root,
                transform=train_transform
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"ImageNet training dataset ({Args.training_root}) cannot be found.") from None
        except RuntimeError as e:
            raise RuntimeError(f"Error occured when trying to open ImageNet training dataset: {e}.") from None

        # Get validation dataset
        try:
            val_dataset = datasets.ImageFolder(
                root=Args.validation_root,
                transform=val_transform
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"ImageNet validation dataset ({Args.validation_root}) cannot be found.") from None
        except RuntimeError as e:
            raise RuntimeError(f"Error occured when trying to open ImageNet validation dataset: {e}.") from None

        # Train loader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=Args.batch_size,
            shuffle=True,
            num_workers=Args.num_workers,
            pin_memory=Args.pin_memory,
            persistent_workers=Args.persistent_workers
        )

        # Validation loader
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=Args.batch_size,
            shuffle=False,
            num_workers=Args.num_workers,
            pin_memory=Args.pin_memory,
            persistent_workers=Args.persistent_workers
        )

        # Dataset statistics
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")

        return train_loader, val_loader

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train Vision Transformer for a single epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            epoch_loss (float): Average loss over epoch.
                Calculated by taking accumulated loss / number of training samples.
            epoch_acc (float): Accuracy over epoch.
                Calculated by taking (correct preds / total preds) * 100.0.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Args.epochs}")

        # Training loop
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(device), targets.to(device) # Ensure both on same device

            # Apply randomized augmentation
            images, targets_a, targets_b, lam = random_augmentation(images, targets, Args.mixup_alpha)

            # Zero gradients at the start of accumulation
            if batch_idx % Args.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            # Apply AMP if cuda is available
            if self.scaler is not None:
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    outputs = self.model(images) # Forward pass

                    # Calculate weighted loss
                    loss = (
                        lam * F.cross_entropy(outputs, targets_a, label_smoothing=Args.label_smoothing) +
                        (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=Args.label_smoothing)
                    )
                    loss = loss / Args.grad_accum_steps # Scale loss for gradient accumulation

                # Accumulate loss
                total_loss += loss.item() * Args.grad_accum_steps
                self.scaler.scale(loss).backward() # Backpropagate loss

                # Update weights every gradient accumulation steps or on final batch even if grad_accum_steps not reached
                if (batch_idx + 1) % Args.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Unscale gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Args.max_norm) # Clip gradients L2 Norm
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Cuda not available
                outputs = self.model(images) # Forward pass

                # Calculate weighted loss
                loss = (
                    lam * F.cross_entropy(outputs, targets_a, label_smoothing=Args.label_smoothing) +
                    (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=Args.label_smoothing)
                )
                loss = loss / Args.grad_accum_steps # Scale loss for gradient accumulation

                # Accumulate loss
                total_loss += loss.item() * Args.grad_accum_steps
                loss.backward() # Backpropagate loss

                # Update weights every gradient accumulation steps or on final batch even if grad_accum_steps not reached
                if (batch_idx + 1) % Args.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Args.max_norm) # Clip gradients L2 Norm

                    # Update weights
                    self.optimizer.step()

            # Calculate stats for non-augmented batches
            if lam == 1.0: 
                # Get predicted class
                predicted = torch.argmax(outputs, dim=1) # shape: [B, num_classes], argmax(num_classes)

                # Get total examples in batch
                total += targets.size(0)

                # Count how many predicted classes matches target class
                correct += predicted.eq(targets_a).sum().item()

            # Update progress bar
            if batch_idx % 100 == 0:
                acc = 100. * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{acc:.2f}%',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

        # Calculate avg training loss/accuracy
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """
        Test Vision Transformer on validation data.

        Returns:
            val_acc (float): Validation accuracy.
                Calculated by taking (correct preds / total preds) * 100.0.
            avg_val_loss (float): Validation loss.
                Calculated by taking validation loss / number of validation samples.
        """
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        # Turn off gradient calculation for validation set
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            # Validation loop
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device) # Ensure images, target on same device

                outputs = self.model(images) # Forward pass

                # Calculate loss, no augmentation for validation
                loss = F.cross_entropy(outputs, targets, label_smoothing=Args.label_smoothing)

                # Statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                acc = 100. * correct / total
                avg_loss = val_loss / (total / targets.size(0))
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})

        # Calculate avg validation loss/accuracy
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(self.val_loader)

        return avg_val_loss, val_acc

    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint.

        epoch (int): Current epoch saved during checkpoint.
        val_acc (float): Validation accuracy.
            Calculated by taking (correct preds / total preds) * 100.0.
        is_best (bool): Inidicates if current model checkpoint is the best.
            Indicated by higher validation accuracy.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'args': Args,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.model_save_dir / 'latest_checkpoint.pth')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.model_save_dir / 'best_checkpoint.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model and optimizer from checkpoint file.

        Args:
            checkpoint_path (str): Path to checkpoint file.

        Returns:
            start_epoch (int): Epoch to resume from.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint['val_acc']
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return start_epoch
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint path ({checkpoint_path}) cannot be found.") from None
        except RuntimeError as e:
            raise RuntimeError(f"Error occured when trying to open checkpoint path: {e}.") from None

    def train(self, resume_from: Optional[str] = None, patience: int = 3):
        """
        Main training loop.

        Args:
            resume_from (Optional[str]): Path to checkpoint to resume training from.
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        start_epoch = 0
        early_stop_counter = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)

        # Log parameters
        print(f"Starting training from epoch {start_epoch}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad()):,}")

        # Main training loop
        for epoch in range(start_epoch, Args.epochs):
            print(f"\nEpoch {epoch + 1}/{Args.epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            # Update lr
            self.scheduler.step()

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            self.save_checkpoint(epoch, val_acc, is_best)

            # Early stopping
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        self.writer.close()
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        self.plot_metrics()

    def plot_metrics(self):
        """
        Plot training loss and validation accuracy over epochs.
        """
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Create trainer
    trainer = ImageNetTrainer()

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
