import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


EXPERIMENTS = {
    "exp1_resnet50_pretrained_headonly": {
        "backbone": "resnet50",
        "pretrained": True,
        "freeze_backbone": True,
        "description": "ResNet-50 | Pretrained ImageNet | Head-Only Fine-Tuning",
    },
    "exp2_resnet50_pretrained_fulltune": {
        "backbone": "resnet50",
        "pretrained": True,
        "freeze_backbone": False,
        "description": "ResNet-50 | Pretrained ImageNet | Full Fine-Tuning",
    },
    "exp3_efficientnet_pretrained_headonly": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "freeze_backbone": True,
        "description": "EfficientNet-B0 | Pretrained ImageNet | Head-Only Fine-Tuning",
    },
    "exp4_resnet50_scratch_fulltrain": {
        "backbone": "resnet50",
        "pretrained": False,
        "freeze_backbone": False,
        "description": "ResNet-50 | No Pretrain (From Scratch) | Full Training",
    },
}

SEED        = 42
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-4
IMG_SIZE    = 224
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
NUM_WORKERS = 4


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(backbone, pretrained, freeze_backbone, num_classes):
    if backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def get_optimizer(model, freeze_backbone, pretrained):
    if not freeze_backbone and pretrained:
        # Separate LR for backbone vs head
        if hasattr(model, 'fc'):
            head_params     = list(model.fc.parameters())
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
        else:
            head_params     = list(model.classifier.parameters())
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith('classifier')]
        return optim.AdamW([
            {'params': backbone_params, 'lr': LR_BACKBONE},
            {'params': head_params,     'lr': LR_HEAD},
        ], weight_decay=1e-4)

    trainable = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable, lr=LR_HEAD, weight_decay=1e-4)


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += imgs.size(0)
    if scheduler:
        scheduler.step()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.max(1)[1]
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    accuracy  = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / total, accuracy, precision, recall, f1


def save_learning_curve(history, exp_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(exp_name.replace('_', ' ').title(), fontsize=13)
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train', color='#E74C3C', lw=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='#3498DB', lw=2)
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train', color='#E74C3C', lw=2)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   color='#3498DB', lw=2)
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = save_dir / f"{exp_name}_curve.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def run_experiment(exp_name, cfg, data_dir, output_dir):
    print(f"\n{'='*60}\n  {exp_name}\n  {cfg['description']}\n{'='*60}")

    set_seed(SEED)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))
    num_classes  = len(full_dataset.classes)
    n            = len(full_dataset)
    n_test       = int(n * TEST_SPLIT)
    n_val        = int(n * VAL_SPLIT)
    n_train      = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_ds.dataset  = datasets.ImageFolder(data_dir, transform=get_transforms(train=False))
    test_ds.dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Device: {device} | Classes: {num_classes} | Train: {n_train}  Val: {n_val}  Test: {n_test}")

    model     = build_model(cfg['backbone'], cfg['pretrained'], cfg['freeze_backbone'], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model, cfg['freeze_backbone'], cfg['pretrained'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_acc                        = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        v_loss, v_acc, v_prec, v_rec, v_f1  = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), exp_dir / 'best_model.pth')

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train {t_loss:.4f}/{t_acc:.4f} | "
                  f"Val {v_loss:.4f}/{v_acc:.4f} | "
                  f"P:{v_prec:.4f} R:{v_rec:.4f} F1:{v_f1:.4f}")

    elapsed = time.time() - start_time

    model.load_state_dict(torch.load(exp_dir / 'best_model.pth', map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test | Acc:{test_acc:.4f}  Prec:{test_prec:.4f}  Rec:{test_rec:.4f}  F1:{test_f1:.4f}  ({elapsed:.1f}s)")

    save_learning_curve(history, exp_name, exp_dir)

    results = {
        "exp_name":           exp_name,
        "description":        cfg['description'],
        "backbone":           cfg['backbone'],
        "pretrained":         cfg['pretrained'],
        "freeze_backbone":    cfg['freeze_backbone'],
        "num_classes":        num_classes,
        "best_val_acc":       best_val_acc,
        "test_accuracy":      test_acc,
        "test_precision":     test_prec,
        "test_recall":        test_rec,
        "test_f1":            test_f1,
        "training_time_sec":  elapsed,
        "history":            history,
    }
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(output_dir / 'classes.json', 'w') as f:
        json.dump(full_dataset.classes, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--exp',        type=str, default='all')
    parser.add_argument('--epochs',     type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    global NUM_EPOCHS
    NUM_EPOCHS = args.epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exp_name, cfg in EXPERIMENTS.items():
        if args.exp != 'all' and not exp_name.startswith(args.exp):
            continue
        all_results[exp_name] = run_experiment(exp_name, cfg, args.data_dir, output_dir)

    if len(all_results) > 1:
        print(f"\n{'='*80}\n  SUMMARY\n{'='*80}")
        print(f"  {'Experiment':<45} {'Acc':>8} {'Recall':>8} {'F1':>8}")
        print(f"  {'-'*72}")
        for name, r in all_results.items():
            print(f"  {name:<45} {r['test_accuracy']:>8.4f} {r['test_recall']:>8.4f} {r['test_f1']:>8.4f}")

    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
