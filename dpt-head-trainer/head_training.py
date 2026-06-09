import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# Adicionar parent directory ao path para importar modelo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.dataset import SemiDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion


def get_logger(save_path):
    """Cria logger para rastrear treinamento"""
    os.makedirs(save_path, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler(os.path.join(save_path, 'train.log'))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def build_model(cfg, num_classes):
    """Constrói modelo DPT com cabeça"""
    model = DPT(
        backbone='dinov2',
        features=cfg['model']['features'],
        # backbone_features=cfg['model']['out_channels'][-1],
        readout_type='ignore',
        channels_last=False,
        use_bn=cfg['model']['use_bn'],
        num_classes=num_classes
    )
    return model


def build_datasets(cfg, class_group=None):
    """Constrói train e val datasets"""
    dataset_root = cfg['dataset']['root_dir']
    
    train_dataset = SemiDataset(
        name=cfg['dataset']['name'],
        root=dataset_root,
        split='train',
        size=512
    )
    
    val_dataset = SemiDataset(
        name=cfg['dataset']['name'],
        root=dataset_root,
        split='val',
        size=512
    )
    
    return train_dataset, val_dataset


def get_class_mapping(cfg, class_group=None):
    """
    Cria mapeamento de classes:
    - Se class_group é especificado, mapeia apenas aquelas classes para [1,2,3...]
    - Classe 0 sempre é 'background'
    """
    dataset_name = cfg['dataset']['name']
    all_classes = CLASSES[dataset_name]
    
    if class_group is None:
        # Usar todas as classes
        class_mapping = {i: i for i in range(len(all_classes))}
        num_classes = len(all_classes)
    else:
        # Mapear apenas classes do grupo
        group_classes = cfg['class_groups'][class_group]
        
        # background sempre é classe 0
        class_mapping = {0: 0}
        
        # Mapear classes do grupo para 1, 2, 3, ...
        new_idx = 1
        for orig_idx, cls_name in enumerate(all_classes):
            if cls_name in group_classes:
                class_mapping[orig_idx] = new_idx
                new_idx += 1
        
        # Adicionar "unknown" para classes fora do grupo
        unknown_idx = new_idx
        for orig_idx in range(len(all_classes)):
            if orig_idx not in class_mapping:
                class_mapping[orig_idx] = 0  # Tratar como background
        
        num_classes = new_idx
    
    return class_mapping, num_classes


def train_epoch(model, train_loader, criterion, optimizer, epoch, logger, device):
    """Treina por uma epoch"""
    model.train()
    loss_meter = AverageMeter()
    
    for i, batch in enumerate(train_loader):
        if len(batch) == 3:
            images, masks, ignore_masks = batch
        else:
            images, masks = batch
            ignore_masks = None
        
        images = images.to(device)
        masks = masks.to(device)
        if ignore_masks is not None:
            ignore_masks = ignore_masks.to(device)
        
        # Forward
        outputs = model(images)
        
        # Loss
        if ignore_masks is not None:
            loss = criterion(outputs, masks)
            loss = loss * (ignore_masks != 255).float()
            loss = loss.sum() / (ignore_masks != 255).sum().clamp(min=1)
        else:
            loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item())
        
        if (i + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}] Iter [{i+1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}")
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, logger, device, num_classes):
    """Valida no val split"""
    model.eval()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    
    for i, batch in enumerate(val_loader):
        images, masks = batch
        
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss_meter.update(loss.item())
        
        # Calcula metrics
        pred = outputs.argmax(dim=1).cpu().numpy()
        masks_np = masks.cpu().numpy()
        
        intersection, union = intersectionAndUnion(pred, masks_np, num_classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
    
    # Calcula IoU
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    mean_iou = iou.mean()
    
    logger.info(f"Epoch [{epoch+1}] Val Loss: {loss_meter.avg:.4f}, mIoU: {mean_iou:.4f}")
    logger.info(f"IoUs: {[f'{x:.4f}' for x in iou]}")
    
    return loss_meter.avg, mean_iou


def main():
    parser = argparse.ArgumentParser(description='DPT Head Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save checkpoints')
    parser.add_argument('--class-group', type=str, default=None, 
                       help='Class group to train (from config.yaml)')
    parser.add_argument('--pretrained', type=str, default=None, 
                       help='Path to pretrained backbone')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Logger
    logger = get_logger(args.save_path)
    logger.info(f"Config:\n{yaml.dump(cfg)}")
    logger.info(f"Device: {device}")
    
    # Class mapping
    class_mapping, num_classes = get_class_mapping(cfg, args.class_group)
    logger.info(f"Class group: {args.class_group}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Model
    model = build_model(cfg, num_classes)
    model = model.to(device)
    logger.info(f"Model parameters: {count_params(model)}")
    
    # Datasets and dataloaders
    train_dataset, val_dataset = build_datasets(cfg, args.class_group)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['train']['learning_rate'],
        weight_decay=cfg['train']['weight_decay']
    )
    
    total_iterations = len(train_loader) * cfg['train']['epochs']
    scheduler = PolynomialLR(optimizer, total_iters=total_iterations, power=0.9)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # TensorBoard
    writer = SummaryWriter(args.save_path)
    
    # Resume if provided
    start_epoch = 0
    best_miou = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['best_miou']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg['train']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, logger, device)
        scheduler.step()
        
        # Validate
        val_loss, miou = validate(model, val_loader, criterion, epoch, logger, device, num_classes)
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mIoU/val', miou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': max(best_miou, miou)
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        
        # Save best
        if miou > best_miou:
            best_miou = miou
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            logger.info(f"Best mIoU updated: {best_miou:.4f}")
    
    logger.info("Training finished!")
    writer.close()


if __name__ == '__main__':
    main()
