from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, size=504, split='train', class_mapping=None):
        self.name = name
        self.root = root
        self.size = size
        self.split = split
        self.class_mapping = class_mapping

        # Loading splits path
        id_path = os.path.join(root, 'splits', f'{split}.txt')
        # print(f"o path que não está encontrando é: {id_path}")

        if not os.path.exists(id_path):
            raise FileNotFoundError(f"File {id_path} not found")
        
        self.image_mask_pairs = []

        with open(id_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 2:
                    img_path, mask_path = parts
                    self.image_mask_pairs.append((img_path, mask_path))
        
        print(f"Loaded: {len(self.image_mask_pairs)} samples for split: '{split}'")
        
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_rel_path, mask_rel_path = self.image_mask_pairs[idx]    

        img_path = os.path.join(self.root, img_rel_path)
        mask_path = os.path.join(self.root, mask_rel_path)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(mask_path)))

        if self.split == 'val':
            # Redimensiona para tamanho fixo
            if self.size:
                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)
            img, mask = normalize(img, mask)
            img = torch.from_numpy(np.array(img)).float()
            mask = torch.from_numpy(np.array(mask)).long()
            
            # Aplicar mapeamento de classes se fornecido
            if self.class_mapping is not None:
                mask_mapped = torch.zeros_like(mask)
                for orig_class, new_class in self.class_mapping.items():
                    mask_mapped[mask == orig_class] = new_class
                mask = mask_mapped
            
            return img, mask
        
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        

        # Normaliza (até aqui está como PIL Image)
        img = normalize(img)
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Aplicar mapeamento de classes se fornecido
        if self.class_mapping is not None:
            mask_mapped = torch.zeros_like(mask)
            for orig_class, new_class in self.class_mapping.items():
                mask_mapped[mask == orig_class] = new_class
            mask = mask_mapped
        
        # Criar ignore_mask como tensor
        ignore_mask = torch.zeros_like(mask)
        ignore_mask[mask == 254] = 255

        return img, mask, ignore_mask