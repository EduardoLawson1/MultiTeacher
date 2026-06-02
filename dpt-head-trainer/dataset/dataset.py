import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from util.classes import CLASSES

class PascalVOCDataset(Dataset):
    """
    PASCAL VOC 2012 Dataset for segmentation.
    Reads data lists in the format:
    'JPEGImages/2011_003271.jpg SegmentationClass/2011_003271.png'
    """
    def __init__(self, root_dir, list_file, target_class_group, image_size=(352, 352)):
        self.root_dir = root_dir
        self.list_file = list_file
        self.target_class_group = target_class_group # List of class names to train on
        self.image_size = image_size

        # --- Read file list ---
        if not os.path.isfile(self.list_file):
            raise FileNotFoundError(f"List file not found: {self.list_file}")
        with open(self.list_file, 'r') as f:
            self.samples = [line.strip().split() for line in f.readlines()]

        # --- Define transformations ---
        self.transform_img = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),
        ])

        # --- Create class mapping ---
        self.all_pascal_classes = CLASSES['pascal']
        self.class_map = self._create_class_map()

    def _create_class_map(self):
        """
        Creates a mapping from original PASCAL class indices to the new group indices.
        Classes not in the target group are mapped to background (0).
        """
        # Start with background class
        group_classes_with_bg = ['background'] + self.target_class_group
        
        # Map original PASCAL index to new index
        class_map = torch.zeros(len(self.all_pascal_classes), dtype=torch.long)
        
        for i, class_name in enumerate(self.all_pascal_classes):
            if class_name in group_classes_with_bg:
                new_idx = group_classes_with_bg.index(class_name)
                class_map[i] = new_idx
            # else, it remains 0 (background)
            
        return class_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path_rel, mask_path_rel = self.samples[idx]
        
        img_path_abs = os.path.join(self.root_dir, img_path_rel)
        mask_path_abs = os.path.join(self.root_dir, mask_path_rel)

        # --- Load image and mask ---
        image = Image.open(img_path_abs).convert('RGB')
        mask = Image.open(mask_path_abs)

        # --- Apply transformations ---
        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))

        # --- Remap mask labels ---
        # PASCAL VOC has a special value 255 for border/void areas.
        # We'll keep them as 255 to ignore them in the loss function later.
        valid_pixels = (mask != 255)
        remapped_mask = torch.zeros_like(mask)
        remapped_mask[valid_pixels] = self.class_map[mask[valid_pixels]]
        remapped_mask[~valid_pixels] = 255 # Restore void label

        return image, remapped_mask.long()

    def get_num_classes(self):
        # Number of classes is the size of the group + 1 for background
        return len(self.target_class_group) + 1