from django.apps import AppConfig
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import numpy as np
from PIL import Image


class ImageProcessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'image_processing'

