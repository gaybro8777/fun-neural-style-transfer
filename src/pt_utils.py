import json
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def image_scale(image, resize=None, max_size=None):
    """Scale the image based on resize or maximum size ratio"""

    if isinstance(max_size, int) and not isinstance(resize, tuple):
        scale = max_size / max(image.size)
        w, h = image.size
        resize = (round(w * scale), round(h * scale))
        image = image.resize(resize)

    elif not isinstance(max_size, int) and isinstance(resize, tuple):
        image = image.resize(resize)

    return image


def load_image(image_path, resize=None, max_size=None):
    """Load and resize an image."""

    image = Image.open(image_path).convert('RGB')
    image = image_scale(image, resize, max_size)

    return image


def load_image_to_tensor(image_path, resize=None, max_size=None):
    """Load and prepare image for PyTorch"""

    # load image path to Pillow object
    image = load_image(image_path, resize=None, max_size=None)

    # transform object to tensor matched with pytorch structure
    # normalization
    # WxHxC to CxHxW
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )  # normalization
    ])

    # expand dimensions from 3 to 4
    # CxHxW to 1xCxHxW
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def load_image_from_tensor(tensor, pillow_able=True, resize=None, max_size=None):
    """Transform tensor image to array one"""

    # detach the pytorch tensor
    image = tensor.to('cpu').clone().detach()

    # convert detached tensor to numpy and remove single-dim
    image = image.numpy().squeeze()

    # transform the structure from CxHxW to WxHxC
    image = image.transpose(1, 2, 0)

    # denormalization
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    # limit the value between 0 and 1
    image = image.clip(0, 1)

    # convert image numpy to Pillow object
    if pillow_able:
        # transform from 0-1 to 0-255
        image = Image.fromarray(np.uint8(image * 255.0))

        # resize the image if it sets
        image = image_scale(image, resize, max_size)

    return image


def torch_device():
    """Specifies the GPU/CPU status of the resource"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def plot_result(p, a, x, figsize):
    """Plot the content, style, and the target side by side!"""

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1.imshow(load_image_from_tensor(p))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.imshow(load_image_from_tensor(a))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3.imshow(load_image_from_tensor(x))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    plt.show()


def cb_every_step(base_save_dir):
    """Callback sample for each step of transferring."""

    if not isinstance(base_save_dir, str) or not len(base_save_dir) > 0:
        def every_step(step, content, style, target, losses):
            pass

        return every_step

    base_save_dir = str(base_save_dir)
    os.makedirs(base_save_dir, exist_ok=True)

    def every_step(step, content, style, target, losses):
        save_dir = os.path.join(base_save_dir, str(step))
        os.makedirs(save_dir, exist_ok=True)

        target_path = os.path.join(save_dir, 'target.jpg')
        target = load_image_from_tensor(target, pillow_able=True)
        target.save(target_path, "JPEG")

    return every_step


def cb_final_step(base_save_dir):
    """Callback sample for final step"""
    if not isinstance(base_save_dir, str) or not len(base_save_dir) > 0:
        def final_step(content, style, target, history):
            pass

        return final_step

    base_save_dir = str(base_save_dir)
    os.makedirs(base_save_dir, exist_ok=True)

    def final_step(content, style, target, history):
        content_path = os.path.join(base_save_dir, 'content.jpg')
        style_path = os.path.join(base_save_dir, 'style.jpg')
        target_path = os.path.join(base_save_dir, 'target.jpg')
        history_path = os.path.join(base_save_dir, 'history.json')

        content = load_image_from_tensor(content, pillow_able=True)
        content.save(content_path, "JPEG")

        style = load_image_from_tensor(style, pillow_able=True)
        style.save(style_path, "JPEG")

        target = load_image_from_tensor(target, pillow_able=True)
        target.save(target_path, "JPEG")

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    return final_step
