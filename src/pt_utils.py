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
