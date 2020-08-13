import torch
import torchvision.models as models
from src.pt_utils import torch_device
from src.utils import print_out


def load_model():
    """Load the VGG19 as suggested in the paper arXiv:1508.06576"""
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters (layers) b/c we don't need
    # to train any layers of the VGG model
    for param in vgg.parameters():
        param.requires_grad_(False)

    # compatible to GPU/CPU by torch_device helper function
    vgg = vgg.to(torch_device())
    return vgg


def gen_features(image, model, layers, info_str=''):
    """Use the feature space provided by the 19-layer VGG network.
        To extract feature maps for each image, either content or style.
    """
    names = {
        'conv1_1': '0',
        'conv1_2': '2',
        'conv2_1': '5',
        'conv2_2': '7',
        'conv3_1': '10',
        'conv3_2': '12',
        'conv3_3': '14',
        'conv3_4': '16',
        'conv4_1': '19',
        'conv4_2': '21',
        'conv4_3': '23',
        'conv4_4': '25',
        'conv5_1': '28',
        'conv5_2': '30',
        'conv5_3': '32',
        'conv5_4': '34'
    }

    layers = {names[layer]: layer for layer in layers}

    if isinstance(info_str, str) and len(info_str) > 0:
        print_out(info_str, params=layers)

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gramian_matrix(T):
    """Calculate the Gramian Matrix of a tensor T
        Check the Gram Matrices article from http://mlwiki.org/index.php/Gram_Matrices
    """
    # As you may know the our tensor built based on this structure
    # `batch_size x channel x height x width` which is not suitable to our
    # Gram matrix procedure, we need to reshape it into `c x h * w` in order to
    # calculate the result!
    _, c, h, w = T.size()
    T = T.view(c, h * w)

    gram = torch.mm(T, T.t())
    return gram
