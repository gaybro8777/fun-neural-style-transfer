import collections
from datetime import datetime
import logging
import os
from tqdm import tqdm
import torch
from src.pt_config import CONTENT_FEATURE_LAYERS
from src.pt_config import STYLE_FEATURE_LAYERS
from src.pt_config import TARGET_FEATURE_LAYERS
from src.pt_models import load_model
from src.pt_models import gen_features
from src.pt_models import gramian_matrix
from src.pt_utils import torch_device
from src.pt_utils import plot_result
from src.pt_utils import load_image_to_tensor
from src.pt_utils import load_image_from_tensor
from src.utils import print_out
from src.utils import set_logger

logger = set_logger(logging, __name__)


def transfer(content_path,
             style_path,
             steps=2000,
             log_every_step=200,
             learning_rate=1e-3,
             content_type='d',
             style_type='e',
             style_weights=None,
             alpha=8,
             beta=1e4,
             save_dir=None):
    """ Image Style Transfer Using Convolutional Neural Networks

    Args:
        content_path (str): The image path as a content (picture)!
        style_path (str): The image path as a style (artwork)!
        steps (int): Required number of steps that is necessary to reduce target loss.
        log_every_step (int): Get feedback every %n step!
        learning_rate (float): The learning rate regarding backward step!
        content_type (str): The type of combination of feature layers which mention in original paper!
        style_type (str): The type of combination of feature layers which mention in original paper!
        style_weights (list): The weights for feature layers in style phase, the original paper set as 1/5 value!
        alpha (float): The impact weight of content.
        beta (float): The impact weight of style.
        save_dir (str): The directory for saving the result of model regarding content, style, and the target!

    Returns:
        A history information of losses during transferring.
    """
    vgg = load_model()

    # content image as p
    content = load_image_to_tensor(content_path).to(torch_device())

    # style image as a
    style = load_image_to_tensor(style_path).to(torch_device())

    # the content image p is passed through the network and the content representation P_l (P)
    content_features = gen_features(
        content,
        vgg,
        layers=CONTENT_FEATURE_LAYERS[content_type],
        info_str='Content [p->P] features for layers:')

    # the style image a is passed through the network and its style representation A_l (A)
    style_features = gen_features(
        style,
        vgg,
        layers=STYLE_FEATURE_LAYERS[style_type],
        info_str='Style [a->A] features for layers:')

    # calculate the Gram Matrices for each layer of style representation (A)
    style_gram_matrices = {layer: gramian_matrix(style_features[layer]) for layer in style_features}

    # the last piece is a random white noise image x as our target
    # which is going to pass through the network.
    # for simplicity and make the process quickly, initialize it with the content image
    target = content.clone().requires_grad_(True).to(torch_device())

    style_weights_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    if not isinstance(style_weights, list):
        style_weights = {
            'conv1_1': 1 / 5,
            'conv2_1': 1 / 5,
            'conv3_1': 1 / 5,
            'conv4_1': 1 / 5,
            'conv5_1': 1 / 5,
        }
    else:
        style_weights = {weight_name: weight_value for weight_name, weight_value in
                         zip(style_weights_names, style_weights)}

    print_out(
        'The style weights for each layers:',
        logger=logger.info,
        params=style_weights)

    print_out(
        'The ratio α/β, for best result can be either [1e-3, 8e-4, 5e-3, 5e-4]',
        logger=logger.info,
        params={'alpha': alpha, 'beta': beta, 'alpha/beta': alpha / beta})

    # optimizer
    optimizer = torch.optim.Adam([target], lr=learning_rate)

    # history
    history = collections.defaultdict(list)

    for step in tqdm(range(1, steps + 1), position=0):
        t0 = datetime.now()

        # generate target feature maps
        target_features = gen_features(target, vgg, layers=TARGET_FEATURE_LAYERS)
        target_gram_matrices = {layer: gramian_matrix(target_features[layer]) for layer in target_features}

        # content loss
        loss_content = torch.mean(
            (target_features[CONTENT_FEATURE_LAYERS[content_type][0]] -
             content_features[CONTENT_FEATURE_LAYERS[content_type][0]]) ** 2)

        # style loss
        loss_style = 0
        for layer in STYLE_FEATURE_LAYERS[style_type]:
            # retrieve shape of target feature for each layer
            _, c, h, w = target_features[layer].shape

            # style loss of each layer
            loss_layer_style = style_weights[layer] * torch.mean(
                (target_gram_matrices[layer] - style_gram_matrices[layer]) ** 2)
            loss_style += loss_layer_style / (c * h * w)

        # calculate the total loss: α * loss_content + β * loss_style
        loss = alpha * loss_content + beta * loss_style

        # update the target
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # display results every n steps
        if step % log_every_step == 0:
            dt = datetime.now() - t0
            print()
            print(f'Step {step}/{steps}, '
                  f'Total Loss: {loss.item():.4}, '
                  f'Content Loss: {loss_content.item():.4f}, '
                  f'Style Loss: {loss_style.item():.4}, '
                  f'Duration: {dt}')

            # put the content, style and target images into one figure
            plot_result(content, style, target, figsize=(10, 4))

    plot_result(content, style, target, figsize=(20, 8))

    if isinstance(save_dir, str) and len(save_dir) > 0:
        print_out(
            'Final Result',
            logger=logger.info,
            params={
                'save_dir': save_dir,
                'content_path': os.path.join(save_dir, 'content.jpg'),
                'style_path': os.path.join(save_dir, 'style.jpg'),
                'target_path': os.path.join(save_dir, 'target.jpg'),
            })
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        content_im = load_image_from_tensor(content, pillow_able=True)
        style_im = load_image_from_tensor(style, pillow_able=True)
        target_im = load_image_from_tensor(target, pillow_able=True)

        content_im.save(os.path.join(save_dir, 'content.jpg'))
        style_im.save(os.path.join(save_dir, 'style.jpg'))
        target_im.save(os.path.join(save_dir, 'target.jpg'))

    return history
