from src.utils import flatten

# as mentioned in paper the layers which used for feature representations are like below
# as probably you may notice the VGG network by Torch consists of a dict of layer states
# which specifies based on the number of layers and we just add names to these numbers
CONTENT_FEATURE_LAYERS = {
    'a': ['conv1_2'],
    'b': ['conv2_2'],
    'c': ['conv3_2'],
    'd': ['conv4_2'],
    'e': ['conv5_2'],
}
STYLE_FEATURE_LAYERS = {
    'a': ['conv1_1'],
    'b': ['conv1_1', 'conv2_1'],
    'c': ['conv1_1', 'conv2_1', 'conv3_1'],
    'd': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'],
    'e': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
}
TARGET_FEATURE_LAYERS = sorted(
    list(set(flatten(list(STYLE_FEATURE_LAYERS.values())) + flatten(list(CONTENT_FEATURE_LAYERS.values())))))
