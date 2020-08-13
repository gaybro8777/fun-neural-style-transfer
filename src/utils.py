import json


def flatten(l):
    """Flat list out of list of lists"""
    return [item for sublist in l for item in sublist]


def print_out(info_str, hashed=20, sep='#', add_to_begin=True, add_to_end=False, params=None):
    if params.__len__() != 0:
        if add_to_begin:
            print(info_str, sep * hashed)
        else:
            print(info_str)

        print(json.dumps(params, indent=2))

        if add_to_end:
            print(sep * (hashed + len(info_str) + 1))
