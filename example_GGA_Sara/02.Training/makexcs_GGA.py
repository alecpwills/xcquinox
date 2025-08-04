import argparse
from typing import Union
import json
from xcquinox.net import load_xcquinox_model
from xcquinox.xc import RXCModel_GGA
import equinox as eqx


def save_eqx_XC_model_all(model, path: str = '', fixing: Union[str, None] = None,
                          tail_info: Union[str, None] = None):
    if fixing is None:
        fixing = ''
    else:
        fixing = f'_{fixing}'
    if tail_info is None:
        tail_info = ''
    else:
        tail_info = f'_{tail_info}'
    model_x = model.xnet
    save_name_x = f'{model_x.name}_d{model_x.depth}_n{model_x.nodes}_s{model_x.seed}\
{fixing}{tail_info}'

    needen_info_x = {'depth': model_x.depth, 'nodes': model_x.nodes,
                     'seed': model_x.seed, 'name': model_x.name}
    eqx.tree_serialise_leaves(f'{path}/{save_name_x}.eqx', model_x)
    with open(f"{path}/{save_name_x}.json", "w") as f:
        json.dump(needen_info_x, f)
    model_c = model.cnet
    save_name_c = f'{model_c.name}_d{model_c.depth}_n{model_c.nodes}_s{model_c.seed}\
{fixing}{tail_info}'
    needen_info_c = {'depth': model_c.depth, 'nodes': model_c.nodes,
                     'seed': model_c.seed, 'name': model_c.name}
    eqx.tree_serialise_leaves(f'{path}/{save_name_c}.eqx', model_c)
    with open(f"{path}/{save_name_c}.json", "w") as f:
        json.dump(needen_info_c, f)
    needen_info = {'xnet': save_name_x, 'cnet': save_name_c}
    save_name_xc = f'model_xc{fixing}{tail_info}'
    with open(f"{path}/{save_name_xc}.json", "w") as f:
        json.dump(needen_info, f)


def load_eqx_XC_model(model_xc_path):

    with open(f"{model_xc_path}.json", "r") as f:
        metadata = json.load(f)
    path = model_xc_path.split('/')
    path = '/'.join(path[:-1])
    model_x = load_xcquinox_model(f'{path}/{metadata["xnet"]}')
    model_c = load_xcquinox_model(f'{path}/{metadata["cnet"]}')
    model = RXCModel_GGA(model_x, model_c)
    return model


if __name__ == '__main__':
    """
    This scripts created the XC model for GGA functionals and stores it in an specific path.
    """
    # parse script arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outpath', action='store', default=None,
                        help='Path to save the model files to.')
    parser.add_argument('--load_xnet_path', action='store',
                        help='Path to .eqx checkpoint file of the desired exchange network to use.', default=None)
    parser.add_argument('--load_cnet_path', action='store',
                        help='Path to .eqx checkpoint file of the desired exchange network to use.', default=None)
    parser.add_argument('--fixing', action='store', default='',
                        help='Fixing string to append to the model name.')
    parser.add_argument('--tail_info', action='store', default='',
                        help='Tail information to append to the model name.')

    args = parser.parse_args()
    if args.load_xnet_path:
        xnet = load_xcquinox_model(args.load_xnet_path)
    else:
        raise ValueError("Please provide a path to the exchange network model.")
    if args.load_cnet_path:
        cnet = load_xcquinox_model(args.load_cnet_path)
    else:
        raise ValueError("Please provide a path to the correlation network model.")
    model = RXCModel_GGA(xnet, cnet)

    if args.outpath is None:
        pathx = '/'.join(args.load_xnet_path.split('/')[:-1])
        pathc = '/'.join(args.load_cnet_path.split('/')[:-1])
        if pathx == pathc:
            outpath = pathx
        else:
            outpath = '.'
            print("Warning: No output path provided, saving to current directory.")
            print("This may lead to errors when loading models later.")
    else:
        outpath = args.outpath
    print("Output path:", outpath)
    needen_info = {'xnet': args.load_xnet_path, 'cnet': args.load_cnet_path}
    save_name_xc = f'model_xc{args.fixing}{args.tail_info}'
    # We directly just dump the paths, because its not trained!
    with open(f"{outpath}/{save_name_xc}.json", "w") as f:
        json.dump(needen_info, f)
