import os
import xcquinox as xce

XCQUINOX_EXAMPLE_DIR = '/home/awills/Documents/Research/xcquinox/examples'

#random seeds for initial weight creation
rseeds = [92017, 17920]

#directories corresponding to the 'level' of the network
dirs = ['mgga', 'nl']

gentopdir = '00.generation'
pttopdir = '01.pretraining'
opttopdir = '02.optimization'

DEPTH = 3
NHIDDEN = 16

for didx, direc in enumerate(dirs):
    for sidx, seed in enumerate(rseeds):
        xsubdir = f'x_{DEPTH}_{NHIDDEN}_{sidx}_{direc}'
        csubdir = f'c_{DEPTH}_{NHIDDEN}_{sidx}_{direc}'
        xcsubdir = f'xc_{DEPTH}_{NHIDDEN}_{sidx}_{direc}'
        print(xsubdir, csubdir)
        # xfp = os.path.join(randir, xsubdir)
        # cfp = os.path.join(randir, csubdir)
        xc = xce.xc.make_xcfunc(level = direc.upper(),
                                x_net_path = os.path.join(XCQUINOX_EXAMPLE_DIR, pttopdir, direc, xsubdir),
                                c_net_path = os.path.join(XCQUINOX_EXAMPLE_DIR, pttopdir, direc, csubdir),
                                savepath = os.path.join(XCQUINOX_EXAMPLE_DIR, opttopdir, direc, xcsubdir),
                                xdsfile = '', cdsfile = '')