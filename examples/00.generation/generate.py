import os
import xcquinox as xce

XCQUINOX_EXAMPLE_DIR = '/home/awills/Documents/Research/xcquinox/examples'

#random seeds for initial weight creation
rseeds = [92017, 17920]

#directories corresponding to the 'level' of the network
dirs = ['mgga', 'nl']

topdir = '00.generation'

DEPTH = 3
NHIDDEN = 16

for didx, direc in enumerate(dirs):
    for sidx, seed in enumerate(rseeds):
        xsubdir = f'x_{DEPTH}_{NHIDDEN}_{sidx}_{direc}'
        csubdir = f'c_{DEPTH}_{NHIDDEN}_{sidx}_{direc}'
        print(xsubdir, csubdir)
        xfp = os.path.join(XCQUINOX_EXAMPLE_DIR, topdir, direc, xsubdir)
        cfp = os.path.join(XCQUINOX_EXAMPLE_DIR, topdir, direc, csubdir)
        x, xc = xce.net.make_net('x', level=direc, depth=DEPTH, nhidden=NHIDDEN, random_seed = seed, savepath = xfp)
        c, cc = xce.net.make_net('c', level=direc, depth=DEPTH, nhidden=NHIDDEN, random_seed = seed, savepath = cfp)
