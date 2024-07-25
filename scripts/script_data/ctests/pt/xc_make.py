import os
import xcquinox as xce

cwd = os.getcwd()

rseeds = [92017, 17920]
levels = ['gga', 'mgga']
constrs = ['c', 'nc']
for lidx, lev in enumerate(levels):
    for sidx, seed in enumerate(rseeds):
        for cidx, constr in enumerate(constrs):
            xdir = f'x_3_16_{constr}{sidx}_{lev}'
            cdir = f'c_3_16_{constr}{sidx}_{lev}'
            xcdir = f'xc_3_16_{constr}{sidx}_{lev}'

            xc = xce.xc.make_xcfunc(level = lev.upper(),
                                    x_net_path = os.path.join(cwd, xdir),
                                    c_net_path = os.path.join(cwd, cdir),
                                    savepath = os.path.join(cwd, xcdir), xdsfile='', cdsfile='')
