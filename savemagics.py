#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:06:15 2017

@author: Stuart Prescott, Isaac Gresham
"""

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
import time
import getpass
import os
import shutil



@magics_class
class CellWriter(Magics):
    """ IPython Magics that write code to disk as well as execute it """

    @cell_magic
    def write(self, line, cell):
        """ write the cell to the specified filename
        
        Example
        =======
        
            %%write foo.py
            
            quux = 1
            goo = quux + 5
        
        """
        created = time.strftime("#Created: %H:%M %d-%m-%Y\n")
        user = "#Creator: %s\n"%getpass.getuser()

        filename = line + ".py"
        
        imports = "\
from brush import FreeformVFP, SoftTruncPDF\n\
from refnx.reflect import SLD, Slab, ReflectModel\n\
from refnx.dataset import ReflectDataset as RD\n"
        
        name_func = "\n\
def name():\n\
    return '%s'\n\
\n\
    "%line
        
        with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
            fh.write(created)
            fh.write(user)
            fh.write(imports)
            fh.write(cell)
            fh.write(name_func)
            
        
        self.shell.run_cell(cell) # Run the contents of the cell
        self.shell.run_cell(name_func)

        print("\n---\n[ Wrote cell contents to '%s' ]" % filename)

    
get_ipython().register_magics(CellWriter)

def package(name, nwalkers, ntemps, nsteps, nthin, nCPUs=8,
            vfp_location='/mnt/1D9D9A242359B87C/Git Repos/refnx/examples/\
analytical_profiles/brushes/brush.py'):
    
    direc = name + '/'    
    objective_file = name + '.py'
    data_file = name + '.dat'
    
    file = open(objective_file, 'r')
    exec(file.read(), globals())
    objective = setup(data_file)
        
    nparameters = len(objective.varying_parameters())
    print (nparameters)
    
    if not os.path.exists(name):
        os.makedirs(name)
    
    
    writeMPI(name, direc, nwalkers, ntemps, nsteps, nthin)
    
    walltime = approx_walltime(nwalkers, ntemps, nsteps, nparameters, nCPUs)
    writeShell(name, direc, walltime, nCPUs)
    
    os.rename(objective_file, direc + objective_file)
    shutil.copyfile(data_file, direc + data_file)
    shutil.copyfile(vfp_location, name + '/brush.py')
    


def approx_walltime(nwalkers, ntemps, nsteps, nparameters, nCPUs, time_per_calc = 0.005):
    calcs_per_CPU = nwalkers * ntemps * nsteps * nparameters / nCPUs
    time_per_calc = 0.005
    time_min = calcs_per_CPU * time_per_calc / 60
    return "%02d:%02d:00" % (time_min/60, time_min%60)
    
def writeMPI(objective_name, direc, nwalkers, ntemps, nsteps, nthin, init_method = 'prior', buffering=100):
    """
    """
    
    code = "\
import sys\n\
from refnx.analysis import CurveFitter\n\
from emcee import interruptible_pool\n\
from %s import *\n\
objective = setup('%s.dat')\n\
with interruptible_pool.InterruptiblePool() as pool:\n\
    #if not pool.is_master():\n\
    #    pool.wait()\n\
    #    sys.exit(0)\n\
\n\
    nwalkers=%d\n\
    ntemps=%d\n\
    nsteps=%d\n\
    nthin=%d\n\
\n\
    fitter = CurveFitter(objective, nwalkers=nwalkers, ntemps=ntemps)\n\
\n\
    fitter.initialise('%s')\n\
\n\
    with open('%s' + '_samplechain_' +\n\
              str(nwalkers) + 'walkers_' +\n\
              str(ntemps) + 'temps_' + \n\
              str(nsteps) + 'steps_' + \n\
              str(nthin) + 'thinned.txt',\n\
              'a', buffering=%d) as fh:\n\
\n\
        fitter.sample(nsteps, nthin=nthin, pool=pool, verbose=True, f=fh)\
"%(objective_name, objective_name, nwalkers, ntemps, nsteps, nthin, init_method, objective_name, buffering)
        
    filename = direc + objective_name + "_run.py"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(code)


def writeShell(name, direc, timeh, nCPUs):
    script = "\
#!/bin/bash\n\
#PBS -P rr87\n\
#PBS -q normal\n\
#PBS -l walltime=%s\n\
#PBS -l mem=2GB\n\
#PBS -l jobfs=2GB\n\
#PBS -l ncpus=%d\n\
#PBS -l software=refnx\n\
#PBS -l wd\n\
\n\
source home/561/ig8882/venv/refnx-activate\n\
\n\
mpirun -np 8 python %s_run.py"%(timeh, nCPUs, name)

    filename = direc + name + ".sh"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(script)