#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:06:15 2017

@author: Stuart Prescott, Isaac Gresham
"""

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
import time
import getpass


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


def writeMPI(objective_name, nwalkers, ntemps, nsteps, nthin, init_method = 'prior', buffering=1000):
    """
    """
    
    code = "\
import sys\n\
from refnx.analysis import CurveFitter\n\
from emcee.utils import MPIPool\n\
from %s import *\n\
objective = setup('%s.dat')\n\
with MPIPool() as pool:\n\
    if not pool.is_master():\n\
        pool.wait()\n\
        sys.exit(0)\n\
\n\
    nwalkers=%d\n\
    ntemps=%d\n\
    nsteps=%d\n\
    nthin=%d\n\
\n\
    fitter_dry = CurveFitter(objective, nwalkers=nwalkers, ntemps=ntemps)\n\
\n\
    fitter_dry.initialise('%s')\n\
\n\
    with open('%s' + '_samplechain_' +\n\
              str(nwalkers) + 'walkers_' +\n\
              str(ntemps) + 'temps_' + \n\
              str(nsteps) + 'steps_' + \n\
              str(nthin) + 'thinned.txt',\n\
              'a', buffering=%d) as fh:\n\
\n\
        fitter_dry.sample(nsteps, nthin=nthin, pool=pool, verbose=True, f=fh)\
"%(objective_name, objective_name, nwalkers, ntemps, nsteps, nthin, init_method, objective_name, buffering)
        
    filename = objective_name + "_run.py"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(code)


def writeShell(name, timeh):
    script = "\
#!/bin/bash\n\
#PBS -P rr87\n\
#PBS -q normal\n\
#PBS -l walltime=%d:00:00\n\
#PBS -l mem=2GB\n\
#PBS -l jobfs=2GB\n\
#PBS -l ncpus=8\n\
#PBS -l software=refnx\n\
#PBS -l wd\n\
\n\
source home/561/ig8882/venv/refnx-activate\n\
\n\
mpirun -np 8 python %s_run.py"%(timeh, name)

    filename = name + ".sh"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(script)