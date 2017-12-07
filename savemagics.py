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

        filename = line
        
        with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
            fh.write(created)
            fh.write(user)
            fh.write(cell)
            
        
        self.shell.run_cell(cell) # Run the contents of the cell


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
objective = setup()\n\
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
"%(objective_name, nwalkers, ntemps, nsteps, nthin, init_method, objective_name, buffering)
        
    filename = objective_name + "_run.py"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(code)
