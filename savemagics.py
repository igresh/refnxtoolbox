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
import re


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

        if '.' in line:
            print ('WARNING: illegal character (.) in name.')
        elif '-' in line:
            print ('WARNING: illegal character (-) in name.')           
            
        filename = line + ".py"
        
        name_func = "\n\
def name():\n\
    return '%s'\n\
\n\
    "%line
        
        with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
            fh.write(created)
            fh.write(user)
            fh.write(cell)
            fh.write(name_func)
            
        
        self.shell.run_cell(cell) # Run the contents of the cell
        self.shell.run_cell(name_func)

        print("\n---\n[ Wrote cell contents to '%s' ]" % filename)

    
get_ipython().register_magics(CellWriter)

def package(name, nwalkers, ntemps, nsamps, nthin, nCPUs=2, useMPI=True,
            vfp_location='/mnt/1D9D9A242359B87C/Git Repos/refnx/examples/\
analytical_profiles/brushes/brush.py'):
    
    direc = name + '/'    
    objective_file = name + '.py'
    
    if nCPUs == 1:
        if useMPI:
            print ('You have only requested 1 CPU: Setting useMPI to false')
        useMPI = False
    
    # Make the directory for the objective, this will be copied to the cluster
    if not os.path.exists(name):
        os.makedirs(name)
        
    # Copy brush file into local directory
    vfp_name = os.path.basename(vfp_location)
    shutil.copyfile(vfp_location, direc + vfp_name)


    # Open the objective file
    file = open(objective_file, 'r')
    s = file.read()
    
    match = re.findall(r'RD.*\((.*?)\)\n',s) # Find all the instances where a
                                             # Data file is opened
    new_s = s
    for datafile in match:
        datafile = datafile[1:-1] # drop the quotes from around the string
        # Get the filename (drop the path)
        fn = os.path.basename(datafile)
        
        # Copy data file into local directory
        shutil.copyfile(datafile, direc + fn)
        
        # Replace old file path with new local file path
        new_s = new_s.replace(datafile, fn)
    
    os.chdir(direc) # Not an ideal way to do things...
    # Write over the objective file
    with open(objective_file, 'w') as fh:
        fh.write(new_s)
    
    # Open and excecute new setup file
    with open(objective_file, 'r') as fh:
        print(objective_file)
        s = fh.read()
        exec(s, globals())

    objective = setup() 
    os.chdir('..') # Not an ideal way to do things...

    nparameters = len(objective.varying_parameters())


    writeRun(name, direc, nwalkers, ntemps, nsamps, nthin, use_MPI=useMPI)
    walltime = approx_walltime(nwalkers, ntemps, nsamps, nthin, nparameters, nCPUs)
    writeShell(name, direc, walltime, nCPUs)


def approx_walltime(nwalkers, ntemps, nsamps, nthin, nparameters, nCPUs,
                    time_per_calc = 0.005, margin=1.5):
    # Time approximation is based on estimated scaling - not actual testing...
    calcs_per_CPU = nparameters**0.8 * (nwalkers * ntemps)**0.8 * (nsamps * nthin) / nCPUs**0.8
    time_per_calc = 0.005
    time = 5 + margin * calcs_per_CPU * time_per_calc / 60
    time_hour = time/60
    time_min  = time%60
    if time_hour < 48: # Right time for Raijin 
        return "%02d:%02d:00" % (time_hour, time_min)
    else: # To long for Raijin
        print ('WARNING: Setting runtime to 47:55 (max). Your run may be cut short')
        return "%02d:%02d:00" % (47, 55)


def writeRun(objective_name, direc, nwalkers, ntemps, nsamps, nthin, use_MPI=True, 
             init_method = 'prior', buffering=100):
    """
    """

    code1 = "\
import sys\n\
import os\n\
import glob\n\
import os\n\
import pickle\n\
from refnx.analysis import CurveFitter, load_chain\n\
from %s import *\n\
\
dir_path = os.path.dirname(os.path.realpath(__file__))\n\
os.chdir(dir_path)\
\n\
objective = setup()\n"%(objective_name)

#If using MPI
    code2a = "\n\
from schwimmbad import MPIPool\n\
with MPIPool() as pool:\n\
    if not pool.is_master():\n\
        pool.wait()\n\
        sys.exit(0)\n\
\n\
    nwalkers=%d\n\
    ntemps=%d\n\
    nthin=%d\n\
    nsamps=%d\n\
\n\
    filename = ('%s' + '_samplechain_' + \n\
                str(nwalkers) + 'walkers_' + \n\
                str(ntemps) + 'temps_' + \n\
                str(nthin) + 'thinned.pkl')\n\
\n\
    maybe_existing_files = glob.glob(filename)\
\n\
    if len(maybe_existing_files) > 0:\n\
        existing_file = maybe_existing_files[0]\n\
        fitter = pickle.load(open(existing_file, 'rb'))\n\
        print ('resuming from chain: ' , existing_file)\n\
    else:\n\
        fitter = CurveFitter(objective, nwalkers=nwalkers, ntemps=ntemps)\n\
        fitter.initialise('%s')\n\
        print ('Created new fitter' , filename)\n\
\n\
\n\
    for i in range(nsamps):\n\
        fitter.sample(1, nthin=nthin, pool=pool)\n\
        print(\"%%d/%%d\"%%(i+1,nsamps))\n\
        pickle.dump(fitter, open(filename, 'wb'))\
"%(nwalkers, ntemps, nthin, nsamps, objective_name, init_method)

#If not using MPI (1 CPU)
    code2b = "\
nwalkers=%d\n\
ntemps=%d\n\
nthin=%d\n\
nsamps=%d\n\
\n\
filename = ('%s' + '_samplechain_' + \n\
            str(nwalkers) + 'walkers_' + \n\
            str(ntemps) + 'temps_' + \n\
            str(nthin) + 'thinned.pkl')\n\
\n\
maybe_existing_files = glob.glob(filename)\
\n\
if len(maybe_existing_files) > 0:\n\
    existing_file = maybe_existing_files[0]\n\
    fitter = pickle.load(open(existing_file, 'rb'))\n\
    print ('resuming from chain: ' , existing_file)\n\
else:\n\
    fitter = CurveFitter(objective, nwalkers=nwalkers, ntemps=ntemps)\n\
    fitter.initialise('%s')\n\
    print ('Created new fitter' , filename)\n\
\n\
\n\
for i in range(nsamps):\n\
    print(\"%%d/%%d\"%%(i,nsamps))\n\
    fitter.sample(1, nthin=nthin, pool=1)\n\
    pickle.dump(fitter, open(filename, 'wb'))\
"%(nwalkers, ntemps, nthin, nsamps, objective_name, init_method)
        
    filename = direc + objective_name + "_run.py"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(code1)
        if use_MPI:
            fh.write(code2a)
        else:
            fh.write(code2b)

def writeShell(name, direc, timeh, nCPUs):
    script1 = "\
#!/bin/bash\n\
#PBS -P rr87\n\
#PBS -q normal\n\
#PBS -l walltime=%s\n\
#PBS -l mem=2GB\n\
#PBS -l jobfs=1GB\n\
#PBS -l ncpus=%d\n\
#PBS -l software=refnx\n\
#PBS -l wd\n\
\n\
source /home/561/ig8882/venv/refnx-activate\n\
\n\
"%(timeh, nCPUs)

    script2a = "mpirun -np %d python %s_run.py"%(nCPUs, name)
    script2b = "python %s_run.py"%(name)

    filename = direc + name + ".sh"
        
    with open(filename, 'w') as fh: # Save the cell + metadata as a .py file
        fh.write(script1)
        if nCPUs == 1:
            fh.write(script2b)
        else:
            fh.write(script2a)