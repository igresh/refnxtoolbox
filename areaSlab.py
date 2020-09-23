# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:32:40 2018

@author: igres
"""
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx.reflect import SLD, Component, Structure
import warnings

import numpy as np

class ConstrainedAmountModel(Component):
    """
    Component that scales the length of a structure to ensure that adsorbed
    amount of a component of interest (COI) is constrained.
    
    If no pure_SLD is provided the system is treated as a 2-component system, 
    made up of the COI and the system solvent (which is supplied to the final
    structure). The solvent volume fractions (and thicknesses) of the provided
    slabs are used to calculate the adsorbed amount. No questions
    are asked if slab SLDs differ.
    
    If a pure_SLD is provided the system is treated as a 3-component system,
    made up of the COI, the system solvent and a 'filler' substance, which is
    assumed to be air (SLD 0). The solvent volume fractions are used as the
    solvent fractions, and the slab SLDs and pure SLDs are used to calculate
    the filler-COI volume fractions. The overall COI volume fraction is used
    to calculate (and constrain) the adsorbed amount.
    
    If both a pure_SLD and filler_SLD is provided the system is treated as a
    3-component system, made up of the COI, the system solvent and a 'filler' 
    substance. The solvent volume fractions are used as the solvent fractions,
    and the slab SLDs and pure SLDs are used to calculate the filler-COI volume
    fractions. The overall COI volume fraction is used to calculate (and
    constrain) the adsorbed amount.
    
    Parameters:
        pure_thick (float) - Hypothetical thickness of a pure layer of the
        compound of interest.
        
        structure (refnx.reflect.structure.Structure) - structure of slabs, the
        adsorbed amount of which will be constrained by scaling its thickness.
        
        name (str) - optional identifier for the component.
        
        pure_sld (SLD, float, None) - SLD of the pure compound of interest.
        
        pure_sld (SLD, float, None) - SLD of the filler (non solvent) compound.
    """
    def __init__(self, pure_thick, structure, name='', pure_sld=None, filler_sld=None):
        super(ConstrainedAmountModel, self).__init__()
        
        self.name = name

        self.pure_thick = possibly_create_parameter(pure_thick,
                                               name='%s - dry thickness' % self.name)
        
        self.structure = structure
        
        self.pure_sld = pure_sld
        
        if isinstance(pure_sld, SLD):
            self.pure_sld = pure_sld
        elif pure_sld != None:
            self.pure_sld = SLD(pure_sld)
          
        if isinstance(filler_sld, SLD):
            self.filler_sld = filler_sld
        elif filler_sld != None:
            self.filler_sld = SLD(filler_sld)
        elif pure_sld != None:
            print ('warning: Filler SLD assumed to be 0')
            self.filler_sld = SLD(0)


    def profile(self, reverse=False, end_roughness=0):
        """
        Calculates the volume fraction profile

        Returns
        -------
        z, vfp : np.ndarray
            Distance from the interface, volume fraction profile
        """
        
        # Just create a SLD structure that varies between 0 and 1. Bit of a
        # hack, but works fine.
        s = Structure()
        s |= SLD(0)(0)

        m = SLD(1.)

        for i, slab in enumerate(self.slabs()):
            layer = m(slab[0], slab[3])
            if self.pure_sld == None: #Two component system
                layer.vfsolv.value = slab[4]
            else: #3+ component system
                # Get COI:Filler ratio from SLD
                unsolvated_nonfill_fraction =\
                (slab[1] - self.filler_sld.real.value)/ (self.pure_sld.real.value - self.filler_sld.real.value)
                
                # Multiply COI:filler fraction by COI+Filler:Solvent Fraction
                layer.vfsolv.value = 1-((1-slab[4])*unsolvated_nonfill_fraction)

            s |= layer
            
        s |= SLD(0)(0,end_roughness)
        s.solvent = 0
        s.reverse_structure = reverse
        # now calculate the VFP.
        total_thickness = np.sum(self.slabs()[:, 0])
        buffer = total_thickness*0.1
        zed = np.linspace(-buffer, buffer+total_thickness, 1000)
        z, s = s.sld_profile(z=zed)
        
        return z, s


    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.pure_thick])
        p.extend([component.parameters for component in self.structure])
        return p
    
    def slabs(self, structure=None):
        """
        slab representation of this component. See :class:`Structure.slabs`
        """
        if self.pure_sld == None: # Two component System
            struct_purethick = np.sum(self.structure.slabs[:,0]*(
                    1-self.structure.slabs[:,4]))
        else: # 3+ Component system
            # Find COI:filler ratio
            # (obs_SLD - filler_SLD)/(pure_SLD - filler_SLD)
            unsolvated_nonfill_fraction = ((self.structure.slabs[:,1] - self.filler_sld.real.value)/
                                   (self.pure_sld.real.value - self.filler_sld.real.value))
            # Find theoretical thickness of a pure COI layer
            struct_purethick = np.sum(self.structure.slabs[:,0]*
                                      (1-self.structure.slabs[:,4])*
                                      unsolvated_nonfill_fraction)
            
        # Will scale thickness such that theoretical thickness will match set
        # thickness.
        scale = self.pure_thick/float(struct_purethick)
        new_slabs = self.structure.slabs
        new_slabs[:,0] = new_slabs[:,0]*scale
        new_slabs[:,3] = new_slabs[:,3]*scale # Will also scale roughnesses
        return new_slabs


class area_slabVF(Component):
    """
    A slab component has uniform SLD over its thickness.

    Parameters
    ----------
    adsorbed_amount : refnx.analysis.Parameter or float
        thickness of unsolvated slab (Angstrom)
    sld : refnx.reflect.SLD, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2)
    rough : float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]

    """

    def __init__(self, adsorbed_amount, dry_sld, rough, name='', vfsolv=0):
        super().__init__()
        self.adsorbed_amount = possibly_create_parameter(adsorbed_amount,
                                               name='%s - dry thickness' % name)
        if isinstance(dry_sld, SLD):
            self.sld = dry_sld
        else:
            self.sld = SLD(dry_sld, name=name)
        self.rough = possibly_create_parameter(rough,
                                               name='%s - rough' % name)
        self.vfsolv = (
            possibly_create_parameter(vfsolv,
                                      name='%s - volfrac solvent' % name))
        self.name = name

        p = Parameters(name=self.name)
        p.extend([self.adsorbed_amount, self.sld.real, self.sld.imag,
                  self.rough, self.vfsolv])

        self._parameters = p

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        slab representation of this component. See :class:`Structure.slabs`
        """
        return np.atleast_2d(np.array([self.thick,
                                       self.sld.real.value,
                                       self.sld.imag.value,
                                       self.rough.value,
                                       self.vfsolv.value]))
    
    @property
    def thick(self):
        return self.adsorbed_amount.value/(1-self.vfsolv.value)
        
    def is_monotonic(self):
        return True
    
    
    def moment(self):
        return self.thick/2
    
    
    def profile(self, reverse=False):
        """
        returns the vfp for this    component.
        """
        m = SLD(1.)
        s = Structure()
        s |= SLD(0)
        
        slab = self.slabs()[0]
        thick = slab[0]
        rough = slab[3]
        vfsolv = slab[4]
        
        layer = m(thick, rough)
        layer.vfsolv.value = vfsolv
        s |= layer
        s |= SLD(0)
        s.solvent = SLD(0)

        if reverse:
            s.reverse_structure = True

        zed = np.linspace(0, thick*1.1, thick*1.1 + 1)
        
        zed[0] = 0.01
        z, s = s.sld_profile(z=zed)
        
        return z, s
    
    
class area_slabT(Component):
    """
    A slab component has uniform SLD over its thickness.

    Parameters
    ----------
    adsorbed_amount : refnx.analysis.Parameter or float
        thickness of unsolvated slab (Angstrom)
    sld : refnx.reflect.SLD, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2)
    rough : float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]

    """

    def __init__(self, adsorbed_amount, dry_sld, thick, rough, name=''):
        super().__init__()
        self.adsorbed_amount = possibly_create_parameter(adsorbed_amount,
                                               name='%s - dry thickness' % name)
        if isinstance(dry_sld, SLD):
            self.sld = dry_sld
        else:
            self.sld = SLD(dry_sld, name=name)
        self.rough = possibly_create_parameter(rough,
                                               name='%s - rough' % name)
        self.thick = (
            possibly_create_parameter(thick,
                                      name='%s - thick' % name))
        self.name = name

        p = Parameters(name=self.name)
        p.extend([self.adsorbed_amount, self.thick, self.sld.real, self.sld.imag,
                  self.rough ])

        self._parameters = p

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters
    
    def is_monotonic(self):
        return True
    
    def __repr__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return repr(self.parameters)

    def slabs(self, structure=None):
        """
        slab representation of this component. See :class:`Structure.slabs`
        """
        vfsolv = 1 - self.adsorbed_amount.value/self.thick.value
        if vfsolv < 0:
            vfsolv = 0
            warnings.warn('Layer thickness less than adsorbed amount. Clipping vfsolv to 0', RuntimeWarning)
        return np.atleast_2d(np.array([self.thick.value,
                                       self.sld.real.value,
                                       self.sld.imag.value,
                                       self.rough.value,
                                       vfsolv]))
    
    def moment(self):
        return self.thick.value/2
    

    def profile(self, reverse=False):
        """
        returns the vfp for this    component.
        """
        m = SLD(1.)
        s = Structure()
        s |= SLD(0)
        
        slab = self.slabs()[0]
        thick = slab[0]
        rough = slab[3]
        vfsolv = slab[4]
        
        layer = m(thick, rough)
        layer.vfsolv.value = vfsolv
        s |= layer
        s |= SLD(0)
        s.solvent = SLD(0)

        if reverse:
            s.reverse_structure = True

        zed = np.linspace(0, thick*1.1, thick*1.1 + 1)
        
        zed[0] = 0.01
        z, s = s.sld_profile(z=zed)
        
        return z, s

