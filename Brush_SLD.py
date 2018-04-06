# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:13:56 2018

@author: igres
"""
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx.reflect import SLD, Slab

class Brush_SLD(SLD):
    """
    Object representing freely varying SLD of a material

    Parameters
    ----------
    value : float or complex
        Scattering length density of a material.
        Units (10**-6 Angstrom**-2)
    name : str, optional
        Name of material.


    Notes
    -----
    An SLD object can be used to create a Slab:

    >>> # an SLD object representing Silicon Dioxide
    >>> sio2 = SLD(3.47, name='SiO2')
    >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
    >>> sio2_layer = SLD(20, 3)

    """
    def __init__(self, dry_sld, adsorbed_amount, dry_thickness, dry_filler_sld=0, name=''):
        self.name = name

        if isinstance(dry_sld, SLD):
            self.dry_sld = dry_sld
        else:
            self.dry_sld = SLD(dry_sld, name='measured dry')

        if isinstance(dry_filler_sld, SLD):
            self.dry_filler_sld = dry_filler_sld
        else:
            self.dry_filler_sld = SLD(dry_filler_sld, name='dry filler')

        if isinstance(adsorbed_amount, Parameter):
            self.adsorbed_amount = adsorbed_amount
        else:
            self.adsorbed_amount = possibly_create_parameter(adsorbed_amount,
                                                            name='adsorbed amount')

        if isinstance(dry_thickness, Parameter):
            self.dry_thickness = dry_thickness
        else:
            self.dry_thickness = possibly_create_parameter(dry_thickness,
                                                            name='dry thickness')

    def __call__(self, thick=0, rough=0):
        return Slab(thick, self, rough, name=self.name)

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other

    @property
    def real(self):
        v = (self.dry_thickness.value*(self.dry_sld.real.value-self.dry_filler_sld.real.value)
                +self.adsorbed_amount.value*self.dry_filler_sld.real.value)/self.adsorbed_amount.value
        return Parameter(name='%s - imag'%self.name, value=v)
    @property
    def imag(self):
        # Not implemented
        return Parameter(name='%s - imag'%self.name, value=0)
        #return (self.dry_thickness*(self.dry_sld.real.value-self.dry_filler_sld.real.value)
        #        +self.adsorbed_amount.value*self.dry_filler_sld.real.value)/self.adsorbed_amount.value

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters = Parameters(name=self.name)
        self._parameters.extend([self.dry_sld.real, self.dry_sld.imag,
                                 self.dry_filler_sld.real, self.dry_filler_sld.imag,
                                 self.adsorbed_amount, self.dry_thickness])
        return self._parameters