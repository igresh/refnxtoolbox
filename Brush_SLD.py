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
            self.dry_sld = SLD(dry_sld)

        if isinstance(dry_filler_sld, SLD):
            self.dry_filler_sld = dry_filler_sld
        else:
            self.dry_filler_sld = SLD(dry_filler_sld)

        if isinstance(adsorbed_amount, Parameter):
            self.adorbed_amount = adsorbed_amount
        else:
            self.adorbed_amount = possibly_create_parameter(adsorbed_amount,
                                                            name='adsorbed amount')

        if isinstance(dry_thickness, Parameter):
            self.dry_thickness = dry_thickness
        else:
            self.dry_thickness = possibly_create_parameter(dry_thickness,
                                                            name='adsorbed amount')

        #self.real = Parameter(value, name='%s - sld' % name)
        self.imag = Parameter(0, name='%s - isld' % name)
        # TODO impliment imaginary SLD

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.real, self.imag, self.dry_sld,self.dry_filler_sld,
                                 self.adorbed_amount, self.dry_thickness])

    def __call__(self, thick=0, rough=0):
        return Slab(thick, self, rough, name=self.name)

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other

    @property
    def real(self):
        return (self.dry_thickness*(self.dry_sld.real.value-self.dry_filler_sld.real.value)
                +self.adorbed_amount.value*self.dry_filler_sld.real.value)/self.adorbed_amount.value

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters
        # p = Parameters(name=self.name)
        # p.extend([self.real, self.imag])
        # return p