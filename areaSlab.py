# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:32:40 2018

@author: igres
"""
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx.reflect import SLD, Component
import numpy as np

class area_Slab(Component):
    """
    A slab component has uniform SLD over its thickness.

    Parameters
    ----------
    dry_thickness : refnx.analysis.Parameter or float
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

    def __init__(self, dry_thickness, dry_sld, rough, name='', vfsolv=0):
        super(area_Slab, self).__init__()
        self.dry_thickness = possibly_create_parameter(dry_thickness,
                                               name='%s - dry thickness' % name)
        if isinstance(dry_sld, SLD):
            self.sld = dry_sld
        else:
            self.sld = SLD(dry_sld)
        self.rough = possibly_create_parameter(rough,
                                               name='%s - rough' % name)
        self.vfsolv = (
            possibly_create_parameter(vfsolv,
                                      name='%s - volfrac solvent' % name))
        self.name = name

        p = Parameters(name=self.name)
        p.extend([self.dry_thickness, self.sld.real, self.sld.imag,
                  self.rough, self.vfsolv])

        self._parameters = p

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    @property
    def slabs(self):
        """
        slab representation of this component. See :class:`Structure.slabs`
        """
        thick = self.dry_thickness/(1-self.vfsolv.value)
        return np.atleast_2d(np.array([thick,
                                       self.sld.real.value,
                                       self.sld.imag.value,
                                       self.rough.value,
                                       self.vfsolv.value]))