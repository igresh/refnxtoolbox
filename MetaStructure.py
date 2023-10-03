
import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip
from scipy.integrate import simps

from refnx.reflect import Structure, Component, SLD, Slab
from refnx.analysis import Parameters, Parameter, possibly_create_parameter

import warnings

class metaComponent (Component):
    def __init__ (self, ComponentOne, ComponentTwo, microslab_max_thickness=1,
                  left_roughness=0, name='MixedLayer', ComponentOneSLD=None, ComponentTwoSLD=None):
        self.ComponentOne = ComponentOne
        self.ComponentTwo = ComponentTwo
        self.ComponentOneSLD = ComponentOneSLD
        self.ComponentTwoSLD = ComponentTwoSLD

        self._interfaces = None

        self.microslab_max_thickness = microslab_max_thickness

        self.left_roughness = left_roughness

        self.name = name


    def __call__ (self, z=None):
        zed1, vfp1 = self.get_vfp(self.ComponentOne.slabs())
        zed2, vfp2 = self.get_vfp(self.ComponentTwo.slabs())

        zed, vfp1, vfp2 = self.establish_basis (zed1, vfp1, zed2, vfp2)

        if np.any(z):
            vfp1 = np.interp(z, zed, vfp1)
            vfp2 = np.interp(z, zed, vfp2)
        else:
            z = zed

        return z, vfp1, vfp2


    def combined_sld(self, z=None):
        """
        Really just for testing


        """
        z, vfp1, vfp2 = self(z)
        total_vfp = vfp1 + vfp2

        if np.any(total_vfp>1.001):
            print (f'your total vf is greater than one: {np.max(total_vfp)}')


        if self.ComponentOneSLD is None:
            sld1 = self.ComponentOne.sld.real.value
        else:
            sld1 = self.ComponentOneSLD.real.value

        if self.ComponentTwoSLD is None:
            sld2 = self.ComponentTwo.sld.real.value
        else:
            sld2 = self.ComponentTwoSLD.real.value

        norm_vfp1 = vfp1/total_vfp
        norm_vfp2 = vfp2/total_vfp

        sld12 = sld1*norm_vfp1 + sld2*norm_vfp2



        return z, sld12, total_vfp


    def slabs(self, structure=None):

        z_init, vfp1_init, vfp2_init = self()
        slab_extent = z_init[-1]

        num_slabs = np.ceil(float(slab_extent) / self.microslab_max_thickness)
        slab_thick = float(slab_extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give last slab a miniscule roughness so it doesn't get contracted
        slabs[-1:, 3] = 0.5



        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick

        z, comb_sld, total_vfp = self.combined_sld(z=dist)

        slabs[:, 1] = comb_sld
        slabs[:, 2] = 0
        slabs[:, 4] = 1-total_vfp

        slabs[0,3] =  self.left_roughness

        return slabs



    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend(self.ComponentOne.parameters)
        p.extend(self.ComponentTwo.parameters)
        return p



    def logp(self):
        return 0


    def get_vfp(self, slabs):
        slabs = np.atleast_2d(slabs)
        s = Structure()
        s |= SLD(0)

        m = SLD(1.)

        for i, slab in enumerate(slabs):
            layer = m(slab[0], slab[3])
            layer.vfsolv.value = slab[4]
            s |= layer

        final_layer=SLD(0)(0)
        final_layer.vfsolv.setp(1)
        s |= final_layer

        s.solvent = SLD(0)


        total_thickness = np.sum(s.slabs()[:, 0])

        # Create a z axis for the volume fraction profile, with a
        # value every angstrom
        zed = np.linspace(0, total_thickness, int(total_thickness+1))

        z, s = s.sld_profile(z=zed)

        return z, s

    def establish_basis(self, zed1, vfp1, zed2, vfp2):

        high_zed = np.max([zed1[-1], zed2[-1]])

        new_zed = np.linspace(0, int(high_zed), int(high_zed+1))

        new_vfp1 = np.interp(new_zed, zed1, vfp1, right=0)
        new_vfp2 = np.interp(new_zed, zed2, vfp2, right=0)

        return new_zed, new_vfp1, new_vfp2