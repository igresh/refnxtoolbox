from __future__ import division
import numpy as np
import sys

from scipy.interpolate import PchipInterpolator as Pchip
from scipy.integrate import simps

from refnx.reflect import Structure, Component, SLD, Slab
from refnx.analysis import Parameters, Parameter, possibly_create_parameter

import warnings


EPS = np.finfo(float).eps

class FreeformVFP(Component):
    """
    """
    def __init__(self, adsorbed_amount, vff, dzf, polymer_sld, name='',
                 left_slabs=(), right_slabs=(),
                 interpolator=Pchip, zgrad=True,
                 microslab_max_thickness=1):
        """
        Parameters
        ----------
        Adsorbed Amount : Parameter or float
            The total extent of the spline region
        vff: sequence of Parameter or float
            Volume fraction at each of the spline knots, as a fraction of
            the volume fraction of the rightmost left slab
        dzf : sequence of Parameter or float
            Separation of successive knots, will be normalised to a 0-1 scale.
        polymer_sld : SLD or float
            SLD of polymer
        name : str
            Name of component
        gamma : Parameter
            The dry adsorbed amount of polymer
        left_slabs : sequence of Slab
            Polymer Slabs to the left of the spline
        right_slabs : sequence of Slab
            Polymer Slabs to the right of the spline
        interpolator : scipy interpolator
            The interpolator for the spline
        zgrad : bool, optional
            Set to `True` to force the gradient of the volume fraction to zero
            at each end of the spline.
        microslab_max_thickness : float
            Thickness of microslicing of spline for reflectivity calculation.
        """

        assert len(vff) + 1 == len(dzf) , 'Length of dzf must be one greater\
                                           than length of vff'

        self.name = name

        if isinstance(polymer_sld, SLD):
            self.polymer_sld = polymer_sld
        else:
            self.polymer_sld = SLD(polymer_sld)

        # left and right slabs are other areas where the same polymer can
        # reside
        self.left_slabs = [slab for slab in left_slabs if
                           isinstance(slab, Slab)]
        self.right_slabs = [slab for slab in right_slabs if
                            isinstance(slab, Slab)]

        # use the volume fraction of the last left_slab as the initial vf of
        # the spline, if not left slabs supplied start at vf 1
        if len(self.left_slabs):
            self.start_vf = 1 - self.left_slabs[-1].vfsolv.value
        else:
            self.start_vf = 1

        # in contrast use a vf = 0 for the last vf of
        # the spline, unless right_slabs is specified
        if len(self.right_slabs):
            self.end_vf = 1 - self.right_slabs[0].vfsolv.value
        else:
            self.end_vf = 0

        self.microslab_max_thickness = microslab_max_thickness

        self.adsorbed_amount = (
            possibly_create_parameter(adsorbed_amount,
                                      name='%s - adsorbed amount' % name))

        # dzf are the spatial gaps between the spline knots
        self.dzf = Parameters(name='dzf - spline')
        for i, z in enumerate(dzf):
            p = possibly_create_parameter(
                z,
                name='%s - spline dzf[%d]' % (name, i))
            p.range(0, 1)
            self.dzf.append(p)

        # vf are the volume fraction values of each of the spline knots
        self.vff = Parameters(name='vff - spline')
        for i, v in enumerate(vff):
            p = possibly_create_parameter(
                v,
                name='%s - spline vff[%d]' % (name, i))
            p.range(0, 1)
            self.vff.append(p)

        self.zgrad = zgrad
        self.interpolator = interpolator

        self.__cached_interpolator = {'zeds': np.array([]),
                                      'vf': np.array([]),
                                      'interp': None,
                                      'adsorbed amount': -1}
    def _update_vfs (self):
        # use the volume fraction of the last left_slab as the initial vf of
        # the spline, if not left slabs supplied start at vf 1
        if len(self.left_slabs):
            self.start_vf = 1 - self.left_slabs[-1].vfsolv.value
        else:
            self.start_vf = 1

        # in contrast use a vf = 0 for the last vf of
        # the spline, unless right_slabs is specified
        if len(self.right_slabs):
            self.end_vf = 1 - self.right_slabs[0].vfsolv.value
        else:
            self.end_vf = 0


    def _vff_to_vf(self):
        self._update_vfs()
        vf = np.cumprod(self.vff) * (self.start_vf-self.end_vf) + self.end_vf
        vf[vf>1] = 1
        return vf

    def _dzf_to_zeds(self):
        zeds = np.cumsum(self.dzf)
        # Normalise dzf to unit interval.
        # clipped to 0 and 1 because we pad on the LHS, RHS later
        # and we need the array to be monotonically increasing
        zeds /= zeds[-1]
        zeds = np.clip(zeds, 0, 1)
        zeds = zeds[0:-1]
        return zeds

    def _extent(self):
        # First calculate slab area:
        slab_area = self._slab_area()
        difference = self.adsorbed_amount - slab_area

        assert difference > 0 , 'Your slab area has exceeded your adsorbed amount!'

        interpolator = self._vfp_interpolator()
        extent = difference/interpolator.integrate(0, 1)

        return extent


    def _slab_area(self):
        area = 0
        for slab in self.left_slabs:
            _slabs = slab.slabs
            area += _slabs[0, 0] * (1 - _slabs[0, 4])
        for slab in self.right_slabs:
            _slabs = slab.slabs
            area += _slabs[0, 0] * (1 - _slabs[0, 4])
        return area


    def _vfp_interpolator(self):
        """
        The spline based volume fraction profile interpolator

        Returns
        -------
        interpolator : scipy.interpolate.Interpolator
        """

        zeds = self._dzf_to_zeds()
        vf = self._vff_to_vf()

        # do you require zero gradient at either end of the spline?
        if self.zgrad:
            zeds = np.concatenate([[-1.1, 0 - EPS],
                                   zeds,
                                   [1 + EPS, 2.1]])
            vf = np.concatenate([[self.start_vf, self.start_vf],
                                 vf,
                                 [self.end_vf, self.end_vf]])
        else:
            zeds = np.concatenate([[0 - EPS], zeds, [1 + EPS]])
            vf = np.concatenate([[self.start_vf], vf, [self.end_vf]])

        # cache the interpolator
        cache_zeds = self.__cached_interpolator['zeds']
        cache_vf = self.__cached_interpolator['vf']
        cache_adsamt = self.__cached_interpolator['adsorbed amount']

        # you don't need to recreate the interpolator
        if (np.array_equal(zeds, cache_zeds) and
                np.array_equal(vf, cache_vf) and
                np.equal(self.adsorbed_amount, cache_adsamt)):
            return self.__cached_interpolator['interp']
        else:
            self.__cached_interpolator['zeds'] = zeds
            self.__cached_interpolator['vf'] = vf
            self.__cached_interpolator['adsorbed amount'] = float(self.adsorbed_amount)

        # TODO make vfp zero for z > self.extent
        interpolator = self.interpolator(zeds, vf)
        self.__cached_interpolator['interp'] = interpolator
        return interpolator


    def __call__(self, z):
        """
        Calculates the volume fraction profile of the spline

        Parameters
        ----------
        z : float
            Distance along vfp

        Returns
        -------
        vfp : float
            Volume fraction
        """
        interpolator = self._vfp_interpolator()
        vfp = interpolator(z / float(self._extent()))
        return vfp


    def moment(self, moment=1):
        """
        Calculates the n'th moment of the profile

        Parameters
        ----------
        moment : int
            order of moment to be calculated

        Returns
        -------
        moment : float
            n'th moment
        """
        zed, profile = self.profile()
        profile *= zed**moment
        val = simps(profile, zed)
        area = self.profile_area()
        return val / area
    
    def is_monotonic(self):
        return np.all(self.dzf.pvals < 1)


    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.adsorbed_amount, self.dzf, self.vff,
                  self.polymer_sld.parameters])
        p.extend([slab.parameters for slab in self.left_slabs])
        p.extend([slab.parameters for slab in self.right_slabs])
        return p


    def lnprob(self):
        lnprob = 0

        return lnprob


    def profile_area(self):
        """
        Calculates integrated area of volume fraction profile

        Returns
        -------
        area: integrated area of volume fraction profile
        """
        interpolator = self._vfp_interpolator()
        area = interpolator.integrate(0, 1) * float(self._extent())

        area += self._slab_area()

        return area


    @property
    def slabs(self):

        cutoff = 10000
        if self._extent() > cutoff:
            warnings.warn('extent > %d, perfoming refl. calc on first %dA.'%
                   (cutoff, cutoff), RuntimeWarning)

            slab_extent = cutoff
        else:
            slab_extent = self._extent()


        num_slabs = np.ceil(float(slab_extent) / self.microslab_max_thickness)
        slab_thick = float(slab_extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give last slab a miniscule roughness so it doesn't get contracted
        slabs[-1:, 3] = 0.5

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick



        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        slabs[:, 4] = 1 - self(dist)

        return slabs


    def profile(self, extra=False):
        """
        Calculates the volume fraction profile

        Returns
        -------
        z, vfp : np.ndarray
            Distance from the interface, volume fraction profile
        """
        s = Structure()
        s |= SLD(0)

        m = SLD(1.)

        for i, slab in enumerate(self.left_slabs):
            layer = m(slab.thick.value, slab.rough.value)
            if not i:
                layer.rough.value = 0
            layer.vfsolv.value = slab.vfsolv.value
            s |= layer

        polymer_slabs = self.slabs
        offset = np.sum(s.slabs[:, 0])

        for i in range(np.size(polymer_slabs, 0)):
            layer = m(polymer_slabs[i, 0], polymer_slabs[i, 3])
            layer.vfsolv.value = polymer_slabs[i, -1]
            s |= layer

        for i, slab in enumerate(self.right_slabs):
            layer = m(slab.thick.value, slab.rough.value)
            layer.vfsolv.value = 1 - slab.vfsolv.value
            s |= layer

        s |= SLD(0, 0)

        # now calculate the VFP.
        total_thickness = np.sum(s.slabs[:, 0])
        if total_thickness < 500:
            num_zed_points = total_thickness
        else:
            num_zed_points = 500
        zed = np.linspace(0, total_thickness, num_zed_points)
        # SLD profile puts a very small roughness on the interfaces with zero
        # roughness.
        zed[0] = 0.01
        z, s = s.sld_profile(z=zed)
        s[0] = s[1]

        # perhaps you'd like to plot the knot locations
        zeds = self._dzf_to_zeds()
        zed_knots = zeds * float(self._extent()) + offset

        if extra:
            return z, s, zed_knots, self._vff_to_vf()
        else:
            return z, s


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
        return Parameter(name='%s - real'%self.name, value=v)
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