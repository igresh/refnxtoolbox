# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:14:32 2018

@author: igres
"""

from scipy.stats import norm
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx.reflect import reflectivity

from copy import copy
import warnings

import numpy as np

class BaseModel (object):
    """
    NOT IMPLIMENTED
    
    Should probably get this implimented in refnx propper at some point...

    Does not touch structures
    """
    
    def __init__(self, bkg, name, dq, threads, quad_order):
        self.name = name
        self.threads = threads
        self.quad_order = quad_order
        self._bkg = possibly_create_parameter(bkg, name='bkg')
        
        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name='dq - resolution')
    
    
    @property
    def bkg(self):
        r"""
        :class:`refnx.analysis.Parameter` - linear background added to all
        model values.

        """
        return self._bkg


    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value

        
    @property
    def dq(self):
        r"""
        :class:`refnx.analysis.Parameter`

            - `dq.value == 0`
               no resolution smearing is employed.
            - `dq.value > 0`
               a constant dQ/Q resolution smearing is employed.  For 5%
               resolution smearing supply 5. However, if `x_err` is supplied to
               the `model` method, then that overrides any setting reported
               here.

        """
        return self._dq
    

    @dq.setter       
    def dq(self, value):
        self._dq.value = value
        


class MetaModel (BaseModel):
    """
    Takes two models with scale factors and combines them
    """
    
    def __init__(self, models, scales, add_params=None, bkg=1e-7, name='', dq=5, threads=-1, quad_order=17):
        super().__init__(bkg=1e-7, name='', dq=5, threads=-1, quad_order=17)
        
        self.models = models
        
        if scales is not None and len(models) == len(scales):
            tscales = scales
        elif scales is not None and len(models) != len(scales):
            raise ValueError("You need to supply scale factor for each"
                             " structure")
        else:
            tscales = [1 / len(models)] * len(models)

        pscales = []
        for scale_num, scale in enumerate(tscales):
            p_scale = possibly_create_parameter(scale, name='scale %d'%scale_num)
            pscales.append(p_scale)
            
        self._scales = pscales
        
        if add_params is not None:
            self.additional_params = []
            for param in add_params:
                self.additional_params.append(param)
    
    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        return self.model(x, p=p, x_err=x_err)
        
    def model(self, x, p=None, x_err=None):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameter, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray

        """
        meta_model = np.zeros_like(x)
        for model, scale in zip(self.models, self._scales):
            model.bkg.setp(0)
            meta_model += model(x, p, x_err) * scale.value
            
        return meta_model + self.bkg.value

    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically calculated elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        logp = 0
        for model in self.models:
            logp += model.logp()

        return logp

    @property
    def scales(self):
        r"""
        :class:`refnx.analysis.Parameter` - the reflectivity from each of the
        structures are multiplied by these values to account for patchiness.
        """
        return self._scales
    
    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters` - parameters associated with this
        model.

        """
        p = Parameters(name='meta instrument parameters')
        p.extend([self.bkg, self.dq])
        p.extend(self.additional_params)
        self._parameters = Parameters(name=self.name)

        for model, scale in zip(self.models, self._scales):
            p.extend([scale])
            p.extend(model.parameters.flattened())

        self._parameters.append(p)
   
        return self._parameters


class DistributionModel (object):
    """
    structure : refnx structure object
        The interfacial structure.
    loc_in_struct : int
        The index of the structural component that you want to impliment as a
        distribution.
    param_name : str
        the name of the parameter of the distribution component that you want
        to vary. (Currently only thickness is implimented)
    pdf : function
        if None, will default to a normal distribution
    pdf_kwargs : dict
        Dictionary with kwargs for the pdf. This will be used to parameterise
        the pdf.
    num_structs : int
        number of discrete points that will be generated along the pdf
    scale : float or refnx.analysis.Parameter, optional
        NOT IMPLIMENTED
    bkg : float or refnx.analysis.Parameter, optional
        Q-independent constant background added to all model values. This is
        turned into a Parameter during the construction of this object.
    name : str, optional
        Name of the Model
    dq : float or refnx.analysis.Parameter, optional

        - `dq == 0` then no resolution smearing is employed.
        - `dq` is a float or refnx.analysis.Parameter
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.

        However, if `x_err` is supplied to the `model` method, then that
        overrides any setting given here. This value is turned into
        a Parameter during the construction of this object.
    threads: int, optional
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == 0` then all available processors are
        used.
    quad_order: int, optional
        the order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example, 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with bragg peaks.

    """
    def __init__ (self, structure, loc_in_struct, param_name='Thickness',
                  pdf=None, pdf_kwargs=None, num_structs=11, scale=1, bkg=1e-7, name='',
                  dq=5, threads=-1, quad_order=17):
        
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order
        
        self.master_structure = structure 
        self.loc_in_struct = loc_in_struct
        self.param_name = param_name.lower()
        self.num_structs = num_structs
        
        if pdf is None:
            self.pdf = norm.pdf
        
            if pdf_kwargs is None:
                self.pdf_params = []
                self.pdf_params.append(possibly_create_parameter(value=10, name='loc'))
                self.pdf_params.append(possibly_create_parameter(value=1, name='scale'))
            else:
                print ('Warning: You have provided pdf_kwargs without providing a pdf')
        else:
            assert pdf_kwargs is not None, 'You must supply pdf_kwargs'
            self.pdf = pdf
            self.pdf_params = []
            for kw in pdf_kwargs:
                self.pdf_params.append(possibly_create_parameter(pdf_kwargs[kw], name=kw))

        self._structures = self.create_structures()
        self._scales = np.ones(self.num_structs)/self.num_structs
        
        self._bkg = possibly_create_parameter(bkg, name='bkg')
        
        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name='dq - resolution')

        
        self.generate_thicknesses()
        
        
    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        return self.model(x, p=p, x_err=x_err)
    

        
    def create_structures(self):
        structures = []
        self.distribution_params = []
        COI = self.master_structure[self.loc_in_struct]
        

        for i in range(self.num_structs):
            new_COI = copy(COI)
            if self.param_name == 'thickness':
                new_COI.thick = Parameter(name='%d - Thick'%i, value=new_COI.thick.value, vary=False)
                self.distribution_params.append(new_COI.thick)
            elif self.param_name == 'adsorbed amount':
                new_COI.adsorbed_amount = Parameter(name='%d - Ads. amnt.'%i, value=new_COI.adsorbed_amount.value, vary=False)
                self.distribution_params.append(new_COI.adsorbed_amount)
            else:
                print ('param_name not recognized')
                
            struct = self.master_structure[0]
            for component in self.master_structure[1:]:
                if component is not COI:
                    struct = struct | component
                else:
                    struct = struct | new_COI
            
            struct.solvent = self.master_structure.solvent
            structures.append(struct)
        
        return structures
    
    @property
    def pdf_kwargs(self):
        temp = {}
        for param in self.pdf_params:
            temp[param.name] = param.value
        return temp
        
        
    def generate_thicknesses (self):
        d = np.linspace(0,5000, 10000)
        pdf = self.pdf(d, **self.pdf_kwargs)
        effective_component = d[pdf > 0.01*pdf.max()]
        effective_pdf = pdf[pdf > 0.01*pdf.max()]
        effective_range = [effective_component.min(), effective_component.max()]
        
        pvals = np.linspace(*effective_range, num=self.num_structs)
        
        for pval, param in zip(pvals, self.distribution_params):
            param.value = pval
            
        scales = np.interp(pvals, effective_component, effective_pdf)
        self._scales = scales/np.sum(scales)
        
    @property
    def dq(self):
        r"""
        :class:`refnx.analysis.Parameter`

            - `dq.value == 0`
               no resolution smearing is employed.
            - `dq.value > 0`
               a constant dQ/Q resolution smearing is employed.  For 5%
               resolution smearing supply 5. However, if `x_err` is supplied to
               the `model` method, then that overrides any setting reported
               here.

        """
        return self._dq
    

    @dq.setter       
    def dq(self, value):
        self._dq.value = value

    @property
    def scales(self):
        r"""
        :class:`refnx.analysis.Parameter` - the reflectivity from each of the
        structures are multiplied by these values to account for patchiness.
        """
        return self._scales

    @property
    def bkg(self):
        r"""
        :class:`refnx.analysis.Parameter` - linear background added to all
        model values.

        """
        return self._bkg

    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value

    def model(self, x, p=None, x_err=None):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameter, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray

        """
        self.generate_thicknesses()

        if p is not None:
            self.parameters.pvals = np.array(p)
        if x_err is None:
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        scales = np.array(self.scales)

        y = np.zeros_like(x)

        for scale, structure in zip(scales, self.structures):
            y += reflectivity(x,
                              structure.slabs()[..., :4],
                              scale=scale,
                              dq=x_err,
                              threads=self.threads,
                              quad_order=self.quad_order)

        return y + self.bkg.value

    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically calculated elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        logp = 0
        for structure in self._structures:
            logp += structure.logp()

        return logp

    @property
    def structures(self):
        r"""
        list of :class:`refnx.reflect.Structure` that describe the patchiness
        of the surface.

        """
        return self._structures

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters` - parameters associated with this
        model.

        """
        p = Parameters(name='instrument parameters')
        p.extend([self.pdf_params, self.bkg, self.dq])

        self._parameters = Parameters(name=self.name)
        self._parameters.append([p])
        self._parameters.extend([structure.parameters for structure
                                 in self._structures])
        return self._parameters

      

        

            