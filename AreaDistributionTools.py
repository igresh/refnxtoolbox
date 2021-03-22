"""
Created on Tue Jan  7 14:30:09 2020

@author: Isaac
"""
import numpy as np

from matplotlib import pyplot as plt

from scipy.stats import gamma
import scipy.special as sc

from refnx.reflect import SLD, Slab, ReflectModel, MixedReflectModel
from refnx.dataset import ReflectDataset as RD
from refnx.analysis import Objective, Parameter

from AreaDistributionModel import MetaModel, DistributionModel

from areaSlab import area_slabT


def dist_pdf(z, loc, scale, a, tail=0, tail_len=0):
    """
    Separation probability density function for confinement cell modelling.

    Used for modelling neutron reflectometry data collected using the
    confinment cell of Prescott et al. 2016-2020.

    It is produced by the normalised summation of
    a typical gamma distribution (parameterised by shape and scale varaiables)
    and a custom tail distribution (linear decay).

    Parameters
    ----------
    z : np.array
        Seperations at which the function is evaluated.
    loc : float
        The separation at which the PDF starts (i.e. shits PDF to lower /
        higher seperations).
    scale : float
        The width of the gamma-component of the distribution.
    a : float, optional
        The shape of the gamma-component of the distribution. Lower a
        result in 'exponential' PDFs, whilst higher a results in more
        'normal' PDFs.
    tail : float, optional
        The weighting of the tail-component. Higher values result in a
        higher tail. Values of zero result in no tail. The default is 0.
        The default is 0.
    tail_len : float, optional
        The length of the tail-component. The default is 0.

    Returns
    -------
    np.array
        Probability density at separations provided.

    """
    pdf1 = gamma.pdf(z, loc=loc, scale=scale, a=a)

    tpeak = loc + (a-1)*scale
    tcut = tail_len + loc + (a-1)*scale
    tstart = gamma.interval(0.999, loc=loc, scale=scale, a=a)[0] #start at 1% of cdf 
    pdf2 = np.ones_like(z)
    pdf2[z < tpeak] = (z[z < tpeak] - tstart)/(tpeak - tstart)
    pdf2[z > tpeak] = (tcut - z[z > tpeak])/(tcut - tpeak)
    pdf2[z > tcut] = 0
    pdf2[z < tstart] = 0

    pdf = pdf1 + pdf2*tail

    pdf[z < 100] = 0

    return pdf/np.trapz(pdf, z)


def plot_distmodel(objective, refl_mode='rq4', maxd=1000):
    """
    Plot a distribution model for maximum introspection.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        An objective containing a MetaModel, configured in the arbitrary
        method used by Gresham circa 2019.
    refl_mode : string, optional
        The method for plotting the reflectometry profiles, either 'log' or
        'rq4'. The default is 'rq4'.
    maxd : float, optional
        The maximum separation at which the distirubtion is plotted.
        The default is 1000.

    Returns
    -------
    None.

    """
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(10, 3), dpi=150)

    q = objective.data.x
    r = objective.data.y
    r_err = objective.data.y_err

    if refl_mode == 'rq4':
        q4 = q**4
        txt_loc2 = 'bottom'
    else:
        q4 = 1
        txt_loc2 = 'top'

    if type(objective.model) == MetaModel:
        distmodel = objective.model.models[0]
        distscale = objective.model.scales[0]
        h2omodel = objective.model.models[1]
        h2oscale = objective.model.scales[1]

        ax1.plot(*h2omodel.structure.sld_profile(), color='b', alpha=1, lw=1)
        ax2.plot(q, h2omodel(q)*q4*h2oscale, color='b', alpha=1, lw=1)
        ax2.plot(q, distmodel(q)*q4*distscale, color='red', alpha=1, lw=1)

        scales = objective.model.scales
        ax2.text(0.95, 0.23, 'mScales: %0.3f, %0.3f' % (scales[0].value, scales[1].value),
                 ha='right', va=txt_loc2, size='small',
                 transform=ax2.transAxes)

    else:
        distmodel = objective.model
        distscale = 1

    maxscale = np.max(distmodel.scales)

    # Need to call the model to refresh parameters
    objective.model(q)

    d = np.linspace(0, maxd, 5000)
    pdf = distmodel.pdf(d, **distmodel.pdf_kwargs)
    ax3.plot(d, pdf)

    for struct, scale in zip(distmodel.structures, distmodel.scales):
        normscale = np.min([1, np.max([scale/maxscale, 0.001])])

        ax1.plot(*struct.sld_profile(), color='xkcd:crimson',
                 alpha=normscale, lw=1)

        dummy_model = ReflectModel(struct, bkg=objective.model.bkg.value)

        ax2.plot(q, dummy_model.model(q)*q4*normscale*distscale,
                 alpha=normscale*0.5, color='xkcd:crimson', lw=1)

        thick = struct[2].thick.value

        ax3.scatter(thick, np.interp(thick, d, pdf), marker='.', color='k',
                    alpha=normscale)

    ax2.plot(q, objective.model(q)*q4, color='k', alpha=1)
    ax2.errorbar(q, r*q4, yerr=r_err*q4, color='b', alpha=0.5)
    ax2.set_yscale('log')

    ax1.set_xlabel('Thickness, $\mathrm{\AA}$')
    ax1.set_ylabel('SLD, $\mathrm{\AA}^{-2}$')
    ax2.set_xlabel('$Q$, $\mathrm{\AA}^{-1}$')
    ax2.set_ylabel('$R$')
    ax3.set_xlabel('Thickness, $\mathrm{\AA}$')

    kwargs = distmodel.pdf_kwargs
    for i, key in enumerate(kwargs):
        ax3.text(0.95, 0.95-0.06*i, '%s: %0.4f' % (key, kwargs[key]),
                 ha='right', va='top', size='small',
                 transform=ax3.transAxes)

    i = 0
    for p in distmodel.master_structure.parameters.flattened():
        if p.vary is True:
            ax1.text(0.95, 0.05+0.06*i, '%s: %0.3f' % (p.name, p.value),
                     ha='right', va='bottom', size='small',
                     transform=ax1.transAxes)
            i += 1

    ax2.text(0.95, 0.17, 'background: %d' % objective.model.bkg.value,
             ha='right', va=txt_loc2, size='small',
             transform=ax2.transAxes)
#    ax2.text(0.95, 0.05, 'lnprob: %d' % (objective.logpost()),
#             ha='right', va=txt_loc2, size='small',
#             transform=ax2.transAxes)
    ax2.text(0.95, 0.11, 'chisqr: %d' % (objective.chisqr()),
             ha='right', va=txt_loc2, size='small',
             transform=ax2.transAxes)

    ax1.set_xbound(-50, maxd)
    fig.tight_layout()


def make_objective(name, data, pset, unconfined_objective=None, bset=None,
                   num_points=101, withtail=True):
    """
    Make an objective featuring a distribution model, in style of Gresham 2020.

    Parameters
    ----------
    name : string
        name of objective, used for identification.
    data : refnx.dataset.ReflectDataset
        Reflection dataset.
    unconfined_objective : refnx.analysis.objective
        An objective that has been fitted to an unconfined dataset that is
        otherwise at the same conditions as the confined dataset. Used for 
        calculation the reflectivity from unconfined areas.
    pset : dictionary
        Dictionary containing parameter values, indexed by name and used to set
        parameters.
    bset : dictionary
        Dictionary containing parameter bounds, indexed by name and used to set
        parameter bounds.
    num_points : int, optional
        The number of points to sample from the distribution when calculating
        the reflectometry. The default is 101.
    withtail : bool, optional
        Whether the tail component of the distribution should be varying.
        The default is True.

    Returns
    -------
    obj : refnx.analysis.objective
        An objective creating a distribution model.
    """
    h2o     = SLD(-0.54, 'h2o')
    melCM   = SLD(2.60,   'Melinex CM soln')
    melinex = SLD(2.56,   'Melinex') 
    sio2    = SLD(3.47,   'sio2')
    si      = SLD(2.07,  'si')
    polymer = SLD(0.85,  'PNIPAM')

    # Make sure nothing in our (already fitted) unconfined objective varys.


    si_l          = si(0, 0)

    # Silica
    sio2_l        = sio2(14.6, 2)
    sio2_l.vfsolv.setp                   (value=0.003)

    polymer_l_mel = area_slabT(adsorbed_amount=200, dry_sld=polymer, rough=2,
                               thick=1500, name='polymer')

    mellinex_l = melinex(0,14)
    mellinex_l.rough.setp                (vary=False, bounds=(1, 20))

    h2o_l = h2o(0,10)

    structure_mel = si_l | sio2_l | polymer_l_mel | mellinex_l

    if unconfined_objective is not None:
        for vp in unconfined_objective.varying_parameters().flattened():
            vp.setp(vary=False)
        unconfined_objective.model.structure[3].adsorbed_amount = polymer_l_mel.adsorbed_amount
        structure_h2o = si_l | sio2_l | unconfined_objective.model.structure[2] | unconfined_objective.model.structure[3] | h2o_l
    else:
        structure_h2o = si_l | h2o_l


    structure_mel.solvent = h2o
    structure_h2o.solvent = h2o

    if unconfined_objective is not None:
        structure_h2o.contract = 1.5

    distmodel = DistributionModel(structure_mel, loc_in_struct=2,
                                  num_structs=num_points, pdf=dist_pdf,
                                  pdf_kwargs={'loc':1, 'scale':1, 'a':1, 'tail':0.00, 'tail_len':400})
    distmodel.pdf_params[0].setp(value=210)
    distmodel.pdf_params[1].setp(value=6)
    distmodel.pdf_params[2].setp(value=3)

    if withtail:
        distmodel.pdf_params[3].setp(value=0.0001)
        distmodel.pdf_params[4].setp(value=400)
    else:
        distmodel.pdf_params[3].setp(value=0, vary=False)
        distmodel.pdf_params[4].setp(value=0, vary=False)   

    h2omodel = ReflectModel(structure_h2o, name='h2o')


    sratio = Parameter(value=0.1, name='scale ratio')

    model = MetaModel([distmodel, h2omodel], [0.5, 0.5], add_params=[sratio])
    model.scales[0].setp(value=0.97)
    model.scales[1] = model.scales[0]*sratio
    model.bkg.setp(value=0)

    obj = Objective(model, data, name=name)
    for key in pset:
        set_param(obj.parameters, key, value=pset[key])

    if bset is not None:
        for key in bset:
            set_param(obj.parameters, key, bounds=bset[key])

    return obj


def set_param(params, pname, value=None, bounds=None):
    """
    Set a parameter in an objective by name.

    Parameters
    ----------
    params : refnx.analysis.Parameters
        A list of parameters from an objective (i.e. objective.Parameters)
    pname : string
        the name of the parameter to set.
    value : float
        The value to set the parameter to.

    Returns
    -------
    None.

    """
    if value is not None:
        for p in params.flattened():
            if p.name == pname:
                p.value = value
                break

    if bounds is not None:
        for p in params.flattened():
            if p.name == pname:
                p.bounds = bounds
                p.vary = True
                break
