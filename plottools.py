import numpy as np
import matplotlib.pyplot as plt
from refnx._lib import flatten
import matplotlib.patches as mpatches
import corner
import refnx


def plot_burnplot(objective, chain, burn=None,
                  number_histograms=15, thin_factor=1, pt_colgen=plt.cm.Blues,
                  params_to_plot=None, plot_meanline=True, legend=False):
    """
    Constructs plot that enables user to determine if ensemble walkers have
    reached their equilibrium position.

    parameters:
    -----------

    objective:  a refnx.analysis.objective object

    chain:      a numpy.ndarray object with shape [#steps, #temperatures,
                #walkers, #parameters] or [#steps, #walkers, #parameters]

    burn:       The number of steps to burn. Does not alter the chain object,
                but samples that would be kept after the burn are displayed in
                red. This enables the user to determine if the set burn time
                is sufficent.

    number_histograms: number of individual histograms to plot (sample density
                of histograms)

    thin_factor: If the samples have already been thinned the thin factor can
                be supplied so that the step number will be correct.

    output:
    -------
    Returns a figure object
    """
    ptchain = None
    n_samps = chain.shape[0]

    if len(chain.shape) > 3:  # Then parallel tempering
        n_temps = chain.shape[1]
        n_walkers = chain.shape[2]
        n_params = chain.shape[3]
        ptchain = chain
        temps = np.flipud(np.linspace(0, n_temps-1, 5).astype(int))
        colours = pt_colgen(temps/np.max(temps))
        chain = chain[:, 0]  # Only use lowest temperature

    else:  # No parallel tempering
        n_temps = 1
        n_walkers = chain.shape[1]
        n_params = chain.shape[2]
        temps = [0]
        colours = 'k'
        ptchain = np.reshape(np.array([chain]),
                             [n_samps, n_temps, n_walkers, n_params])

    if params_to_plot == None:
        param_index = range(n_params)
    else:
        param_index = params_to_plot
        n_params = len(params_to_plot)

    num_subplot_rows = int(n_params)

    if number_histograms is not None:
        fig, ax = plt.subplots(num_subplot_rows, 2)
        fig.set_size_inches(7, 3*num_subplot_rows)

        chain_index = np.linspace(int(0.05*n_samps), n_samps-1,
                                  number_histograms).astype(int)
        alphas = 0.09 + 0.9*(chain_index -chain_index[0])/float(chain_index[-1] - chain_index[0])

        if burn is None:            # If burn is not supplied then
            burn = chain_index[-1]  # do not plot any as red
    else:
        fig, ax = plt.subplots(num_subplot_rows, 1, sharex=True)
        fig.set_size_inches(3.5, 3*num_subplot_rows)
        ax = np.atleast_2d(ax).T
    fig.set_dpi(200)

    for pindex, axis in zip(param_index, ax):

        param = objective.varying_parameters()[pindex]

        plot_walker_trace(param, ptchain[:, :, :, pindex], axis=axis[0],
                          temps=temps, tcolors=colours,
                          thin_factor=thin_factor, plot_meanline=plot_meanline,
                          legend=legend)

        axis[0].set_title(param.name + ' - value trace')
        axis[0].ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
        axis[0].set_xlabel('step number')
        axis[0].set_ylabel(param.nameco)

        if number_histograms is not None:

            axis[1].set_title(param.name + ' - PDF')
            axis[1].set_xlabel('parameter value')
            axis[1].ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
            axis[1].set_ylabel('probability density')

            for cindex, alpha in zip(chain_index, alphas):
                if cindex < burn:
                    col = 'k'
                else:
                    col = 'r'

                axis[1].hist(chain[cindex, :, pindex], bins=12, density=True,
                             histtype='step', alpha=alpha, color=col)
                mod_cindex = thin_factor*cindex
                try:
                    axis[0].plot([mod_cindex, mod_cindex],
                                 [param.bounds.lb, param.bounds.ub],
                                 linestyle='dashed', color=col, alpha=alpha)
                except AttributeError: #probably a PDF
                    y_limits = axis[0].get_ylim()
                    axis[0].plot([mod_cindex, mod_cindex],
                                 [-1e8, 1e8],
                                 linestyle='dashed', color=col, alpha=alpha)
                    axis[0].set_ylim(y_limits)
        try:
            axis[0].set_ybound(param.bounds.lb, param.bounds.ub)
        except AttributeError:  # Probably a PDF,so no lower and upper bounds
            print('Assuming %s is an unbounded PDF' % param.name)

        axis[0].set_xbound(0, (n_samps-1)*thin_factor)

    fig.tight_layout()
    return fig, fig.axes


def plot_logpost_distribution(objective, chain, burn=0, axis=None, colour='k'):
    """
    parameters:
    -----------

    objective:  Refnx Objective Object

    chain:      a numpy.ndarray object with shape [#steps, #temperatures,
                #walkers, #parameters] or [#steps, #walkers, #parameters]

    burn:       Number of steps to remove from chain before processing
    """
    if len(chain.shape) > 3:  # Parallel tempering
        chain = chain[0]

    # Only look at walkers post-burn
    chain = chain[:, burn:, :]
    samples = chain.reshape((-1, chain.shape[2]))
    if axis is None:
        fig, axis = plt.subplots()

    logposts = []
    for sample in samples:
        objective.setp(sample)
        logposts.append(objective.logpost())

    axis.hist(logposts, density=True, histtype='step', alpha=1, color=colour)


def plot_walker_trace(parameter, samples, temps=[0], tcolors=['k'],
                      thin_factor=1, axis=None, legend=False,
                      plot_meanline=True):
    """
    parameters:
    -----------
    parameter:  refnx parameter object of parameter to be plotted

    samples:    parameter values to plot with shape [ntemps, nwalkers, nsteps]
                or [nwalkers, nsteps]]

    temps:      list of temps to plot

    tcolours:   list of colours to use for temps - must be sample length as
                temps

    axis:       a matplotlib axis object on which the trace profile
                will be plotted.

    thin_factor: If the samples have already been thinned the thin factor can
                 be supplied so that the step number will be correct.
    """

    if axis is None:
        fig, axis = plt.subplots()

    if len(samples.shape) == 2:  # No parallel tempering
        samples = np.array([samples])
        temps = [0]

    steps = np.linspace(0, thin_factor*(samples.shape[0]-1), samples.shape[0])
    leg = []
    for t, c in zip(temps, tcolors):
        leg.append(mpatches.Patch(color=c, label=('T %d' % t)))
        for samp in samples[:, t].T:
            axis.plot(steps, samp, color=c, alpha=0.2)
    try:
        axis.plot(steps, np.ones(steps.shape) * parameter.bounds.lb, color='b')
        axis.plot(steps, np.ones(steps.shape) * parameter.bounds.ub, color='b')
    except AttributeError:  # Probably a PDF,so no lower and upper bounds
        print('Assuming %s is an unbounded PDF'%parameter.name)

    if plot_meanline:
        axis.plot(steps, np.mean(samples[:, 0, :], axis=1), color='k')

    if legend:
        leg.reverse()
        axis.legend(handles=leg, loc='lower center', ncol=5, fontsize='xx-small')


def plot_quantile_profile(x_axes, y_axes, axis=None, quantiles=[68, 95, 99.8],
                          color='k', fullreturn=False):
    """
    Turn an ensembel of profiles into a plot with shaded areas corresponding
    to distribution quantiles

    parameters:
    -----------
    x_axes: python list-type object of profile x-axes
    y_axes: python list-type object of profile y-axes
    axis:   matplotlib axis object
    """
    max_len = 0
    max_x_axis = []

    if axis is None:
        fig, axis = plt.subplots()

    if len(x_axes) == len(y_axes): #One xaxis for every yaxis
        for x_axis, y_axis in zip(x_axes, y_axes):
            assert len(x_axis) == len(y_axis)
            if y_axis.shape[0] > max_len:
                max_len = y_axis.shape[0]
                max_x_axis = x_axis
    else: #Maybe one xaxis for all yaxes?
        x_axis = x_axes
        for y_axis in y_axes:
            assert len(x_axis) == len(y_axis)
            if y_axis.shape[0] > max_len:
                max_len = y_axis.shape[0]
                max_x_axis = x_axis

    x_axis = max_x_axis
    y_axes_array = np.zeros([len(y_axes), max_len])
    for index, y_axis in enumerate(y_axes):
        y_axes_array[index, 0:len(y_axis)] = y_axis

    tran_y_axes = y_axes_array.T

    quant_dict = {}
    for quantile in quantiles:
        q_l = (100 - quantile)/2
        q_h = 100-(100 - quantile)/2
        print(q_l, q_h)

        y_l = np.percentile(tran_y_axes, q_l, axis=1)
        y_h = np.percentile(tran_y_axes, q_h, axis=1)

        quant_dict[str(q_l) + ' low'] = y_l
        quant_dict[str(q_l) + ' high'] = y_h

        mask = y_h > 0
        axis.fill_between(x_axis[mask], y_l[mask], y_h[mask],
                          color=color, alpha=0.3)

    y_median = np.median(tran_y_axes, axis=1)
    mask = y_median > 0
    axis.plot(x_axis[mask], y_median[mask], color=color)

    quant_dict['median'] = y_median
    quant_dict['xaxis'] = x_axis

    if fullreturn is True:
        return quant_dict


def plot_corner(objective, samples):
    labels = []

    for i in flatten(objective.parameters.varying_parameters()):
        labels.append(i.name)

    fig = corner.corner(samples, labels=labels, quantiles=[0.025, 0.5, 0.975],
                        show_titles=True, title_kwargs={"fontsize": 12})

    return fig


def process_samples(objective, pvecs, vfp_location=None):
    if type(objective.model) == refnx.reflect.reflect_model.MixedReflectModel:
        structures = objective.model.structures
    else:
        structures = [objective.model.structure]

    if vfp_location is not None:
        return _process_samples_brush(objective, structures,
                                      pvecs, vfp_location)
    else:
        return _process_samples_generic(objective, structures, pvecs)


def _process_samples_brush (objective, structures, pvecs, vfp_location):
    """
    objective: objective
    """
    num_structs = len(structures)

    report = {}
    samples = []
    moments = []
    areas = []
    scale_fctrs = []
    ismono = []
    logposts = []
    logps = []
    logls = []
    sld_profiles = []
    vfp_profiles = []
    vfp_knots = []
    ref_profiles = []

    best_logpost = -1e108
    best_profile = None
    best_area = None
    best_moment = None
    best_logl = None
    best_logp = None

    max_z = 0

    counter = 0
    for pvec in pvecs:
        counter += 1
        objective.setp(pvec)
        samples.append(pvec)

        for struct in structures:
            vfp = struct[vfp_location]
            moments.append(vfp.moment())
            areas.append(vfp.adsorbed_amount.value)
            ismono.append(is_monotonic(objective))

            z, phi, zk, phik = vfp.profile(extra=True)
            if np.max(z) > max_z:
                max_z = np.max(z)
            vfp_profiles.append([z, phi])
            vfp_knots.append([zk, phik])

            sld_profile = struct.sld_profile()
            sld_profiles.append(sld_profile)

        if len(structures) > 1:
            scale_fctrs.append(objective.model.scales.pvals)
        else:
            scale_fctrs.append(objective.model.scale.value)

        logposts.append(objective.logpost())
        logps.append(objective.logp())
        logls.append(objective.logl())

        ref_profile = [objective.data.x,
                       objective.model(objective.data.x,
                                       x_err=objective.data.x_err)]
        ref_profiles.append(ref_profile)

        if objective.logpost() > best_logpost:
            best_profile = [z, phi, zk, phik]
            best_area = vfp.profile_area()
            best_moment = vfp.moment()
            best_logp = objective.logp()
            best_logl = objective.logl()
            best_sld = sld_profile
            best_ref = ref_profile
            best_logpost = objective.logpost()

    vfp_profiles = unify_xaxes(vfp_profiles, max_z, numpoints=500)
    sld_profiles = unify_xaxes(sld_profiles, max_z, numpoints=500)

    moments = np.reshape(np.array(moments), (counter, num_structs))
    areas = np.reshape(areas, (counter, num_structs))
    scale_fctrs = np.reshape(scale_fctrs, (counter, num_structs))
    ismono = np.reshape(ismono, (counter, num_structs)).T
    vfp_profiles = np.reshape(vfp_profiles, (counter, num_structs, 2, -1))
    vfp_knots = np.reshape(vfp_knots, (counter, num_structs, 2, -1))
    sld_profiles = np.reshape(sld_profiles, (counter, num_structs, 2, -1))

    report['scale factor/s'] = scale_fctrs

    report['1st moment - mean'] = np.mean(moments)
    report['1st moment - stdev'] = np.std(moments)
    report['1st moment - data'] = np.array(moments)
    report['1st moment - best'] = best_moment

    report['area - mean'] = np.mean(areas)
    report['area - stdev'] = np.std(areas)
    report['area - data'] = np.array(areas)
    report['area - best'] = best_area

    report['logpost - mean'] = np.mean(logposts)
    report['logpost - stdev'] = np.std(logposts)
    report['logpost - data'] = np.array(logposts)
    report['logpost - best'] = best_logpost

    report['logp - mean'] = np.mean(logps)
    report['logp - stdev'] = np.std(logps)
    report['logp - data'] = np.array(logps)
    report['logp - best'] = best_logp

    report['is monotonic'] = ismono

    report['logl - mean'] = np.mean(logls)
    report['logl - stdev'] = np.std(logls)
    report['logl - data'] = np.array(logls)
    report['logl - best'] = best_logl

    report['vfp - best'] = best_profile
    report['vfp - profiles'] = np.array(vfp_profiles)
    report['vfp - knots'] = np.array(vfp_knots)

    report['sld - best'] = best_sld
    report['sld - profiles'] = np.array(sld_profiles)

    report['refl - best'] = best_ref
    report['refl - profiles'] = np.array(ref_profiles)

    d = objective.data
    report['refl - data'] = {'Q': d.x, 'R': d.y,
                             'Q err': d.x_err, 'R err': d.y_err}

    report['parameter samples'] = np.array(samples)
    report['objective'] = objective

    return report


def _process_samples_generic(objective, structures, pvecs):
    """
    objective: objective
    """
    num_structs = len(structures)

    report = {}
    samples = []
    scale_fctrs = []
    logposts = []
    logps = []
    logls = []
    sld_profiles = []
    ref_profiles = []

    best_logpost = -1e108

    max_z = 0

    counter = 0
    for pvec in pvecs:
        counter += 1
        objective.setp(pvec)
        samples.append(pvec)

        for struct in structures:
            z, SLD = struct.sld_profile()
            sld_profile = [z, SLD]
            if z[-1] > max_z:
                max_z = np.max(z)
            sld_profiles.append(sld_profile)

        if len(structures) > 1:
            scale_fctrs.append(objective.model.scales.pvals)
        else:
            scale_fctrs.append(objective.model.scale.value)

        logposts.append(objective.logpost())
        logps.append(objective.logp())
        logls.append(objective.logl())

        ref_profile = [objective.data.x,
                       objective.model(objective.data.x,
                                       x_err=objective.data.x_err)]
        ref_profiles.append(ref_profile)

        if objective.logpost() > best_logpost:
            best_logp = objective.logp()
            best_logl = objective.logl()
            best_sld = sld_profile
            best_ref = ref_profile
            best_logpost = objective.logpost()

    sld_profiles = unify_xaxes(sld_profiles, max_z, numpoints=500)

    scale_fctrs = np.reshape(scale_fctrs, (counter, num_structs))
    sld_profiles = np.reshape(sld_profiles, (counter, num_structs, 2, -1))

    report['scale factor/s'] = scale_fctrs

    report['logpost - mean'] = np.mean(logposts)
    report['logpost - stdev'] = np.std(logposts)
    report['logpost - data'] = np.array(logposts)
    report['logpost - best'] = best_logpost

    report['logp - mean'] = np.mean(logps)
    report['logp - stdev'] = np.std(logps)
    report['logp - data'] = np.array(logps)
    report['logp - best'] = best_logp

    report['logl - mean'] = np.mean(logls)
    report['logl - stdev'] = np.std(logls)
    report['logl - data'] = np.array(logls)
    report['logl - best'] = best_logl

    report['sld - best'] = best_sld
    report['sld - profiles'] = np.array(sld_profiles)

    report['refl - best'] = best_ref
    report['refl - profiles'] = np.array(ref_profiles)

    d = objective.data
    report['refl - data'] = {'Q': d.x, 'R': d.y,
                             'Q err': d.x_err, 'R err': d.y_err}

    report['parameter samples'] = np.array(samples)
    report['objective'] = objective

    return report


def unify_xaxes(profiles, max_z, numpoints=500):
    z_axis_master = np.linspace(0, max_z, numpoints)
    profiles_array = np.zeros([len(profiles), 2, numpoints])

    for index, [z_axis, y_axis] in enumerate(profiles):
        profiles_array[index, 1, :] = np.interp(z_axis_master, z_axis, y_axis)
        profiles_array[index, 0, :] = z_axis_master

    return profiles_array


def hist_plot(report, show_prior=False):
    # Fixme: Impliment support for supplying own figures
    """
    report:
        dictionary generated by process_data
    """

    try:
        moment = report['1st moment - data']
        area = report['area - data']
        vfp_exists = True
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        fig.set_size_inches(8, 2.5)
    except KeyError:
        vfp_exists = False
        fig, ax3 = plt.subplots()
        fig.set_size_inches(2.5, 2.5)

    logpost = report['logpost - data']
    logl = report['logl - data']
    logp = report['logp - data']
    norm_scales = (report['scale factor/s'].T/np.sum(report['scale factor/s'],
                   axis=1)).T

    if vfp_exists:
        num_structs = moment.shape[1]
        for idx in range(num_structs):
            c = prob_color(col_mod=idx)
            ax1.hist(moment[:, idx], density=True, color=c, histtype='step')
            ax2.hist(area[:, idx], density=True, color=c, histtype='step')

        moment = np.reshape(moment, (-1))
        area = np.reshape(area, (-1))
        norm_scales = np.reshape(norm_scales, (-1))

        ax1.hist(moment, density=True, weights=norm_scales, color='k', histtype='step')
        ax2.hist(area, density=True, weights=norm_scales, color='k', histtype='step')

        ax1.set_ylabel('Normalised frequency')
        ax1.set_xlabel('Location of 1st Moment')
        ax2.set_xlabel('VFP Area (true dry layer thickness)')
        ax1.tick_params(axis='y', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)


    ax3.hist(logl, density=True, histtype='step', color='xkcd:blue',
             linewidth=1, label='likelihood')
    ax3.hist(logpost, density=True, histtype='step', color='xkcd:purple',
             linewidth=1, label='posterior')

    if show_prior:
        ax4 = ax3.twiny()
        ax4.hist(logp, density=True, histtype='step', color='xkcd:red',
                 linewidth=0.5, label='prior')
        ax4.set_xlabel('ln(prior)', color='xkcd:red')


    ax3.set_xlabel('ln(like), ln(prob)')
    ax3.tick_params(axis='y', labelsize=8)
    ax3.legend(loc='upper left', fontsize='x-small')

    fig.tight_layout()
    fig.set_dpi(200)

    return


def graph_plot(objective=None, pvecs=None, report=None, vfp_location=None, plot_knots=False,
               fig=None, ax=None, logpost_limits=None, ystyle='log', xstyle='lin'):
    """
    objective (refnx.objective):
        Objective object to be plotted.
    pvecs (iterable):
        Iterable providing (or list containing) array-type with length equal
        to the number of varying parameters in the supplied objective.
        If None will use parameter values already in objective.
    report (dict):
        Dictionary as returned by process_samples. Can be provded for a faster plot.
    vfp_location (int):
        Location of volume fraction profile object within the model structure
        (objective.model.structure[vfp_location]). If None will not display
        volume fraction profile.
    plot_knots (Boolean):
        If true will show know locations on the volume fraction profile
    fig, ax:
        graph_plot will use supplied fig, ax objects if provided.
    logpost_limits (list):
        List containing lower and upper limits of the logpost for the system.
        If provided will set the colour of profiles based on their probability.
    """



    if report is not None:
        if logpost_limits is None:
            logpost_limits = 'auto'
        fig = _report_graph_plot(report, plot_knots=False, fig=None, ax=None,
                                 logpost_limits=logpost_limits, ystyle=ystyle, xstyle=xstyle)
    if objective is not None:
        fig = _objective_graph_plot (objective=objective, pvecs=pvecs, vfp_location=vfp_location,
                                plot_knots=plot_knots, fig=fig, ax=ax, logpost_limits=logpost_limits,
                                ystyle=ystyle, xstyle=xstyle)


    fig.tight_layout()
    fig.set_dpi(200)

    return fig, fig.gca()

def _report_graph_plot (report, plot_knots=False, fig=None, ax=None, logpost_limits='auto', ystyle='log', xstyle='lin'):

    try:
        vfps = report['vfp - profiles']
        vfp_exists = True
    except KeyError:
        vfp_exists = False

    slds = report['sld - profiles']
    refs = report['refl - profiles']
    data = report['refl - data']
    logposts = report['logpost - data']

    if ystyle == 'rq4':
        ymult = data['Q']**4
    else:
        ymult = 1

    if logpost_limits == 'auto':
        logpost_limits = [np.min(logposts), np.max(logposts)]

    if vfp_exists:
        fig, [ax1, ax2, ax3] = CreateAxes(fig, ax, 3)
        fig.set_size_inches(8, 2.5)

        for vfp, logpost in zip(vfps, logposts):

            for vfp_substruct, cmod in zip(vfp, [0, 1, 2]):
                c = prob_color(logpost, logpost_limits, cmod)
                ax1.plot(*vfp_substruct, color=c, alpha=0.02)

    else:
        fig, [ax2, ax3] = CreateAxes (fig, ax, 2, xstyle=xstyle, ystyle=ystyle)
        fig.set_size_inches(6, 2.5)

    ax3.errorbar(data['Q'], data['R']*ymult, yerr=data['R err']*ymult,
                 fmt='none', capsize=2, linewidth=1, color='k', alpha=0.7)

    for sld, ref, logpost in zip(slds, refs, logposts):
        c = prob_color(logpost, logpost_limits, 0)
        ax3.plot(ref[0], ref[1]*ymult, color=c, alpha=0.02)

        for sld_substruct, cmod in zip(sld, [0,1,2]):
            c = prob_color(logpost, logpost_limits, cmod)
            ax2.plot(*sld_substruct, color=c, alpha=0.02)

    if logpost_limits is not None:
        leg_patches = [mpatches.Patch(color=(0,0,0,0), label='logpost:'),
                   mpatches.Patch(color=prob_color(logpost_limits[0], logpost_limits, 0), label='   %d'%logpost_limits[0]),
                   mpatches.Patch(color=prob_color(logpost_limits[1], logpost_limits, 0), label='   %d'%logpost_limits[1])]
        ax2.legend(handles=leg_patches, fontsize='x-small', frameon=False)

    return fig



def _objective_graph_plot (objective, pvecs=None, vfp_location=None, plot_knots=False,
               fig=None, ax=None, logpost_limits=None, ystyle='log', xstyle='lin'):
    """
    graph plot if report objective is provided instead of report
    """

    if vfp_location is None:
        fig, [ax2, ax3] = CreateAxes (fig, ax, num_plots=2, xstyle=xstyle, ystyle=ystyle)
    else:
        fig, [ax1, ax2, ax3] = CreateAxes (fig, ax, num_plots=3, xstyle=xstyle, ystyle=ystyle)

    if ystyle == 'rq4':
        ymult = objective.data.x**4
    else:
        ymult = 1
    # Plot the reflectometry data on the R vs Q plot
    ax3.errorbar(objective.data.x, objective.data.y*ymult,
                 yerr=objective.data.y_err*ymult, fmt='none', capsize=2,
                 linewidth=1, color='k', alpha=0.7)

    if pvecs is None:
        al = 1
        pvecs = [objective.varying_parameters()]
    else:
        al = 0.02

    if type(objective.model) == refnx.reflect.reflect_model.MixedReflectModel:
        structures = objective.model.structures
    else:
        structures = [objective.model.structure]

    # Iterate through selected parameter variables and plot them
    for pvec in pvecs:
        objective.setp(pvec)

        for struct, c_mod in zip(structures, [0, 1, 2]):
            c = prob_color(objective.logpost(), logpost_limits, c_mod)
            if vfp_location is not None:
                z, phi, zk, phik = struct[vfp_location].profile(extra=True)
                ax1.plot(z, phi, color=c, alpha=al)
                if plot_knots:
                    ax1.scatter(zk, phik, color='r', alpha=al)

            ax2.plot(*struct.sld_profile(), color=c, alpha=al)

        c = prob_color(objective.logpost(), logpost_limits, 0)
        ax3.plot(objective.data.x, objective.model(objective.data.x, x_err=objective.data.x_err)*ymult,
                 color=c, alpha=al)

    if logpost_limits is not None:
        leg_patches = [mpatches.Patch(color=(0, 0, 0, 0), label='logpost:'),
                       mpatches.Patch(color=prob_color(logpost_limits[0], logpost_limits, 0), label='   %d'%logpost_limits[0]),
                       mpatches.Patch(color=prob_color(logpost_limits[1], logpost_limits, 0), label='   %d'%logpost_limits[1])]

        ax2.legend(handles=leg_patches, fontsize='x-small', frameon=False)

    return fig


def prob_color(logpost=None, logpost_bounds=None, col_mod=0):
    """
    logpost: probability within logpost_bounds
    logpost_bounds: upper and lower bounds of logpostaility
    col_mod: Modifies the colour, making it lighter or darker
    """
    if logpost_bounds is None:
        return plt.cm.Set1(col_mod/8.0)

    x = (logpost-logpost_bounds[0])/(logpost_bounds[1]-logpost_bounds[0])

    if x > 0.99:
        x = 0.99
    elif x < 0.01:
        x = 0.01

    x = (x*0.8 + 0.1)


    col_mod = (col_mod)*0.3 + 1
    col = np.array(plt.cm.plasma(x))*col_mod
    col[col>1] = 1
    col[col<0] = 0

    return tuple(col)


def CreateAxes (fig, ax, num_plots, ystyle='log', xstyle='lin'):
    if num_plots is 3:
        if fig is None:
            assert ax is None , 'You must provide both a figure and axes objects'
            fig, ax = plt.subplots(1, 3)
            fig.set_dpi(100), fig.set_size_inches(10, 3)

        else:
            assert ax is not None , 'You must provide both a figure and axes objects'
            assert len(ax) == 3 , 'Must provide three axes for vfp plot'
        [ax1, ax2, ax3] = ax
        # Plot labels on axis 1
        ax1.set_ylabel(r'Volume Fraction')
        ax1.set_xlabel(r'Distance from substrate, $\rm{\AA}$')

    else:
        if fig is None:
            assert ax is None , 'You must provide both a figure and axes objects'
            fig, ax = plt.subplots(1, 2)
            fig.set_dpi(100), fig.set_size_inches(6.6, 3)
        else:
            assert ax is not None , 'You must provide both a figure and axes objects'
            assert len(ax) == 2 , 'Must provide three axes for vfp plot'
        [ax2, ax3] = ax
    # Plot other labels
    ax2.set_ylabel(r'SLD, $\rm{\AA}^{-2}$')
    ax2.set_xlabel(r'Distance from substrate, $\rm{\AA}$')

    ax3.set_ylabel(r'$R$')
    ax3.set_xlabel(r'$Q$, $\rm{\AA}^{-1}$')

    if ystyle is not 'lin':
        ax3.set_yscale('log')

    if xstyle is 'log':
        ax3.set_xscale('log')

    return fig, ax


def is_monotonic(objective):
    mono = True
    vparams = objective.varying_parameters()
    vpnames = [x for x in vparams.names() if 'spline vff' in x]
    for vpname in vpnames:
        if vparams[vpname].value > 1:
            mono = False
            return mono

    return mono