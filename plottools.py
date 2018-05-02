import numpy as np
import matplotlib.pyplot as plt
from refnx._lib import flatten
import matplotlib.patches as mpatches
import corner
import refnx
import sys
import re
import array

import brush_redux
from refnx._lib import possibly_open_file
from refnx.analysis import process_chain
import matplotlib.patches as mpatches




def unpack_objective(obj):
    data = obj.data
    structure = obj.model.structure
    model = obj.model

    return data, structure, model


def plot_logrefl(objective, axis=None, colour=None, alpha=1, limits=None, plot_lines=True, plot_data=True, plot_labels=True, ymult=1):
    """
    Plots the log reflectivity. If function recieves an axis object, will plot
    on existing axis. Otherwise will plot in a new figure.


    parameters:
    -----------
    objective:  a refnx.analysis.objective object
    axis:       an matplotlib axis object on which the reflectivity profile will be
                plotted
    colour:     colour of the plotted profiles
    alpha:      alpha of the plotted profiles
    limits:     an array with format [xlow, xhigh, ylow, yhigh]
    plot_data:  if true will plot the reflectivity data contained in the objective
    plot_labels:Whether or not to plot axis labels
    ymult:      Constant to multiply the y axis by
    """
    if axis is None:
        fig, axis = plt.subplots()

    plot_refl(objective, axis=axis, colour=colour, alpha=alpha, limits=limits,
              plot_data=plot_data, plot_lines=plot_lines, scale = 'log', ymult=ymult)

    if plot_labels:
        axis.set_xlabel(r'$Q$')
        axis.set_ylabel(r'$R$')

    axis.set_yscale('log')

    return axis


def plot_rq4refl(objective, axis=None, colour=None, alpha=1, limits=None, plot_lines=True, plot_data=True, plot_labels=True, ymult=1):
    """
    Plots the log reflectivity. If function recieves an axis object, will plot
    on existing axis. Otherwise will plot in a new figure.


    parameters:
    -----------
    objective:  a refnx.analysis.objective object
    axis:       an matplotlib axis object on which the reflectivity profile will be
                plotted
    colour:     colour of the plotted profiles
    alpha:      alpha of the plotted profiles
    limits:     an array with format [xlow, xhigh, ylow, yhigh]
    plot_data:  if true will plot the reflectivity data contained in the objective
    plot_labels:Whether or not to plot axis labels
    ymult:      Constant to multiply the y axis by
    """
    if axis is None:
        fig, axis = plt.subplots()

    plot_refl(objective, axis=axis, colour=colour, alpha=alpha, limits=limits,
              plot_data=plot_data, plot_lines=plot_lines,  scale = 'logRQ4', ymult=ymult)

    if plot_labels:
        axis.set_xlabel(r'$Q$')
        axis.set_ylabel(r'$RQ^4$')

    axis.set_yscale('log')

    return axis


def plot_refl(objective, axis, colour = None, alpha = 1, limits = None, plot_lines=True, plot_data = False, scale='log', ymult=1):
    """
    Internal function for plotting reflectivity
    """

    if isinstance(objective, refnx.analysis.objective.GlobalObjective):
        num_objectives = len(objective.objectives)

        if colour is None or len(colour) != num_objectives:
            colour = [colour]*num_objectives

        if isinstance(alpha, int):
            alpha = [alpha]*num_objectives
        elif len(alpha) != num_objectives:
            print ("Length of alpha does not match number of objectives")
            return

        for obj, col, al in zip(objective.objectives, colour, alpha):
            plot_logrefl(obj, axis=axis, colour = col, alpha = al, limits = limits)
        return

    data, structure, model = unpack_objective(objective)

    if scale == 'log':
        ymod = 1 * ymult
    elif scale == 'logRQ4':
        ymod = (data.x**4) * ymult

    if plot_lines:
        axis.plot(data.x, model(data.x, x_err=data.x_err)*ymod, color=colour, alpha=alpha)

    if plot_data:
        axis.errorbar(data.x, data.y*ymod, yerr=data.y_err*ymod, fmt='none', color=colour, alpha=alpha)

    if limits is not None:
        assert len(limits) == 4, "Must supply limits in format [xlow, xhigh, ylow, yhigh]"
        axis.xlim(limits[0:2])
        axis.ylim(limits[2:])




def plot_SLD(objective, axis=None, colour = 'k', alpha = 1, plot_labels=False):
    if plot_labels:
        axis.set_xlabel(r'z, $\mathrm{\AA}$')
        axis.set_ylabel(r'SLD')

    if isinstance(objective, refnx.analysis.objective.GlobalObjective):
        num_objectives = len(objective.objectives)

        if colour is None or len(colour) != num_objectives:
            colour = [colour]*num_objectives

        if isinstance(alpha, int):
            alpha = [alpha]*num_objectives
        elif len(alpha) != num_objectives:
            print ("Length of alpha does not match number of objectives")
            return

        for obj, col, al in zip(objective.objectives, colour, alpha):
            plot_SLD(obj, axis=axis, colour = col, alpha = al)
        return


    data, structure, model = unpack_objective(objective)

    if axis is None:
        fig, axis = plt.subplots()

    axis.plot(*structure.sld_profile(), color=colour, alpha = alpha)



def plot_VFP(objective, axis=None, colour = 'k', alpha = 1, plot_labels=False):

    if plot_labels:
        axis.set_xlabel(r'z, $\mathrm{\AA}$')
        axis.set_ylabel(r'Volume Fraction')

    if isinstance(objective, refnx.analysis.objective.GlobalObjective):
        num_objectives = len(objective.objectives)

        if colour is None or len(colour) != num_objectives:
            colour = [colour]*num_objectives

        if isinstance(alpha, int):
            alpha = [alpha]*num_objectives
        elif len(alpha) != num_objectives:
            print ("Length of alpha does not match number of objectives")
            return

        for obj, col, al in zip(objective.objectives, colour, alpha):
            plot_VFP(obj, axis=axis, colour = col, alpha = al)
        return


    data, structure, model = unpack_objective(objective)

    if axis is None:
        fig, axis = plt.subplots()

    spline = find_FFVFP (model.structure)
    vfp = spline.profile()

    axis.plot(*vfp, color=colour, alpha = alpha)


def plot_burnplot (objective, chain, burn=None, number_histograms=15, thin_factor=1):
    """
    Constructs plot that enables user to determine if ensemble walkers have
    reached their equilibrium position.

    parameters:
    -----------

    objective:  a refnx.analysis.objective object

    chain:      a numpy.ndarray object with shape [#temperatures, #walkers,
                #steps, #parameters] or [#walkers, #steps, #parameters]

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
    if len(chain.shape) > 3: # Then parallel tempering
        ptchain = chain
        temps = np.flipud(np.linspace(0, ptchain.shape[0]-1, 5).astype(int))
        colours = plt.cm.Blues(temps/np.max(temps))
        chain = chain[0]   # Only use lowest temperature
    else:
        temps = [0]
        colours = 'k'
        ptchain = np.array([chain])

    num_subplot_rows = int(chain.shape[2]) + 1
    fig, ax = plt.subplots(num_subplot_rows,2)

    chain_index = np.linspace(int(0.05*chain.shape[1]), chain.shape[1]-1, number_histograms).astype(int)
    param_index = range(len(objective.varying_parameters()))
    alphas = 0.09 + 0.9*(chain_index - chain_index[0])/float(chain_index[-1] - chain_index[0])

    if burn is None:            # If burn is not supplied then
        burn = chain_index[-1]  # do not plot any as red

    ax[0][1].set_title('LnProb for Post-Burn Samples')
    plot_lnprob_distribution(objective, chain, burn=burn, axis=ax[0][1])

    for pindex, axis in zip(param_index, ax[1:]):

        param = objective.varying_parameters()[pindex]


        plot_walker_trace(param, ptchain[:,:,:,pindex], axis=axis[0], temps=temps,
                          tcolors=colours, thin_factor=thin_factor)

        axis[0].set_title(param.name + ' - value trace')
        axis[1].set_title(param.name + ' - PDF')
        axis[1].set_xlabel('parameter value')
        axis[1].set_ylabel('probability density')


        for cindex, alpha in zip(chain_index, alphas):
            if cindex < burn:
                col = 'k'
            else:
                col = 'r'
            axis[1].hist(chain[:,cindex,pindex], bins=12,  normed=True,
                         histtype='step',alpha=alpha, color=col)
            mod_cindex = thin_factor*cindex
            axis[0].plot([mod_cindex,mod_cindex], [param.bounds.lb, param.bounds.ub],
                         linestyle='dashed', color=col, alpha=alpha)

    return fig

def plot_lnprob_distribution(objective, chain, burn=0, axis=None, colour='k'):
    """
    parameters:
    -----------

    objective:  Refnx Objective Object

    chain:      a numpy.ndarray object with shape [#temperatures, #walkers,
                #steps, #parameters] or [#walkers, #steps, #parameters]

    burn:       Number of steps to remove from chain before processing
    """
    if len(chain.shape) > 3: # PT
        chain = chain [0]

    #burn
    chain = chain[:, burn:, :]
    samples = chain.reshape((-1,chain.shape[2]))
    if axis is None:
        fig, axis = plt.subplots()

    lnprobs = []
    for sample in samples:
        objective.setp(sample)
        lnprobs.append(objective.lnprob())

    axis.hist(lnprobs,  normed=True, histtype='step',alpha=1, color=colour)


def plot_walker_trace(parameter, samples, temps=[0], tcolors=['k'], thin_factor=1,
                      axis=None, legend=False):
    """
    parameters:
    -----------

    parameter:  refnx parameter object of parameter to be plotted

    samples:    parameter values to plot with shape [ntemps, nwalkers, nsteps]
                or [nwalkers, nsteps]]

    temps:      list of temps to plot

    tcolours:   list of colours to use for temps - must be sample length as temps


    axis:       a matplotlib axis object on which the trace profile
                will be plotted.

    thin_factor: If the samples have already been thinned the thin factor can
                be supplied so that the step number will be correct.
    """

    if axis is None:
        fig, axis = plt.subplots()

    if len(samples.shape) == 2: # No parallel tempering
        samples = np.array([samples])
        temps = [0]


    steps = np.linspace(0,thin_factor*(samples.shape[2]-1), samples.shape[2])

    leg = []
    for t, c in zip(temps, tcolors):
        leg.append(mpatches.Patch(color=c, label=('T %d' % t)))
        for samp in samples[t]:
            axis.plot(steps, samp, color=c, alpha=0.2)


    axis.plot(steps, np.ones(steps.shape) * parameter.bounds.lb, color='b')
    axis.plot(steps, np.ones(steps.shape) * parameter.bounds.ub, color='b')
    axis.set_xlabel('step number')
    axis.set_ylabel('parameter value')

    if legend:
        leg.reverse()
        axis.legend(handles=leg, loc='lower center', ncol=5)



def plot_quantile_profile(x_axes, y_axes, axis=None, quantiles=[68,95,99.8], color='k', fullreturn=False):
    """
    Turn an ensembel of profiles into a plot with shaded areas corresponding to distribution quantiles

    parameters:
    -----------
    x_axes: python list-type object of profile x-axes
    y_axes: python list-type object of profile y-axes
    axis:   matplotlib axis object
    """
    max_len = 0
    max_x_axis = []

    if axis==None:
        fig, axis = plt.subplots()

    for x_axis, y_axis in zip(x_axes, y_axes):
        assert len(x_axis) == len(y_axis) , 'x and y axes must be the same length'
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
        print (q_l, q_h)

        y_l = np.percentile(tran_y_axes, q_l, axis=1)
        y_h = np.percentile(tran_y_axes, q_h, axis=1)

        quant_dict[str(q_l) + ' low'] = y_l
        quant_dict[str(q_l) + ' high'] = y_h

        mask =  y_h > 0
        axis.fill_between(x_axis[mask], y_l[mask], y_h[mask], color=color, alpha=0.3)


    y_median = np.median(tran_y_axes, axis=1)
    mask = y_median > 0
    axis.plot(x_axis[mask], y_median[mask], color=color)

    quant_dict['median'] = y_median
    quant_dict['xaxis'] = x_axis

    if fullreturn == True:
        return quant_dict


def plot_corner(objective, samples):
    labels = []
    for i in flatten(objective.parameters.varying_parameters()):
        labels.append(i.name)

    fig = corner.corner(samples, labels=labels, quantiles=[0.025, 0.5, 0.975],
                       show_titles=True, title_kwargs={"fontsize": 12})

    return fig


def plot_cluster_profiles(db, objective, samples, lnprob = None):

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = list(reversed(list(set(labels))))
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(14,10))

    leg_labels = []
    c = 0
    for k, col in zip(unique_labels, colors):
        print (c , ":" , n_clusters_)
        c += 1
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        clustered_samp = samples[class_member_mask]

        plot_profiles (objective, clustered_samp, tuple(col))

        if np.all(lnprob) != None:
            string = 'max local prob  = %.2f' % np.max(lnprob[class_member_mask])
            leg_labels.append(mpatches.Patch(color=col, label=string))

    if np.all(lnprob) != None:
        high_prob = np.max(lnprob)
        high_prob_index =  lnprob == high_prob
        high_prob_index = np.reshape(high_prob_index, -1)
        high_prob_sample = samples[high_prob_index, :]

        plot_profiles (objective, high_prob_sample, colour='red', alpha=1)

        string = 'max global prob = %.2f' % high_prob
        leg_labels.append(mpatches.Patch(color='red', label=string))

        plt.subplot(2,2,4)
        plt.legend(handles=leg_labels)

    return plt.gcf()





### OLD BELOW ###

def process_samples(objective, pvecs, vfp_location=3):
    """
    objective: objective
    """


    if type(objective.model) == refnx.reflect.reflect_model.MixedReflectModel:
        structures = objective.model.structures
    else:
        structures = [objective.model.structure]

    num_structs = len(structures)

    report = {}
    samples = []
    moments = []
    areas = []
    scale_fctrs = []
    ismono = []
    lnprobs = []
    lnpriors = []
    lnlikes = []
    sld_profiles = []
    vfp_profiles = []
    vfp_knots    = []
    ref_profiles = []

    best_lnprob = -1e108
    best_profile = None
    best_area = None
    best_moment = None
    best_lnlike = None
    best_lnprior = None

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

            z, phi, zk, phik = vfp.profile(extra=True, zpoints=250)
            vfp_profiles.append([z, phi])
            vfp_knots.append([zk, phik])

            sld_profile = struct.sld_profile()
            sld_profiles.append(sld_profile)

        if len(structures) > 1:
            scale_fctrs.append(objective.model.scales.pvals)
        else:
            scale_fctrs.append(objective.model.scale.value)

        lnprobs.append(objective.lnprob())
        lnpriors.append(objective.lnprior())
        lnlikes.append(objective.lnlike())

        ref_profile = [objective.data.x, objective.model(objective.data.x, x_err=objective.data.x_err)]
        ref_profiles.append(ref_profile)

        if objective.lnprob() > best_lnprob:
            best_profile = [z, phi, zk, phik]
            best_area = vfp.profile_area()
            best_moment = vfp.moment()
            best_lnprior = objective.lnprior()
            best_lnlike = objective.lnlike()
            best_sld = sld_profile
            best_ref = ref_profile
            best_lnprob = objective.lnprob()


    moments = np.reshape(np.array(moments), (counter, num_structs))
    areas = np.reshape(areas, (counter, num_structs))
    scale_fctrs = np.reshape(scale_fctrs, (counter, num_structs))
    ismono = np.reshape(ismono, (counter, num_structs)).T
    vfp_profiles = np.reshape(vfp_profiles, (counter, num_structs, 2, -1))
    vfp_knots = np.reshape(vfp_knots, (counter, num_structs, 2, -1))
    sld_profiles = np.reshape(sld_profiles, (counter, num_structs, 2, -1))

    #try:
    #    report['evidence - mean'], report['evidence - stdev'] = fitter.sampler.log_evidence_estimate(fburnin=0.8)
    #except AttributeError:
    #    print ('samples not collected with PT, cannot approximate evidence')
    #    report['evidence - mean'], report['evidence - stdev'] = None, None

    report['scale factor/s'] = scale_fctrs

    report['1st moment - mean'] = np.mean(moments)
    report['1st moment - stdev'] = np.std(moments)
    report['1st moment - data'] = np.array(moments)
    report['1st moment - best'] = best_moment

    report['area - mean'] = np.mean(areas)
    report['area - stdev'] = np.std(areas)
    report['area - data'] = np.array(areas)
    report['area - best'] = best_area

    report['lnprob - mean'] = np.mean(lnprobs)
    report['lnprob - stdev'] = np.std(lnprobs)
    report['lnprob - data'] = np.array(lnprobs)
    report['lnprob - best'] = best_lnprob

    report['lnprior - mean'] = np.mean(lnpriors)
    report['lnprior - stdev'] = np.std(lnpriors)
    report['lnprior - data'] = np.array(lnpriors)
    report['lnprior - best'] = best_lnprior

    report['is monotonic'] = ismono

    report['lnlike - mean'] = np.mean(lnlikes)
    report['lnlike - stdev'] = np.std(lnlikes)
    report['lnlike - data'] = np.array(lnlikes)
    report['lnlike - best'] = best_lnlike

    report['vfp - best'] = best_profile
    report['vfp - profiles']  = np.array(vfp_profiles)
    report['vfp - knots']  = np.array(vfp_knots)

    report['sld - best'] = best_sld
    report['sld - profiles'] = np.array(sld_profiles)

    report['refl - best'] = best_ref
    report['refl - profiles'] = np.array(ref_profiles)

    report['parameter samples'] = np.array(samples)
    report['objective'] = objective

    return report


def hist_plot(report, show_prior=False):
    """
    report:
        dictionary generated by process_data
    """
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)

    moment = report['1st moment - data']
    area = report['area - data']
    lnprob = report['lnprob - data']
    lnlike = report['lnlike - data']
    lnprior = report['lnprior - data']
    norm_scales = (report['scale factor/s'].T/np.sum(report['scale factor/s'], axis=1)).T

    num_structs = moment.shape[1]

    for idx in range(num_structs):
        c = prob_color(col_mod=idx)
        ax1.hist(moment[:,idx], density=True, color=c, histtype='step')
        ax2.hist(area[:,idx], density=True, color=c, histtype='step')

    moment = np.reshape(moment, (-1))
    area = np.reshape(area, (-1))
    norm_scales = np.reshape(norm_scales, (-1))

    ax1.hist(moment, density=True, weights=norm_scales, color='k', histtype='step')
    ax2.hist(area, density=True, weights=norm_scales, color='k', histtype='step')

    ax3.hist(lnlike, density=True, histtype='step', color='xkcd:blue',
             linewidth=1, label='likelihood')
    ax3.hist(lnprob, density=True, histtype='step', color='xkcd:purple',
             linewidth=1, label='posterior')

    if show_prior:
        ax4 = ax3.twiny()
        ax4.hist(lnprior, density=True, histtype='step', color='xkcd:red',
                 linewidth=0.5, label='prior')
        ax4.set_xlabel('ln(prior)', color='xkcd:red')

    ax1.set_ylabel('Normalised frequency')
    ax1.set_xlabel('Location of 1st Moment')
    ax2.set_xlabel('VFP Area (true dry layer thickness)')
    ax3.set_xlabel('ln(like), ln(prob)')
    ax1.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)

    ax3.legend(loc='upper left', fontsize='x-small')

    fig.set_size_inches(8, 2.5)
    fig.set_dpi(200)

    return


def graph_plot(objective, pvecs=None, vfp_location=None, plot_knots=False,
               fig=None, ax=None, lnprob_limits=None):
    """
    objective (refnx.objective):
        Objective object to be plotted
    pvecs (iterable):
        Iterable providing (or list containing) array-type with length equal
        to the number of varying parameters in the supplied objective.
        If None will use parameter values already in objective.
    vfp_location (int):
        Location of volume fraction profile object within the model structure
        (objective.model.structure[vfp_location]). If None will not display
        volume fraction profile.
    plot_knots (Boolean):
        If true will show know locations on the volume fraction profile
    fig, ax:
        graph_plot will use supplied fig, ax objects if provided.
    lnprob_limits (list):
        List containing lower and upper limits of the lnprob for the system.
        If provided will set the colour of profiles based on their probability.
    """

    if vfp_location is None:
        fig, [ax2, ax3] = CreateAxes (fig, ax, num_plots=2)
    else:
        fig, [ax1, ax2, ax3] = CreateAxes (fig, ax, num_plots=3)

    # Plot the reflectometry data on the R vs Q plot
    ax3.errorbar(objective.data.x, objective.data.y,
                 yerr=objective.data.y_err, fmt='none', capsize=2,
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
            c = prob_color(objective.lnprob(), lnprob_limits, c_mod)
            if vfp_location is not None:
                z, phi, zk, phik = struct[vfp_location].profile(extra=True)
                ax1.plot(z, phi, color=c, alpha=al)
                if plot_knots:
                    ax1.scatter(zk, phik, color='r', alpha=al)

            ax2.plot(*struct.sld_profile(), color=c, alpha=al)

        c = prob_color(objective.lnprob(), lnprob_limits, 0)
        ax3.plot(objective.data.x, objective.model(objective.data.x, x_err=objective.data.x_err),
                 color=c, alpha=al)

    if lnprob_limits is not None:
        leg_patches = [mpatches.Patch(color=(0,0,0,0), label='lnprob:'),
                       mpatches.Patch(color=prob_color(lnprob_limits[0], lnprob_limits, 0), label='   %d'%lnprob_limits[0]),
                       mpatches.Patch(color=prob_color(lnprob_limits[1], lnprob_limits, 0), label='   %d'%lnprob_limits[1])]

        ax2.legend(handles=leg_patches, fontsize='x-small')

    fig.tight_layout()

    return fig, fig.gca()


def prob_color(lnprob=None, lnprob_bounds=None, col_mod=1):
    """
    lnprob: probability within lnprob_bounds
    lnprob_bounds: upper and lower bounds of lnprobaility
    col_mod: Modifies the colour, making it lighter or darker
    """
    if lnprob_bounds is None:
        return plt.cm.Set1(col_mod/8.0)

    x = (lnprob-lnprob_bounds[0])/(lnprob_bounds[1]-lnprob_bounds[0])


    if x > 0.99:
        x = 0.99
    elif x < 0.01:
        x = 0.01

    col_mod = (col_mod-1)*0.4 + 1
    col = np.array(plt.cm.plasma(x))*col_mod
    col[col>1] = 1
    col[col<0] = 0

    return tuple(col)


def CreateAxes (fig, ax, num_plots):
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

    # Plot the R vs Q plot on a log-log scale
    ax3.set_yscale('log'), ax3.set_xscale('log')

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