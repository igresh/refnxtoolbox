import numpy as np
import matplotlib.pyplot as plt
from refnx._lib import flatten
import matplotlib.patches as mpatches
import corner
import refnx
import sys
sys.path.append("/mnt/1D9D9A242359B87C/Git Repos/refnx/examples/analytical_profiles/brushes/")
sys.path.append("/Users/Isaac/Documents/GitRepos/refnx/examples/analytical_profiles/brushes/")

import brush

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
        axis.set_ylabel(r'log$(R)$')
    
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
        axis.set_ylabel(r'log$(RQ^4)$')
        
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
        axis.errorbar(data.x, data.y*ymod, yerr=data.y_err*ymod, marker='.', color=colour, alpha=alpha)

    if limits is not None:
        assert len(limits) == 4, "Must supply limits in format [xlow, xhigh, ylow, yhigh]"
        axis.xlim(limits[0:2])
        axis.ylim(limits[2:])
        

        

def plot_SLD(objective, axis=None, colour = 'k', alpha = 1, plot_labels=False):
    if plot_labels:
        axis.set_xlabel(r'z, $\AA$')
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
        axis.set_xlabel(r'z, $\AA$')
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
    
    if len(chain.shape) > 3: # Then parallel tempering
          chain = chain[0]   # Only use lowest temperature

    num_subplot_rows = int(chain.shape[2])
    fig, ax = plt.subplots(num_subplot_rows,2)

    chain_index = np.linspace(int(0.05*chain.shape[1]), chain.shape[1]-1, number_histograms).astype(int)
    param_index = range(len(objective.varying_parameters()))
    alphas = 0.09 + 0.9*(chain_index - chain_index[0])/float(chain_index[-1] - chain_index[0])

    if burn is None:            # If burn is not supplied then
        burn = chain_index[-1]  # do not plot any as red 
        
    for pindex, axis in\
        zip(param_index, ax):
        
        param = objective.varying_parameters()[pindex]

         
        plot_walker_trace(param, chain[:,:,pindex], axis=axis[0],
                          thin_factor=thin_factor)
        
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

        
def plot_walker_trace(parameter, samples, temps=[0], tcolors=['k'], thin_factor=1,
                      axis=None):
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
        leg.append(mpatches.Patch(color=c, label=('temp: %d' % t)))
        for samp in samples[t]:
            axis.plot(steps, samp, color=c, alpha=0.2)
    
    
    axis.plot(steps, np.ones(steps.shape) * parameter.bounds.lb, color='b')
    axis.plot(steps, np.ones(steps.shape) * parameter.bounds.ub, color='b')
    axis.set_xlabel('step number')
    axis.set_ylabel('parameter value')
    axis.legend(handles=leg, loc='lower left')

        
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

def process_data(fitter, burn=0, flatten = False, thin=0):
    """
    fitter.process_chain returns an inconvienient data structure
    """
    if len(fitter.sampler.chain.shape) == 3: # No PT
        samples = fitter.sampler.chain[:,burn:]
        lnprob = fitter.sampler.lnprobability[:,burn:]
        if flatten:
            samples = samples.reshape(-1, samples.shape[2])
            samples = samples[0::thin]
            lnprob = lnprob.reshape(-1, 1)
            lnprob = lnprob[0::thin]
    else: # Then PT
        chain = fitter.sampler.chain
        chain = chain.reshape((chain.shape[0]*chain.shape[1], chain.shape[2], chain.shape[3]))
        
        lnprob = fitter.sampler.lnprobability
        lnprob = lnprob.reshape((lnprob.shape[0]*lnprob.shape[1], lnprob.shape[2]))
        
        samples = chain[:,burn:]
        lnprob = lnprob[:,burn:]
        if flatten:
            samples = samples.reshape(-1, samples.shape[2])
            samples = samples[0::thin]
            lnprob = lnprob.reshape(-1, 1)
            lnprob = lnprob[0::thin]
    return samples, lnprob


def TempSort (objectives, Temps):
    objectives = [x for _,x in sorted(zip(Temps,objectives))]
    Temps = sorted(Temps)   
    Temps = np.array(Temps)
    return objectives, Temps

def pltpub_SLD(ax, objectives, colours=[], markers=[], z=None, fontsize=14, title='VFP profile'):
    
    if markers == []:
        markers = ['solid']*len(objectives)
    if colours == []:
        colours = [plt.cm.set2(each) for each in np.linspace(0, 1, len(objectives))]
    
    if np.all(z) == None:
        z = np.linspace(0,1200, 1000)
    

    for obj, col, marker in zip (objectives, colours, markers):
        sld = obj.model.structure.sld_profile()
        ax.plot(sld[0], sld[1], color=col, linestyle=marker)

    
    return ax
    
def pltpub_VFP(ax, objectives, colours=[], markers=[], z=None, fontsize=14, title='VFP profile'):
    if markers == []:
        markers = ['solid']*len(objectives)
    if colours == []:
        colours = [plt.cm.set2(each) for each in np.linspace(0, 1, len(objectives))]

    if np.all(z) == None:
        z = np.linspace(0,1200, 10000)
    
    # Areas = []

    for obj, col, marker in zip (objectives, colours, markers):
        spline = find_FFVFP (obj.model.structure)
        vfp = spline.volume_fraction_profile(z)
        mask = vfp > 0
        ax.plot(z[mask], vfp[mask], color=col, linestyle=marker)

            

            
    #leg_areas = []
    #for L, A in zip(leg, areas):
    #    leg_areas.append(('%s   %.1fnm' % (L, A/10)))

    #ax.legend(leg_areas)
          
    #fig = plt.gcf()
    #plt.show()
    
    return ax
    
def pltpub_logrefl(ax, objectives, colours=[], markers=[],  z=None,  max_q=0.3, offset=100, show_data=True):
    if markers == []:
        markers = ['solid']*len(objectives)
    if colours == []:
        colours = [plt.cm.set2(each) for each in np.linspace(0, 1, len(objectives))]
        
    
    scale = 1

    for obj, col, marker in zip (objectives, colours, markers):
        data, structure, model = unpack_objective(obj)
        q_mask = data.x < max_q
        x = data.x[q_mask]
        y = data.y[q_mask]
        x_err = data.x_err[q_mask]
        y_err = data.y_err[q_mask]
            
        ax.plot(x, model(x, x_err=x_err)/scale, color=col, linestyle = marker)
        if show_data:
            ax.errorbar(x, y/scale, yerr=y_err/scale, marker='.', color=col, alpha=0.25)
        scale *= offset


    ax.set_yscale('log')
    ax.axes.yaxis.set_ticklabels([])
    ticks = 10**(-np.linspace(0,np.log10(offset)*len(objectives)+4,2*len(objectives)+5))
    ax.set_yticks(ticks) 


    
    return ax
    
    
def pltpub_rq4refl(ax, objectives, colours=[], markers=[],  z=None,  max_q=0.3, offset=100, show_data=True):
    if markers == []:
        markers = ['solid']*len(objectives)
    if colours == []:
        colours = [plt.cm.set2(each) for each in np.linspace(0, 1, len(objectives))]
        
    
    scale = 1
    
    for obj, col, marker in zip (objectives, colours, markers):
        data, structure, model = unpack_objective(obj)
        q_mask = data.x < max_q
        x = data.x[q_mask]
        y = data.y[q_mask]
        x_err = data.x_err[q_mask]
        y_err = data.y_err[q_mask]
            
        ax.plot(x, model(x, x_err=x_err)*(x**4)/scale, color=col, linestyle = marker)
        if show_data:
            ax.errorbar(x, y*(x**4)/scale, yerr=y_err*(x**4)/scale, marker='.', color=col, alpha=0.25)
        scale *= offset


    #plt.xlabel('Scattering Vector, ' + r'$Q$' + r' $(\AA^{-1})$')
    #plt.ylabel(r'$RQ^4$')
    ax.set_yscale('log')

    ax.axes.yaxis.set_ticklabels([])
    ticks = 10**(-np.linspace(7,7+np.log10(offset)*len(objectives)+1,2*len(objectives)+2))
    ax.set_yticks(ticks) 
    
    return ax
    
    
    

def plot_publishable(objectives, labels, z=None, max_q=0.2, Temps=[]):
    if len(Temps) == 0:
        colours = [plt.cm.plasma(each)
          for each in np.linspace(0, 1, len(objectives))]
    else:
        assert len(Temps) == len(labels), "you have supplied %d temperatures and %d labels." % (len(Temps), len(labels))
        
        objectives = [x for _,x in sorted(zip(Temps,objectives))]
        Temps = sorted(Temps)   
        Temps = np.array(Temps)
        scaledTemps = (Temps-10)/40 #assuming maximum temperature we'll get is 50 and minimum is 10
        colours = [plt.cm.plasma(each)
          for each in np.linspace(min(scaledTemps), max(scaledTemps), len(objectives))]
            
    if np.all(z) == None:
        z = np.linspace(0,1200, 10000)
        
    marker = 'solid'
    for obj, col in zip (objectives, colours):
        sld = obj.model.structure.sld_profile()
        plt.plot(*sld, color=col, linestyle=marker)
        if marker == 'solid':
            marker = 'dashed'
        else:
            marker = 'solid'
    
    if len(Temps) == 0:
        LA = []
        label_length = max_len(labels)
        plt.legend(labels)
    else:
        TA = []
        for T in Temps:
            TA.append(('%.1f°C ' % T))
        fig=plt.gcf()
        ax=fig.gca()
        plt.text(0.815, 0.96, '         Area', transform=ax.transAxes)
        plt.legend(TA,loc='upper right', bbox_to_anchor=(0.999, 0.97))
        
    plt.title("SLD Profile")
    plt.xlabel('z (Angstrom)')
    plt.ylabel('SLD Profile')

    
    areas = []
    marker = 'solid'
    for obj, col in zip (objectives, colours):
        spline = find_FFVFP (obj.model.structure)
        areas.append(spline.profile_area())
        vfp = spline.volume_fraction_profile(z)
        mask = vfp > 0
        plt.plot(z[mask], vfp[mask], color=col, linestyle=marker)
        if marker == 'solid':
            marker = 'dashed'
        else:
            marker = 'solid'
    
    if len(Temps) == 0:
        LA = []
        label_length = max_len(labels)
        for L, A in zip(labels, areas):
            LA.append((L + ' '*(label_length - len(L)) + ' %d.0 A' % A))
        plt.legend(LA)
    else:
        TA = []
        for T, A in zip(Temps, areas):
            print 
            TA.append(('%.1f°C   %.1fnm' % (T, A/10)))
        fig=plt.gcf()
        ax=fig.gca()
        plt.text(0.815, 0.96, 'Temp     Area', transform=ax.transAxes)
        plt.legend(TA,loc='upper right', bbox_to_anchor=(0.999, 0.97))
        
        
    plt.title("Volume Fraction Profile")
    plt.xlabel('z (Angstrom)')
    plt.ylabel('Volume Fraction Profile')
        
    plt.show()
    
    plt.figure(figsize=(14, 14))
    

    plt.subplot(1,2,1)
    offset = 1
    marker = 'solid'
    for obj, col in zip (objectives, colours):
        data, structure, model = unpack_objective(obj)
        q_mask = data.x < max_q
        x = data.x[q_mask]
        y = data.y[q_mask]
        x_err = data.x_err[q_mask]
        y_err = data.y_err[q_mask]
            
        plt.plot(x, model(x, x_err=x_err)/offset, color=col, linestyle = marker)
        plt.errorbar(x, y/offset, yerr=y_err/offset, marker='.', color=col)
        offset *= 100
        
        if marker == 'solid':
            marker = 'dashed'
        else:
            marker = 'solid'

    plt.xlabel('Q')
    plt.title("Log Reflectivity")
    plt.ylabel(r'$\mathrm{log}(R)$')
    plt.yscale('log')
    plt.gca().axes.yaxis.set_ticklabels([])
    ticks = 10**(-np.linspace(0,2*len(objectives)+4,2*len(objectives)+5))
    plt.gca().set_yticks(ticks) 
    
    plt.subplot(1,2,2)
    
    offset = 1
    marker = 'solid'
    for obj, col in zip (objectives, colours):
        data, structure, model = unpack_objective(obj)
        q_mask = data.x < max_q
        x = data.x[q_mask]
        y = data.y[q_mask]
        x_err = data.x_err[q_mask]
        y_err = data.y_err[q_mask]

        plt.plot(x, model(x, x_err=x_err)*(x**4)/offset, color=col, linestyle = marker)
        plt.errorbar(x, y*(x**4)/offset, yerr=y_err*(x**4)/offset, marker='.', color=col)
        offset *= 100
        
        if marker == 'solid':
            marker = 'dashed'
        else:
            marker = 'solid'
    
    plt.xlabel('Q')
    plt.ylabel(r'$RQ^4$')
    plt.yscale('log')
    plt.title(r"$RQ^4$ Reflectivity")
    plt.gca().axes.yaxis.set_ticklabels([])
    ticks = 10**(-np.linspace(7,7+2*len(objectives)+1,2*len(objectives)+2))
    plt.gca().set_yticks(ticks) 
    

        
def max_len(L):
    ml = 0
    for i in L:
        if len(i) > ml:
            ml = len(i)
    return ml
        
def find_FFVFP (struct):
    for i in struct:
        if isinstance(i, brush.FreeformVFP):
            return i
    return None