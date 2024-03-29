# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:59:16 2020

@author: Isaac
"""
import warnings
import objective_processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.patheffects as pe
from refnx.analysis import Objective, Parameter, Parameters, GlobalObjective
import pandas as pd
import copy
from refnx.reflect import SLD
from refnx._lib import flatten
import corner

def graph_plot(report=None, objective=None,
               sld_plot=True, refl_plot=True, vf_plot=False,
               logpost_limits='auto', ystyle='r', xstyle='lin', color=None,
               cbar=False, orientation='v', fig_kwargs=None, offset=1,
               profile_offset=False, SLD_shift=0, 
               flip_sld=False, lpkwrds={}):
    """
    Process an objective, generating a report.

    Parameters
    ----------
    report : objective_processing.objective_report or list of
             objective_processing.objective_report objects
        If objective_processing.objective_report plots a single report. If
        list plots many all reports in the list.
    objective : refnx.objective
        Objective object to be plotted. Will just convert this to a report
        before plotting
    sld_plot : bool, optional
        If true will plot an SLD profile. The default is True.
    refl_plot : bool, optional
        If true will plot an reflectonmetry profile. The default is True.
    vf_plot : bool, optional
        If true will plot an SLD profile - note, there must be a element in the
        structure with a volume fraction component. The default is False.
    logpost_limits : list)
        List containing lower and upper limits of the logpost for the system.
        If provided will set the colour of profiles based on their probability
    ystyle : string, optional
        Style of the reflectometry y-axis. Options are 'r' for normal
        log-scaling, 'rq2' for q^2 scaling, and 'rq4' for rq4 scaling.
        The default is 'r'.
    xstyle : string, optional
        Style of the reflectometry x-axis. Options and 'lin' for a linear
        scale and 'log' for a log scale. The default is 'lin'.
    color : any color format understood by matploblib, matplotlib.pyplot.cm, or
            a list containing any of the preceeding.
        If report or objective is a single entry, then a single colour or
        colormap is required. If report or objective is a  The default is None.
    cbar : bool, optional
        Whether to include a colorbar. This will only work if you have supplied
        a colormap as color. The default is False.
    orientation : string, optional
        'h' for a horizontal layout, 'v' for a vertical layout. The default
        is 'v'.
    fig_kwargs : dictionary, optional
        keyword arguments to be passed to pyplot.subplots() when creating the
        axes. The default is None.
    offset : float, optional
        Value to allow vertical offset of reflectometry data, for use when
        multiple reports or objectives are supplied if you want to vertically
        offset curves a value of 0.01 is reccomended. The default is 1, which
        does not offset curves.
    profile_offset : bool, optional
        Value to allow vertical offset of VFP and SLD profiles. Default False.
    SLD_shift : float, optional
        Distance in angstroms to shift SLD profiles (left or right).
        The default is 0.
    flip_sld : bool, optional
        If true reverses the SLD profile. Default false.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        created figure.
    ax : list
        list of created axes.
    """
    fig, ax = CreateAxes(sld_plot=sld_plot, refl_plot=refl_plot,
                         vf_plot=vf_plot, ystyle=ystyle, xstyle=xstyle,
                         orientation=orientation, fig_kwargs=fig_kwargs)

    if not report and objective:
        if type(objective) == list:
            report = []
            for obj in objective:
                rep = objective_processing.objective_report(obj)
                rep.process_objective()
                report.append(rep)
        else:
            report = objective_processing.objective_report(objective)
            report.process_objective()
    elif not report:
        warnings.warn('No reports or objectives given')
    elif objective:
        warnings.warn('Since you provided a report, nothing is being done\
                      with your supplied objective')

    if type(report) == list:
        reflOS = 1
        num_reports = len(report)
        if 'colormap' in str(type(color)).lower():
            colors = color(np.linspace(start=0.2, stop=0.8, num=num_reports))
        elif type(color) == list:
            colors = color * num_reports
        else:
            colors = [color] * num_reports

        profileOS = 0

        for rep, col in zip(report, colors):
            _report_graph_plot(rep, ax=ax,
                               logpost_limits=logpost_limits, ystyle=ystyle,
                               color=col, cbar=cbar, offset=reflOS,
                               profile_offset=profileOS, flip_sld=flip_sld,
                               SLD_shift=SLD_shift, lpkwrds=lpkwrds)
            reflOS *= offset
            if profile_offset:
                profileOS -= 1

            cbar = False
    else:
        _report_graph_plot(report, ax=ax,
                           logpost_limits=logpost_limits, ystyle=ystyle,
                           color=color, cbar=cbar, flip_sld=flip_sld,
                           SLD_shift=SLD_shift, lpkwrds=lpkwrds)

    return fig, ax


def _report_graph_plot(report, ax, logpost_limits='auto', ystyle='r',
                       xstyle='lin', color=None, cbar=False, offset=1,
                       profile_offset=0, SLD_shift=0, 
                       flip_sld=False, lpkwrds={}):
    """
    Plot a single report on a given set of axes.

    Parameters
    ----------
    report : objective_processing.objective_report
        report to plot.
    ax : matplotlib axis
        axis to plot on.
    logpost_limits : TYPE, optional
        DESCRIPTION. The default is 'auto'.
    ystyle : string, optional
        Style of the reflectometry y-axis. Options are 'r' for normal
        log-scaling, 'rq2' for q^2 scaling, and 'rq4' for rq4 scaling.
        The default is 'r'.
    xstyle : string, optional
        Style of the reflectometry x-axis. Options and 'lin' for a linear
        scale and 'log' for a log scale. The default is 'lin'.
    color : any color format understood by matploblib, or matplotlib.pyplot.cm
        Color to use. If a colurmap is provided plots will be coloured
        according to their probability. The default is None, in which case
        profiles will be black.
    cbar : bool, optional
        Wheter or not to include a colorbar. A colormap must have been provided
        as the color. The default is False.
    offset : float, optional
        Value to allow vertical offset of reflectometry data, for use when
        multiple reports or objectives are supplied if you want to vertically
        offset curves a value of 0.01 is reccomended. The default is 1, which
        does not offset curves.
    profile_offset : float, optional
        Value to allow vertical offset of VFP and SLD profiles
    flip_sld : bool, optional
        If true reverses the SLD profile. Default false.
    """
    name = report.name
    vfps = report.model.vfp
    slds = report.model.sld
    refs = report.ref

    q, r, rerr = report.Qdat, report.Rdat, report.Rdat_err
    logposts = report.logpost

    [axVF, axSLD, axR] = ax

    if ystyle == 'rq4':
        ymult = q**4
    elif ystyle == 'rq2':
        ymult = q**2
    else:
        ymult = 1

    if logpost_limits == 'auto':
        logpost_limits = [np.min(logposts), np.max(logposts)]
    else:
        assert len(list(logpost_limits)) == 2, 'bad format for logpost_limits'

    alpha = np.max([1 / report.num_samples**0.6, 0.001])
    lp = lineplotter(color=color, alpha=alpha, cmap_bounds=logpost_limits,
                     **lpkwrds)

    if axVF:
        plot_profiles(vfps, ax=axVF, line_plotter=lp, cmap_keys=logposts,
                      yoffset=profile_offset, label=name)
    if axSLD:
        pOS = profile_offset * (np.max(slds[0][1]) - np.min(slds[0][1]))
        plot_profiles(slds, ax=axSLD, line_plotter=lp, cmap_keys=logposts,
                      yoffset=pOS, xoffset=SLD_shift, 
                      flip=flip_sld, label=name)
    if axR:
        rerr = clean_log_errors(r, rerr)
        axR.errorbar(q, r * offset * ymult, yerr=rerr * offset * ymult,
                     fmt='none', capsize=2, linewidth=1, color='k', alpha=0.7)
        plot_profiles(refs, ax=axR, line_plotter=lp, cmap_keys=logposts,
                      ymult=ymult * offset, label=name)

    if cbar:
        lp.make_cbar(axR)


def fix_legend_alpha(ax, create_legend=False):
    if create_legend:
        x = ax.legend()
    else:
        x = ax.get_legend()
    for h in x.legendHandles:
        h.set_alpha(1)


def clean_log_errors(y, yerr):
    """
    Remove problemic error bars from plots that will be log-scaled.

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    yerr : TYPE
        DESCRIPTION.

    Returns
    -------
    yerr : TYPE
        DESCRIPTION.

    """
    problem_mask = (y - yerr) <= 0
    yerr[problem_mask] = y[problem_mask] * 0.9

    if np.any(problem_mask):
        warnings.warn('cleaned up errorbar where y - yerr < 0')
    return yerr


def plot_profiles(profiles, ax, line_plotter, cmap_keys, ymult=1, yoffset=0,
                  xoffset=0,label=None, flip=False):
    """
    Iterate through provided profiles and plot them using lineplotter.

    Parameters
    ----------
    profiles : list or array of shape n, 2, z.
         n is the number of profiles, and z is the number of points in each
         profile.
    ax : matplotlib axis
        axis to plot on.
    line_plotter : plottools.lineplotter
        class that handles styling and plotting of lines.
    cmap_keys : list or array of shape n
        will be used to set the color of each line in the profile, if a
        colormap is being used.
    ymult : float, optional
        Vertical offset for the profile (y datapoints will be multiplied by
        this value). The default is 1.
    flip : bool, optional
        Whether to flip the profile along the x axis
    """
    for profile, cmap_key in zip(profiles, cmap_keys):
        x = profile[0]-xoffset
        if flip:
            y = np.flip(profile[1])
        else:
            y = profile[1] * ymult - yoffset
        line_plotter.plot_line(ax, x, y,
                               cmap_key=cmap_key, label=label)
        label = None


class lineplotter (object):
    """
    Handles plotted linestyles.

    Parameters
    ----------
    weight : float, optional
        Line weight. The default is 2.
    color : any color format understood by matploblib, or matplotlib.pyplot.cm
        The color the lines will be. The default is 'k'.
    alpha : float, optional
        The alpha value of the line. The default is 1.
    cmap_bounds : list or tuple, optional
        bounds to use for the colormap. Will depend on what is being used for
        the colormap key. The default is [0, 1].
    """

    def __init__(self, weight=2, color='k', alpha=1, cmap_bounds=[0, 1]):
        self.lw = weight

        if color is None:
            self._col = 'k'
            self._colormap = False
        elif 'colormap' in str(color).lower():
            norm = clrs.Normalize(vmin=cmap_bounds[0], vmax=cmap_bounds[1])
            self._col = plt.cm.ScalarMappable(cmap=color, norm=norm)
            self._colormap = True
        else:
            self._col = color
            self._colormap = False

        self.alpha = alpha

        self._cmap_key = None

    def make_cbar(self, ax):
        """
        Make a colorbar on a given axis.

        Parameters
        ----------
        ax : matplotplib axis
            The axis to plot the colorbar on.
        """
        if self._colormap:
            plt.colorbar(self._col, ax=ax)
        else:
            warnings.warn('Must be using colormap to have a colorbar')

    def _kwargs_dict(self):
        return {'lw': self.lw, 'alpha': self.alpha,
                'color': self.get_color}

    def _merge_kwargs_dicts(self, immediate_dict):
        master_dict = self._kwargs_dict()
        for key in immediate_dict:
            master_dict[key] = immediate_dict[key]
        return master_dict

    @property
    def get_color(self):
        """
        Sanitises self._col, returning a single color.

        Returns
        -------
        a color format understood by matplotlib
        """
        if self._colormap:
            return self._col.to_rgba(self._cmap_key)
        else:
            return self._col

    def plot_line(self, ax, x, y, cmap_key=None, **kwargs):
        """
        Plot a line with characteristics defined by the class.

        Parameters
        ----------
        ax : matplotplib axis
            The axis to plot the line on.
        x : list, np.array
            horizontal axis datapoints.
        y : list, np.array
            vertical axis datapoints.
        cmap_key : float, optional
            If a color map is being used cmap_key is used to determine the
            color of the line. The default is None.
        **kwargs : dict
            Supplies to matplotlib.pyplot.plot. Overrides class settings.
        """
        self._cmap_key = cmap_key
        mkwargs = self._merge_kwargs_dicts(kwargs)
        ax.plot(x, y, **mkwargs)


def CreateAxes(sld_plot=True, refl_plot=True, vf_plot=False,
               orientation='h', ystyle='r', xstyle='lin',
               fig_kwargs=None):
    """
    Create figure and axis objects.

    Parameters
    ----------
    fig : TYPE, optional
        DESCRIPTION. The default is None.
    sld_plot : TYPE, optional
        DESCRIPTION. The default is True.
    refl_plot : TYPE, optional
        DESCRIPTION. The default is True.
    vf_plot : TYPE, optional
        DESCRIPTION. The default is False.
    orientation : TYPE, optional
        DESCRIPTION. The default is 'h'.
    ystyle : TYPE, optional
        DESCRIPTION. The default is 'r'.
    xstyle : TYPE, optional
        DESCRIPTION. The default is 'lin'.
    fig_kwargs : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    to_plot_ls : TYPE
        DESCRIPTION.

    """
    to_plot_ls = [vf_plot, sld_plot, refl_plot]

    if type(sld_plot) is not bool or \
       type(refl_plot) is not bool or\
       type(vf_plot) is not bool:
        assert sld_plot is not True, "Unsure what to do with bool and axis"
        assert refl_plot is not True, "Unsure what to do with bool and axis"
        assert vf_plot is not True, "Unsure what to do with bool and axis"

        AXS = []
        for ax in to_plot_ls:
            if ax is not False:
                AXS.append(ax)

        AXS = np.array(AXS)
        fig = AXS[0].get_figure()

    else:
        num_axes = vf_plot + sld_plot + refl_plot
        if fig_kwargs is None:
            fig_kwargs = {'figsize': (1 + 3 * num_axes, 3), 'dpi': 100,
                          'constrained_layout': True}
        assert num_axes != 0
        if orientation.lower() == 'v':
            fig_kwargs['figsize'] = (4, 1 + 2 * num_axes)
            fig, AXS = plt.subplots(num_axes, 1, **fig_kwargs)
        elif orientation.lower() == 'h':
            fig, AXS = plt.subplots(1, num_axes, **fig_kwargs)
        else:
            raise ValueError('use "v" for vertical layout and "h" for \
                             horizontal layout.')

        if len(AXS) > 1:
            ax_labeler(AXS)

        ax_idx = 0
        AXS = np.atleast_1d(AXS)
        for plt_idx, active_axis in enumerate(to_plot_ls):
            if active_axis:
                to_plot_ls[plt_idx] = AXS[ax_idx]
                ax_idx += 1

        [vf_plot, sld_plot, refl_plot] = to_plot_ls

        if refl_plot:
            refl_plot.set_xlabel('$Q,\ \mathrm{\AA}^{-1}$', labelpad=0.1)
            if ystyle == 'r':
                refl_plot.set_ylabel('$R$')
            elif ystyle == 'rq2':
                refl_plot.set_ylabel('$RQ^2,\ \mathrm{\AA}^{-2}$')
            elif ystyle == 'rq4':
                refl_plot.set_ylabel('$RQ^4,\ \mathrm{\AA}^{-4}$')
            else:
                raise ValueError('use "r", "rq2" or "rq4".')

        if sld_plot:
            sld_plot.set_ylabel('SLD, $\\rm{\\AA}^{-2}$')
            sld_plot.set_xlabel(r'distance from substrate, $\mathrm{\AA}$',
                                labelpad=0.1)

        if vf_plot:
            vf_plot.set_ylabel('volume fraction')
            vf_plot.set_xlabel(r'distance from substrate, $\mathrm{\AA}$',
                               labelpad=0.1)

    if refl_plot:
        refl_plot.set_yscale('log')
        if xstyle == 'log':
            refl_plot.set_xscale('log')

    return fig, to_plot_ls


def ax_labeler(axs, start_label='a', labelpos=(0.02, 0.98)):
    axs = np.ravel(axs)
    label = start_label
    for ax in axs:
        ax.text(*labelpos, s=f'{label})', ha='left', va='top',
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        label = chr(ord(label) + 1)


def save_parameters(name, params=None, objective=None):
    """
    Saves out all the parameters from either an Objective, GlobalObjective or
    Parameters object as a CSV file.

    If the objective is named, it includes this information in the table.

    Its a bit stuffed for dependencies in global objectives, for reasons
    I'm not quite sure of (something to do with inter-object linking)

    """
    assert params is not None or objective is not None, "Must supply either params or objective"

    predf_list = process_parameters(params=params, objective=objective)

    col_names = ['variable name', 'value', 'is varying', 'Std Error', 'lower bound', 'upper bound', 'dependencies']

    padded_col_names = ['']*(len(predf_list[0])-len(col_names)) + col_names

    df = pd.DataFrame(data=predf_list, columns=padded_col_names)

    df.to_csv(name, index=False)



def process_parameters(params=None, objective=None, line_addon=[]):
    predf_list = []

    if params:
        assert objective is None, "Must supply EITHER objective or params"


        for param in params:
            dep_names = ''
            for p in param.dependencies():
                dep_names += p.name
                dep_names += ', '

            if param.vary == True:
                predf_list.append(line_addon + [param.name, param.value, 'True', param.stderr, param.bounds.lb, param.bounds.ub, dep_names])
            else:
                predf_list.append(line_addon + [param.name, param.value, 'False', '', '', '', dep_names])

    if objective:
        assert params is None, "Must supply EITHER objective or params"

        if type(objective) is GlobalObjective:
            for obj in objective.objectives:
                predf_list = predf_list + process_parameters(objective=obj)
        else:
            predf_list = process_parameters(params=objective.parameters.flattened(), line_addon=line_addon + [objective.name])

    return predf_list


def force_vfp(structure, vfp_index, z_offset=0, backside_roughness=True):

    if type(vfp_index) is not list:
        vfp_index = [vfp_index]

    struc = copy.deepcopy(structure)
    struc.solvent = SLD(0)

    for idx, item in enumerate (struc):
        if idx in vfp_index:
            try:
                item.sld.real.value = 1
            except AttributeError:
                item.polymer_sld.real.value = 1
        else:
            item.sld.real.value = 0


    if not backside_roughness:
        struc = struc[min(vfp_index):]

    z, phi = struc.sld_profile()
    z = z + z_offset
    return z, phi


def plot_corner(objective, samples):
    labels = []

    for i in flatten(objective.parameters.varying_parameters()):
        labels.append(i.name)

    fig = corner.corner(samples, labels=labels, quantiles=[0.025, 0.5, 0.975],
                        show_titles=True, title_kwargs={"fontsize": 12})

    return fig
