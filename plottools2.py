# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:59:16 2020

@author: Isaac
"""
import warnings
import objective_processing2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs


def graph_plot(report=None, objective=None, sld_plot=True, refl_plot=True,
               vf_plot=False, fig=None, ax=None,
               logpost_limits='auto', ystyle='r', xstyle='lin', color=None,
               cbar=False, orientation='v', fig_kwargs=None, offset=1):
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
    sld_plot : TYPE, optional
        DESCRIPTION. The default is True.
    refl_plot : TYPE, optional
        DESCRIPTION. The default is True.
    vf_plot : TYPE, optional
        DESCRIPTION. The default is False.
    plot_knots : TYPE, optional
        DESCRIPTION. The default is False.
    fig : matplotlib figure
        graph_plot will use supplied fig object if provided. If None will
        generate its own fig and ax objects
    ax : matplotlib axis
        graph_plot will use supplied axis object if provided. If None will
        generate its own fig and ax objects
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


    Returns
    -------
    fig : matplotlib.pyplot.figure
        created figure.
    ax : list
        list of created axes.
    """
    fig, ax = CreateAxes(fig=fig, sld_plot=sld_plot, refl_plot=refl_plot,
                         vf_plot=vf_plot, ystyle=ystyle, xstyle=xstyle,
                         orientation=orientation, fig_kwargs=fig_kwargs)

    if not report and objective:
        if type(objective) == list:
            report = []
            for obj in objective:
                rep = objective_processing2.objective_report(obj)
                rep.process_objective()
                report.append(rep)
        else:
            report = objective_processing2.objective_report(objective)
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

        for rep, col in zip(report, colors):
            _report_graph_plot(rep, plot_knots=False, fig=fig, ax=ax,
                               logpost_limits=logpost_limits, ystyle=ystyle,
                               color=col, cbar=cbar, offset=reflOS)
            reflOS *= offset
    else:
        _report_graph_plot(report, plot_knots=False, fig=fig, ax=ax,
                           logpost_limits=logpost_limits, ystyle=ystyle,
                           color=color, cbar=cbar)

    return fig, ax


def _report_graph_plot(report, plot_knots=False, fig=None, ax=None,
                       logpost_limits='auto', ystyle='log', xstyle='lin',
                       color=None, cbar=False, offset=1):
    """
    

    Parameters
    ----------
    report : TYPE
        DESCRIPTION.
    plot_knots : TYPE, optional
        DESCRIPTION. The default is False.
    fig : TYPE, optional
        DESCRIPTION. The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    logpost_limits : TYPE, optional
        DESCRIPTION. The default is 'auto'.
    ystyle : TYPE, optional
        DESCRIPTION. The default is 'log'.
    xstyle : TYPE, optional
        DESCRIPTION. The default is 'lin'.
    color : TYPE, optional
        DESCRIPTION. The default is None.
    cbar : TYPE, optional
        DESCRIPTION. The default is False.
    offset : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
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
    lp = lineplotter(color=color, alpha=alpha, cmap_bounds=logpost_limits)

    axR.errorbar(q, r * offset * ymult, yerr=rerr * offset * ymult,
                 fmt='none', capsize=2, linewidth=1, color='k', alpha=0.7)

    if axVF:
        plot_profiles(vfps, ax=axVF, line_plotter=lp, cmap_keys=logposts)
    if axSLD:
        plot_profiles(slds, ax=axSLD, line_plotter=lp, cmap_keys=logposts)
    if axR:
        plot_profiles(refs, ax=axR, line_plotter=lp, cmap_keys=logposts,
                      ymult=ymult, offset=offset)

    if cbar:
        lp.make_cbar(axR)

    return fig

def plot_profiles(profiles, ax, line_plotter, cmap_keys, offset=1, ymult=1):
    for profile, cmap_key in zip(profiles, cmap_keys):
        line_plotter.plot_line(ax,
                               profile[0],
                               profile[1] * offset * ymult,
                               cmap_key=cmap_key)


class lineplotter (object):
    def __init__(self, weight=2, color='k', alpha=1, cmap_bounds=[0,1]):
        self.lw = weight

        if color is None:
            self._col = 'k'
            self._colormap = False
        elif 'colormap' in str(color).lower():
            norm = clrs.Normalize(vmin=cmap_bounds[0],vmax=cmap_bounds[1])
            self._col = plt.cm.ScalarMappable(cmap=color, norm=norm)
            self._colormap = True
        else:
            self._col = color
            self._colormap = False

        self.alpha = alpha

        self._cmap_key = None

    def make_cbar(self, ax):
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
        if self._colormap:
            return self._col.to_rgba(self._cmap_key)
        else:
            return self._col

    def plot_line(self, ax, x, y, cmap_key=None, **kwargs):
        self._cmap_key = cmap_key
        mkwargs = self._merge_kwargs_dicts(kwargs)
        ax.plot(x, y, **mkwargs)


def CreateAxes(fig=None, sld_plot=True, refl_plot=True, vf_plot=False,
               orientation='h', ystyle='r', xstyle='lin',
               fig_kwargs=None):

    to_plot_ls = [vf_plot, sld_plot, refl_plot]

    if type(sld_plot) is not bool or type(refl_plot) is not bool or type(vf_plot) is not bool:
        assert sld_plot is not True
        assert refl_plot is not True
        assert vf_plot is not True

    else:
        num_axes = sld_plot + refl_plot + vf_plot
        if fig_kwargs is None:
            fig_kwargs = {'figsize': (1 + 3*num_axes, 3), 'dpi': 100,
                          'constrained_layout': True}
        assert num_axes != 0
        if orientation.lower() == 'v':
            fig_kwargs['figsize'] = (4, 1 + 2*num_axes)
            fig, AXS = plt.subplots(num_axes, 1, **fig_kwargs)
        elif orientation.lower() == 'h':
            fig, AXS = plt.subplots(1, num_axes, **fig_kwargs)
        else:
            raise ValueError('use "v" for vertical layout and "h" for \
                             horizontal layout.')
        ax_idx = 0
        for plt_idx, active_axis in enumerate(to_plot_ls):
            if active_axis:
                to_plot_ls[plt_idx] = AXS[ax_idx]
                ax_idx += 1

    [vf_plot, sld_plot, refl_plot] = to_plot_ls

    if refl_plot:
        refl_plot.set_yscale('log')
        if xstyle == 'log':
            refl_plot.set_xscale('log')
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
        vf_plot.set_ylabel('polymer volume fraction')
        vf_plot.set_xlabel(r'distance from substrate, $\mathrm{\AA}$',
                            labelpad=0.1)

    return fig, to_plot_ls
