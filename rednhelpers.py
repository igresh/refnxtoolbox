
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import os.path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import refnx.dataset
import refnx.reduce as r


def qcrit(rho1, rho2, picky=False):
    """
    Calculate the Q-value of the critical edge.

    Calculates the Q-value of the critical edge in a reflectometry
    experiment. The order the two phases is not important in the
    calculation unless the parameter `picky = True` (but of course
    the ordering of the phases in the experiment is important).

    If `rho1` and `rho2` are specified in units of A^-2 then the
    return value will be in units of A^-1.

    Parameters
    ----------
    rho1 : float
        scattering length density of the incident phase
    rho2 : float
        scattering length density of the reflecting phase
    picky : bool, optional
        If `True`, then no critical edge will be calculated unless
        `rho1` < `rho2`, reflecting the physical reality of the
        experiment; default value is `False` for convenience.

    Returns
    -------
    Qc : float
        Q-value of the critical edge.
    """
    if picky and rho2 < rho1:
        return 0.

    return np.sqrt(16. * np.pi * np.abs(rho2-rho1))


def trim_data(ds, qmin, qmax):
    """
    Trim data set down to a specified Q range.

    Returns a new ReflectDataset initialised with views of the Q array
    and the reflected data that are restricted to a given Q-range.

    Parameters
    ----------
    ds : ReflectDataset
        data set to trim
    qmin : float
        minimum Q value to keep
    qmax : float
        maximum Q value to keep

    Returns
    -------
    ds_trimmed : ReflectDataset
        New ReflectDataset object with data as a view (not deep copy)
        of the original data.
    """
    q = ds.data[0]
    mask = np.logical_and(q >= qmin, q <= qmax)
    return refnx.dataset.ReflectDataset((ds.data[0][mask],
                                         ds.data[1][mask],
                                         ds.data[2][mask],
                                         ds.data[3][mask]))


def check_scale_factor(dataset, qc, name=None, show=True, 
                       calc_max=0.9, plot_max=1.1):
    """ check the scale factor below the critical edge """

    try:
        ds = dataset.ds
        name = dataset.name
        if not ds:
            return
    except AttributeError:
        ds = dataset

    tds = trim_data(ds, 0, qc * plot_max)
    calc_boundary = qc * calc_max
    cds = trim_data(ds, 0, calc_boundary)
    scale = np.mean(cds.data[1])

    print("Apparent scale: %5.3f" % scale)

    if show:
        fig, ax = plt.subplots(1, 1)
        ax.set(xlabel='$Q$//A', ylabel='$Q$')

        ax.axvline(qc, color='r')
        ax.text(qc * 1.01, scale * 2., "Qc")

        ax.plot(tds.data[0], tds.data[1])
        ax.set_yscale('log')

        ax.axhline(scale, color='g', ls="--")
        ax.axhline(1.0, color='k', ls="-")
        ax.axvline(calc_boundary, color='k', ls=':')

        if name:
            fig.suptitle(name)
    return scale


def check_scale_factors(datasets, qc, **kwargs):
    """kwargs:
        name=None
        show=True
        calc_max=0.9
        plot_max=1.1
        """
    scales = []
    for d in datasets:
        print(d.name, end='\t')
        scales.append(check_scale_factor(d, qc, **kwargs))
    return scales


def apply_scale_factors(datasets, qc=None, scales=None, **kwargs):
    if scales is None:
        scales = check_scale_factors(datasets, qc, **kwargs)
    for d, s in zip(datasets, scales):
        print("Rescaling %s by %f" % (d.name, s))
        d.ds.scale(s)


def plot_data_sets(data, qmax=None, title=None, offset=1, divisor=1,
                  fig=None, axs=None, cmap=None):
    """
    Plot reflectometry data sets

    Plots side-by-side log R vs Q and RQ^4 vs Q for all the data sets
    provided.

    Parameters
    ----------
    data : list
        The data sets to be plotted. Must be either:
        list of `(ReflectDataset, label)` tuples or
        list of `ReductionEntry` objects.
        If `(ReflectDataset, label)` tuple is provided then the `label`
        is used in the legend.
    qmax : float, optional
        If specified, sets the maximum $Q$ for the plot
    title : str, optional
        If specified, added to the top of the plot as the suptitle.
    offset : float, optional
        Factor applied to each plot to offset curves.
    divisor : float, optional
        Factor applied to all curves being plotted.
    fig : matplotlib Figure, optional
        Plot Figure onto which curves will be plotted
    axs : list or tuple of matplotlib axes
        Axes onto which the curves will be plotted
    cmap : str
        Name of a matplotlib colormap to apply to the data sets
    """
    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    else:
        (ax1, ax2) = axs

    factor = divisor

    if cmap:
        colormap = matplotlib.cm.get_cmap(cmap)
        colorcycle = colormap(np.linspace(0.2, 0.8, len(data)))
        ax1.set_prop_cycle('color', colorcycle)
        ax2.set_prop_cycle('color', colorcycle)
    
    for dataset in data:
        try:
            ds = dataset.ds
            label = dataset.name
        except AttributeError:
            ds, label = dataset
            #raise ValueError("Don't know how to deal with dataset", dataset)

        try:
            ax1.errorbar(ds.data[0], ds.data[1] * factor,
                         yerr=ds.data[2] * factor,
                         marker='.', label=label)

            #ax2.plot(ds.data[0], ds.data[1] * ds.data[0]**4)
            ax2.errorbar(ds.data[0], ds.data[1] * factor * ds.data[0]**4,
                         yerr=(ds.data[2] * factor * ds.data[0]**4),
                         marker='.', label=label)

            factor /= offset
        except AttributeError:
            print("Warning: data set was skipped: %s\n%s" % (dataset.name, str(dataset)))

    ax1.set_yscale('log')
    if qmax is not None:
        ax1.set_xlim(0, qmax)
    ax1.set(xlabel='$Q$//A', ylabel='$R$')

    ax2.set_yscale('log')
    if qmax is not None:
        ax2.set_xlim(0, qmax)
    ax2.set(xlabel='$Q$//A', ylabel='$RQ^4$')

    if title is not None and fig is not None:
        fig.suptitle(title)

    # FIXME: this can crash if there is no data to plot
    ax1.legend(loc='upper right')
    return fig, (ax1, ax2)

