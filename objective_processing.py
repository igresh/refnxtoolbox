# -*- coding: utf-8 -*-
import refnx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import pickle


def plot_reports(reports, refl_spacing=10, refl_mode='log',
                 colors=['r', 'g', 'b']):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(8, 2.5)
    fig.set_dpi(200)
    offset = 1

    if type(reports) == list:
        for report, c in zip(reports, colors):
            offset = _graph_plot(report, fig, ax, offset=offset, colors=c,
                                 refl_spacing=refl_spacing,
                                 refl_mode=refl_mode)
    else:
        _graph_plot(reports, fig, ax, refl_spacing=refl_spacing,
                    refl_mode=refl_mode)


def _graph_plot(report, fig, ax, offset=1, refl_spacing=10, colors=None,
                refl_mode='log'):
    """
    colors - Either a color recognised by matplotlib or a color
    map recognised by matplotlib.
    refl_mode: 'log or rq4, changes y axis plotting for refl plot
    """

    vfps = report.get_vfp_profiles()
    slds = report.get_sld_profiles()
    refls = report.get_refl_profiles()
    lnprobs = report.get_probs()['lnprobs']

    [ax1, ax2, ax3] = ax

    # Coordinates for plotting color gradients
    rect = np.array([0.4, 1, 0.58, 0.09])

    if colors is None:
        colors = ['autumn', 'winter', 'cool']
    else:
        colors = [colors] # This will be problematic

    for obj_key, c_name in zip(vfps, colors):
        obj_lnprobs = lnprobs[obj_key]
        lnprob_limits = [np.min(obj_lnprobs), np.max(obj_lnprobs)]

        try:  # If colour maps are supplied, then they are used
            col = plt.get_cmap(c_name)
            rect[1] -= 0.1
            ax1_I = add_subplot_axes(ax1, rect)
            plot_color_gradients(col, ax1_I)
            ypos = rect[1] + 0.5*rect[3] - 0.005

            xpos1 = rect[0]
            xpos2 = rect[0]+rect[2]

            l_left = '%0.0f' % lnprob_limits[0]
            l_right = '%0.0f' % lnprob_limits[1]

            ax1_I.text(xpos1, ypos, l_left, horizontalalignment='left',
                       verticalalignment='center', transform=ax1.transAxes)
            ax1_I.text(xpos2, ypos, l_right, horizontalalignment='right',
                       verticalalignment='center', transform=ax1.transAxes)


        except ValueError:  # Otherwise flat colours are used
            col = c_name    # if user suplies flat color ('r')
        except TypeError:
            col = c_name

        obj_vfps = vfps[obj_key]
        obj_slds = slds[obj_key]
        refl     = refls[obj_key]
        alpha = 1/np.sqrt(len(refl['R']))

        for struct_key, ls in zip(obj_vfps, ['-',':']):
            vfp = obj_vfps[struct_key]
            sld = obj_slds[struct_key]
            for z, vf, lnp in zip(vfp['z'], vfp['vf'], obj_lnprobs):
                c = prob_color(lnp, lnprob_limits, col_map=col)
                ax1.plot(z, vf, color=c, linestyle=ls, alpha=alpha)

            for z, sld, lnp in zip(sld['z'], sld['sld'], obj_lnprobs):
                c = prob_color(lnp, lnprob_limits, col_map=col)
                ax2.plot(z, sld, color=c, linestyle=ls, alpha=alpha)

        if refl_mode == 'rq4':
            scale = offset*refl['data Q']**4
        else:
            scale = offset

        ax3.errorbar(refl['data Q'], refl['data R']*scale,
                     yerr=refl['data R err']*scale, fmt=',', color='k')
        for R, lnp in zip(refl['R'], obj_lnprobs):
            c = prob_color(lnp, lnprob_limits, col_map=col)
            ax3.plot(refl['Q'], R*scale, color=c, alpha=alpha)

        offset /= refl_spacing

    ax3.set_yscale('log')

    return offset


def plot_color_gradients(cmap, ax):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



class report (object):
    """
    Object for processing and storing data related refnx objectives. Can handle
    both multiple objectives (in a GlobalObjective) and multiple structures
    (in a MixedReflectModel) simultaneously.

    Data is added to the report by calling the process_objective method and
    supplying an objective and (optionally) an array of objective parameters.
    """
    def __init__(self):
        self.vfp_location = None

        self.lnprob = []
        self.lnprior = []
        self.lnlike = []
        self.chisqr = []
        self.pvecs = []

        self.obj_reports = None

    def process_objective (self, master_objective, pvecs=None):
        if self.obj_reports is None:
            self.obj_reports = []
            if type(master_objective) == refnx.analysis.objective.GlobalObjective:
                self.objectives = master_objective.objectives
            else:
                self.objectives = [master_objective]

            self.num_objectives = len(self.objectives)

            for i in range(self.num_objectives):
                self.obj_reports.append(objective_report())

        if pvecs is None:
            pvecs = [master_objective.varying_parameters().pvals]

        for pvec in pvecs:
            master_objective.setp(pvec)
            self.lnprob.append(master_objective.logpost)
            self.lnprior.append(master_objective.logp)
            self.lnlike.append(master_objective.logl)
            self.chisqr.append(master_objective.chisqr)
            self.pvecs.append(pvec)
            for obj, obj_report in zip(self.objectives, self.obj_reports):
                obj_report.add_sample(obj)

    def get_refl_profiles (self):
        assert self.obj_reports is not None ,\
        'you must have processed some samples to produce profiles'

        refl_profiles = {}
        for obj_report in self.obj_reports:
            refl_profiles[obj_report.name] = obj_report.get_refl_profile()

        return refl_profiles

    def get_vfp_profiles (self):
        assert self.obj_reports is not None ,\
        'you must have processed some samples to produce profiles'

        vfp_profiles = {}
        for obj_report in self.obj_reports:
            vfp_profiles[obj_report.name] = obj_report.get_vfp_profiles()

        return vfp_profiles

    def get_sld_profiles (self):
        assert self.obj_reports is not None ,\
        'you must have processed some samples to produce profiles'

        sld_profiles = {}
        for obj_report in self.obj_reports:
            sld_profiles[obj_report.name] = obj_report.get_sld_profiles()

        return sld_profiles

    def get_probs (self):
        assert self.obj_reports is not None ,\
        'you must have processed some samples to produce profiles'

        probs = {}
        probs['lnprobs']  = {}
        probs['lnlikes']  = {}
        probs['lnpriors'] = {}

        probs['lnprobs']['master objective']  = self.lnprob
        probs['lnlikes']['master objective']  = self.lnlike
        probs['lnpriors']['master objective'] = self.lnprior

        for obj_report in self.obj_reports:
            probs['lnprobs'][obj_report.name]  = obj_report.lnprob
            probs['lnlikes'][obj_report.name]  = obj_report.lnlike
            probs['lnpriors'][obj_report.name] = obj_report.lnprior

        return probs

    def save_profiles (self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        vfps = self.get_vfp_profiles()
        slds = self.get_sld_profiles()
        refls = self.get_refl_profiles()

        for obj_key in slds:
            obj_vfps = vfps[obj_key]
            obj_slds = slds[obj_key]
            refl     = refls[obj_key]

            refl_arr = [refl['data Q'], refl['data Q err'],
                        refl['data R'], refl['data R err']]
            refl_header = 'Q, Q error, measured R, measured R error'

            for i, R in enumerate(refl['R']):
                refl_arr.append(R)
                refl_header += ', model R %d'%i

            with open('%s/%s_refl.csv'%(dir_name, obj_key), 'wb') as fh:
                np.savetxt(fh, np.transpose(refl_arr), delimiter=',', header=refl_header)

            for struct_key in obj_vfps:
                vfp = obj_vfps[struct_key]
                sld = obj_slds[struct_key]

                vfp_z, vfp_vfs  = unify_xaxes(vfp['z'], vfp['vf'])
                sld_z, sld_slds = unify_xaxes(sld['z'], sld['sld'])

                vfp_arr = np.concatenate(([vfp_z] , vfp_vfs))
                sld_arr = np.concatenate(([sld_z], sld_slds))

                vfp_header = 'z, vf'
                sld_header = 'z, sld'

                with open('%s/%s_%s_vfps.csv'%(dir_name, obj_key, struct_key), 'wb') as fh:
                    np.savetxt(fh, np.transpose(vfp_arr), delimiter=',', header=vfp_header)

                with open('%s/%s_%s_slds.csv'%(dir_name, obj_key, struct_key), 'wb') as fh:
                    np.savetxt(fh, np.transpose(sld_arr), delimiter=',', header=sld_header)



class objective_report (object):
    def __init__(self, name=None):
        self.name = name

        self.lnprob = []
        self.lnprior = []
        self.lnlike = []
        self.chisqr = []
        self.refl_Q = None
        self.refl_R = []
        self.data_Q = None
        self.data_R = None
        self.data_Qerr = None
        self.data_Rerr = None
        self.num_objectives = 1

        self.struct_reports = []


    def add_sample (self, objective):
        self.lnprob.append(objective.logpost())
        self.lnprior.append(objective.logp())
        self.lnlike.append(objective.logl())
        self.chisqr.append(objective.chisqr())

        if self.refl_Q is None:  # Has not run yet
            if self.name is None:
                self.name = str(objective.name)

            self.refl_Q = objective.data.x
            self.data_Q = objective.data.x
            self.data_R = objective.data.y
            self.data_Qerr = objective.data.x_err
            self.data_Rerr = objective.data.y_err

            # Account for Mixed Area models
            if type(objective.model) == refnx.reflect.reflect_model.MixedReflectModel:
                self.structures = objective.model.structures
            else:  # Just one Structure
                self.structures = [objective.model.structure]

            self.num_structures = len(self.structures)

            for i in range(self.num_structures):
                self.struct_reports.append(structure_report(name='structure %d'%i))

        self.refl_R.append(objective.model(objective.data.x,
                                           x_err=objective.data.x_err))

        for struct, struct_report in zip(self.structures, self.struct_reports):
            struct_report.add_sample(struct)

    def get_refl_profile(self):
        refl_profile = {}
        refl_profile['Q'] = self.refl_Q
        refl_profile['R'] = self.refl_R
        refl_profile['data Q'] = self.data_Q
        refl_profile['data R'] = self.data_R
        refl_profile['data Q err'] = self.data_Qerr
        refl_profile['data R err'] = self.data_Rerr

        return refl_profile

    def get_vfp_profiles(self):
        vfp_profiles = {}
        for struct_report in self.struct_reports:
            vfp_profiles[struct_report.name] = struct_report.get_vfp_profile()

        return vfp_profiles

    def get_sld_profiles(self):
        sld_profiles = {}
        for struct_report in self.struct_reports:
            sld_profiles[struct_report.name] = struct_report.get_sld_profile()

        return sld_profiles


class structure_report (object):
    def __init__(self, name='structure'):
        self.name = name
        self.moments = []
        self.areas = []
        self.scale_fctrs = []
        self.ismono = []

        self.sld_z = []
        self.sld_sld = []

        self.vfp_z = []
        self.vfp_phi = []
        self.vfp_zk = []
        self.vfp_phik = []

        self.vfp_location = None

    def add_sample(self, structure):
        if self.vfp_location is None:
            self.vfp_location = find_vfp(structure)

        if self.vfp_location is not None:
            vfp = structure[self.vfp_location]
            self.moments.append(vfp.moment())
            self.areas.append(vfp.adsorbed_amount.value)
            self.ismono.append(is_monotonic(vfp))

            z, phi, zk, phik = vfp.profile(extra=True)

            self.vfp_z.append(z)
            self.vfp_phi.append(phi)
            self.vfp_zk.append(zk)
            self.vfp_phik.append(phik)

        z, sld = structure.sld_profile()

        self.sld_z.append(z)
        self.sld_sld.append(sld)


    def get_vfp_profile(self):
        vfp_profile = {}
        vfp_profile['z'] = self.vfp_z
        vfp_profile['vf'] = self.vfp_phi

        return vfp_profile

    def get_sld_profile(self):
        sld_profile = {}
        sld_profile['z'] = self.sld_z
        sld_profile['sld'] = self.sld_sld

        return sld_profile





def find_vfp(structure):
    for i, element in enumerate(structure):
        try:
            element.profile()
            return i
        except AttributeError:
            x = 1  # do nothing

def is_monotonic(vfp):
    mono = True
    for vff in vfp.vff:
        if vff.value > 1:
            mono = False
            return mono

    return mono

def prob_color(lnprob=None, lnprob_bounds=None, col_map=plt.cm.plasma):
    """
    lnprob: probability within lnprob_bounds
    lnprob_bounds: upper and lower bounds of lnprobaility
    """
    if type(col_map) is not matplotlib.colors.LinearSegmentedColormap and\
        type(col_map) is not matplotlib.colors.ListedColormap:
            return col_map
    if lnprob_bounds[0] == lnprob_bounds[1]:
        x = 0.99
    else:
        x = (lnprob-lnprob_bounds[0])/(lnprob_bounds[1]-lnprob_bounds[0])

    if x > 0.99:
        x = 0.99
    elif x < 0.01:
        x = 0.01

    col = np.array(col_map(x))
    col[col > 1] = 1
    col[col < 0] = 0

    return tuple(col)


def unify_xaxes(xs, ys, numpoints=500):
    """
    Uses interpolation to place all y values onto a single x-axis, where the
    maximum x value is defined by max x and the number of data points on the
    axis is defined by numpoints.
    """
    max_x = 0
    for x in xs:
        if np.max(x) > max_x:
            max_x = np.max(x)

    new_x = np.linspace(0, max_x, numpoints)
    new_ys = np.zeros([len(ys), numpoints])

    for index, [x, y] in enumerate(zip(xs, ys)):
        new_ys[index, :] = np.interp(new_x, x, y)

    return new_x, new_ys


def pretty_ptemcee(fitter, nsamples, nthin, name=None, save=True):
        objective = fitter.objective

        if name is None:
            name = objective.name

        print('fitting: %s' % name)

        for i in range(nsamples):
            fitter.sample(1, nthin=nthin)

            average_lnprob = np.mean(fitter.logpost[:, 0], axis=1)

            if fitter.chain.shape[0] > 1:
                diff = np.diff(average_lnprob)/nthin
            else:
                diff = [0]

            t = time.strftime('%d/%m %H:%M: ')

            print("%s %d/%d - lnprob: %d  dlnprob/dstep: %0.2f" %
                  (t, i+1, nsamples, average_lnprob[-1], diff[-1]))

            if save:
                pickle.dump(fitter, open(name + '_fitter.pkl', 'wb'))


def darwinize (fitter, lnprob_cut=None):
    """
    Kills off low lnprob walkers, replacing them with beautiful clones high lnprob walkers.
    """
    if lnprob_cut:
        assert lnprob_cut < fitter.lnprob[-1,0].max() , 'There are no samples that qualify!'
        mask = np.greater_equal(fitter.lnprob[-1,0].T, lnprob_cut).T
    else:
        mask = np.greater_equal(fitter.lnprob[-1,0].T, np.mean(fitter.lnprob[-1,0], axis=0)).T


    target_size = fitter.chain.shape[2]
    select_samps = fitter.chain[-1][:,mask]
    new_pos = select_samps
    while new_pos.shape[1] < target_size:
        new_pos = np.concatenate((new_pos, select_samps), axis=1)
        print(new_pos.shape)

    new_pos = new_pos[:, :target_size]

    print(new_pos.shape)

    print('replacing chain')
    fitter._state.coords = new_pos

    return fitter
