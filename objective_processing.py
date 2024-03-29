# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:59:16 2020

@author: Isaac
"""
import warnings
import refnx


def process_globalobjective(gobjective, pvecs=None):
    """
    Create a list of reports from a global objective.

    Parameters
    ----------
    gobjective : refnx.analysis.objective.GlobalObjective
        a 'global' objective - that is one that combines multiple obejectives
        so that they can be corefined.
    pvecs : Parameters, list or np.array, optional
        Contains parameter values. Is of shape [x, p] where x is the number of
        samples and p is the number of parameters (either total number of
        parameters or the number of varying parameters. The default is None.

    Returns
    -------
    reports : list
        List containing reports for each objective in gobjective.objectives.

    """
    if pvecs is None:
        pvecs = [gobjective.parameters.flattened()]

    reports = []
    for obj in gobjective.objectives:
        reports.append(objective_report(obj))

    for pvec in pvecs:
        gobjective.setp(pvec)

        for obj, rep in zip(gobjective.objectives, reports):
            rpvec = [obj.parameters.flattened()]
            rep.process_objective(rpvec)

    return reports

class objective_report (object):
    """
    Makes a report.

    The idea behind creating reports is to:
        a) standardise plotting
        b) reduce plotting time by only processing MCMC results once

    Parameters
    ----------
    objective : refnx.analysis.objective.Objective
        DESCRIPTION.
    """

    def __init__(self, objective):
        self.objective = objective
        self.name = objective.name
        self.num_samples = 0
        self.logpost = []
        self.logp = []
        self.logl = []
        self.chisqr = []
        self.pvecs = []
        self.ref = []

        self.Qdat = objective.data.x
        self.Rdat = objective.data.y
        self.Rdat_err = objective.data.y_err
        self.Qdat_err = objective.data.x_err

        self.pvec_names = None

        if type(self.objective.model) is refnx.reflect.reflect_model.ReflectModel:
            self.model = model_report(self.objective.model)
        else:
            self.model = None
            warnings.warn('Not implimented')

    def process_objective(self, pvecs=None):
        """
        Process an objective, generating a report.

        Parameters
        ----------
        pvecs : TYPE, optional
            Parameter sets to interate through. The default is None, in which
            case the current parameter values in self.objective are used.

        Returns
        -------
        None.

        """
        if pvecs is None:
            pvecs = [self.objective.parameters.flattened()]

        for pvec in pvecs:
            self.objective.setp(pvec)
            self._log_values(self.objective)
            self.model._log_values(self.objective.model)
            self.num_samples += 1

    def _log_values(self, objective):
        if self.pvec_names is None:
            self.pvec_names = objective.parameters.names

        self.pvecs.append(objective.parameters.pvals)

        self.ref.append([objective.data.x, objective.generative()])

        self.logpost.append(objective.logpost())
        self.logp.append(objective.logp())
        self.logl.append(objective.logl())
        self.chisqr.append(objective.chisqr())


class model_report (object):
    """
    report for a model, rather than an objective.

    Parameters
    ----------
    objective : refnx.reflect.reflect_model.ReflectModel
        DESCRIPTION.
    """

    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose

        self.scale = []
        self.bkg = []
        self.sld = []
        self.vfp = []
        self.knots = []

        self.freeform_location = self.find_freeform_location()

    def _log_values(self, model):
        assert type(model) is refnx.reflect.reflect_model.ReflectModel
        self.scale.append(model.scale.value)
        self.bkg.append(model.bkg.value)

        # Might regret putting this in later - but unflip the SLD profile
        # by default
        reverse_state = model.structure.reverse_structure
        model.structure.reverse_structure = False
        model.structure.sld_profile()
        self.sld.append(model.structure.sld_profile())

        if self.freeform_location:
            z, phi, kz, kphi = model.structure[self.freeform_location].profile(True)
            self.vfp.append([z, phi])
            self.knots.append([kz, kphi])

        model.structure.reverse_structure = reverse_state

    def find_freeform_location(self):
        """
        Find the index of the FreeformVFP.FreeformVFP object in a structure.

        Returns
        -------
        idx : int
            Index of freeform component.

        """
        for idx, element in enumerate(self.model.structure):
            if 'FreeformVFP' in str(type(element)):
                return idx

        if self.verbose:
            print('No freeform component found in structure')
        return None
