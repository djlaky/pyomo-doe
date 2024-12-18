#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.common.dependencies import numpy as np

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import json
import logging
from pathlib import Path
import pandas as pd

from tclab_pyomo import (
    TC_Lab_data,
    TC_Lab_experiment,
    extract_results,
    extract_plot_results,
    results_summary,
)

number_tclab_states = 2

# Seeing if D-optimal experiment matches for both the
# greybox objective and the algebraic objective
def compare_reactor_doe():
    # Read in file
    DATA_DIR = Path(__file__).parent
    file_path = DATA_DIR / ".." / "data" / "tclab_sine_test_5min_period.csv"

    df = pd.read_csv(file_path)

    skip = 6

    # Create data object considering control points in data file
    tc_data = TC_Lab_data(
        name="Sine Wave Test for Heater 1, 5 minute period",
        time=df["Time"].values[::skip],
        T1=df["T1"].values[::skip],
        u1=df["Q1"].values[::skip],
        P1=200,
        TS1_data=None,
        T2=df["T2"].values[::skip],
        u2=df["Q2"].values[::skip],
        P2=200,
        TS2_data=None,
        Tamb=df["T1"].values[0],
    )

    # Theta estimates from parmest notebook
    theta_values = {
        'Ua': 0.041705,
        'Ub': 0.0094407,
        'inv_CpH': 0.165909,
        'inv_CpS': 5.835756,
    }

    # Create a ReactorExperiment object; data and discretization information are part
    # of the constructor of this object
    experiment = TC_Lab_experiment(data=tc_data, theta_initial=theta_values, number_of_states=number_tclab_states)

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-2

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    # gather prior information
    doe_obj = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        tee=True,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    FIM = doe_obj.compute_FIM(method="sequential")

    # Create the DesignOfExperiments object
    # We will not be passing any prior information in this example
    # and allow the experiment object and the DesignOfExperiments
    # call of ``run_doe`` perform model initialization.
    doe_obj = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=FIM,
        jac_initial=None,
        fim_initial=FIM,
        L_diagonal_lower_bound=1e-7,
        #solver=solver,
        tee=True,
        get_labeled_model_args=None,
        #logger_level=logging.ERROR,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    # Begin optimal DoE
    ####################
    doe_obj.run_doe()

    # Print out a results summary
    #print("Optimal experiment values: ")
    #print(
    #    "\tInitial concentration: {:.2f}".format(
    #        doe_obj.results["Experiment Design"][0]
    #    )
    #)
    #print(
    #    ("\tTemperature values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
    #        *doe_obj.results["Experiment Design"][1:]
    #    )
    #)
    print("FIM at optimal design:\n {}".format(np.array(doe_obj.results["FIM"])))
    print(
        "Objective value at optimal design: {:.2f}".format(
            pyo.value(doe_obj.model.objective)
        )
    )

    #print(doe_obj.results["Experiment Design Names"])

    ###################
    # End optimal DoE

    # Begin optimal grey box DoE
    ############################
    doe_obj_grey_box = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        use_grey_box_objective=True,  # New object with grey box set to True
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        #prior_FIM=np.array(doe_obj.results["FIM"]),
        prior_FIM=FIM,
        jac_initial=None,
        fim_initial=np.array(doe_obj.results["FIM"]),
        L_diagonal_lower_bound=1e-7,
        #solver=solver,
        tee=True,
        get_labeled_model_args=None,
        #logger_level=logging.ERROR,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    doe_obj_grey_box.run_doe()
    # Print out a results summary                                                                                                                      
    #print("Optimal experiment values: ")
    #print(
    #    "\tInitial concentration: {:.2f}".format(
    #        doe_obj_grey_box.results["Experiment Design"][0]
    #    )
    #)
    #print(
    #    ("\tTemperature values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
    #        *doe_obj_grey_box.results["Experiment Design"][1:]
    #    )
    #)
    print("FIM at optimal design:\n {}".format(np.array(doe_obj_grey_box.results["FIM"])))
    print(
        "Objective value at optimal design: {:.2f}".format(
            pyo.value(doe_obj_grey_box.model.objective)
        )
    )
    print("Raw logdet: {:.2f}".format(np.log10(np.linalg.det(np.array(doe_obj_grey_box.results["FIM"])))))

    #print(doe_obj_grey_box.results["Experiment Design Names"])

    # Print out a results summary
    #print("Optimal experiment values: ")
    #print(
    #    "\tInitial concentration: {:.2f}".format(
    #        doe_obj.results["Experiment Design"][0]
    #    )
    #)
    #print(
    #    ("\tTemperature values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
    #        *doe_obj.results["Experiment Design"][1:]
    #    )
    #)
    print("FIM at optimal design:\n {}".format(np.array(doe_obj.results["FIM"])))
    print(
        "Objective value at optimal design: {:.2f}".format(
            pyo.value(doe_obj.model.objective)
        )
    )

    #print(doe_obj.results["Experiment Design Names"])

    
if __name__ == "__main__":
    compare_reactor_doe()
