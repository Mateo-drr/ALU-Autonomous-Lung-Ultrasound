#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:06:49 2024

@author: mateo-drr
"""

from sklearn.utils import check_random_state
from skopt.utils import cook_estimator, normalize_dimensions, eval_callbacks
import numpy as np
import warnings
import numbers
from skopt.optimizer import Optimizer
try:
    from collections.abc import Iterable
except ImportError:
    from collections.abc import Iterable
from skopt.callbacks import VerboseCallback, check_callback

###############################################################################

class ManualGPMinimize:
    def __init__(
        self,
        func,
        dimensions,
        base_estimator=None,
        n_calls=100, #maximum number of calls to cost function
        # initial_point=None,  # Accept an initial point
        n_initial_points=0,  # Set to 0 since we are giving an initial point
        initial_point_generator="random",
        acq_func="gp_hedge",
        acq_optimizer="auto",
        random_state=None,
        callback=None,
        verbose=False,
        n_points=10000,
        n_restarts_optimizer=5,
        xi=0.01,
        kappa=1.96,
        noise="gaussian",
        n_jobs=1,
        model_queue_size=None,
        space_constraint=None,
    ):
        
        #Setting initial points to none because of cstm class
        x0 = None
        y0 = None
        
        # Initialize parameters and optimizer
        self.rng = check_random_state(random_state)
        self.space = normalize_dimensions(dimensions)
        
        if base_estimator is None:
            base_estimator = cook_estimator(
                "GP",
                space=self.space,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                noise=noise,
            )
        
        # # Initialize optimizer with the provided initial point
        # self.optimizer = Optimizer(
        #     dimensions,
        #     base_estimator,
        #     n_initial_points=n_initial_points,
        #     initial_point_generator=initial_point_generator,
        #     n_jobs=n_jobs,
        #     acq_func=acq_func,
        #     acq_optimizer=acq_optimizer,
        #     random_state=random_state,
        #     model_queue_size=model_queue_size,
        #     space_constraint=space_constraint,
        # )
        
        #Code from base.minimize
        ################
        self.specs = {"args": locals(), "function": "base_minimize"}   
        acq_optimizer_kwargs = {
            "n_points": n_points,
            "n_restarts_optimizer": n_restarts_optimizer,
            "n_jobs": n_jobs,
        }
        acq_func_kwargs = {"xi": xi, "kappa": kappa}
        
        # # Check initial point
        # self.x0 = []
        # if initial_point is not None:
        #     self.x0 = [initial_point]  # Store the provided initial point
            
        if x0 is None:
            self.x0 = []
        elif not isinstance(x0[0], (list, tuple)):
            self.x0 = [x0]
        if not isinstance(self.x0, list):
            raise ValueError("`x0` should be a list, but got %s" % type(self.x0))
        
        #Check if initial points is set or if they were given (xo)
        if n_initial_points <= 0 and not self.x0:
            raise ValueError("Either set `n_initial_points` > 0," " or provide `x0`")
            
        # check y0: list-like, requirement of maximal calls
        if isinstance(y0, Iterable):
            self.y0 = list(y0)
        elif isinstance(y0, numbers.Number):
            self.y0 = [y0]
        elif y0 is None:
            self.y0 = []
            
        required_calls = n_initial_points + (len(self.x0) if not self.y0 else 0)
        if n_calls < required_calls:
            raise ValueError("Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))
        # calculate the total number of initial points
        n_random = n_initial_points
        n_initial_points = n_initial_points + len(self.x0)    
        
        
        # create optimizer class
        self.optimizer = Optimizer(
            dimensions,
            base_estimator,
            n_initial_points=n_initial_points,
            initial_point_generator=initial_point_generator,
            n_jobs=n_jobs,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            random_state=random_state,
            model_queue_size=model_queue_size,
            space_constraint=space_constraint,
            acq_optimizer_kwargs=acq_optimizer_kwargs,
            acq_func_kwargs=acq_func_kwargs,
        )
        
        
        # check x0: element-wise data type, dimensionality 
        assert all(isinstance(p, Iterable) for p in self.x0)
        if not all(len(p) == self.optimizer.space.n_dims for p in self.x0):
            raise RuntimeError(
                "Optimization space (%s) and initial points in x0 "
                "use inconsistent dimensions." % self.optimizer.space
            )
        
        # check callback --> verbose code
        self.callbacks = check_callback(callback)
        if verbose:
            self.callbacks.append(
                VerboseCallback(
                    n_init=len(self.x0) if not self.y0 else 0,
                    n_random=n_random,
                    n_total=n_calls,
                )
            )
            
                                                    
        ################
        

        
        # self.y0 = y0 if y0 is not None else []
        # if self.x0 and not self.y0:
        #     self.y0 = list(map(func, self.x0))
        
        # # Record provided point
        # if self.x0:
        #     self.optimizer.tell(self.x0, self.y0)
        
        # Initialize lists to store tested positions and scores
        self.tested_positions = []
        self.scores = []
        
        self.best_x = None
        self.best_y = np.inf  # Initialize with a large value
        
        # self.callbacks = check_callback(callback)

    def step(self):
        """Performs a single optimization step."""
        next_x = self.optimizer.ask()
        return next_x

    def update(self, next_x, next_y):
        """Updates the optimizer with new observations."""
        result = self.optimizer.tell(next_x, next_y)
        
        # Track the best observed point
        if next_y < self.best_y:
            self.best_y = next_y
            self.best_x = next_x
            
        # Store the tested position and its score
        self.tested_positions.append(next_x)
        self.scores.append(next_y)
        
        # print('here', self.optimizer.models)
        
        # result.specs = {"args": locals(), "function": "base_minimize"}
        result.specs = self.specs
        if eval_callbacks(self.callbacks, result): #keep this line to allow verbose prints
            return None #not used
        return self.optimizer.models, self.optimizer.Xi
    
    def getResult(self):
        """Retrieve the best result found during optimization."""
        return self.best_x, self.best_y

    def get_tested_positions_and_scores(self):
        """Retrieve all tested positions and their corresponding scores."""
        return self.tested_positions, self.scores

'''
# Example usage:
# Define your function and dimensions
def objective_function(x):
    return np.sum(np.array(x) ** 2)

dimensions = [(-5.0, 5.0), (-5.0, 5.0)]

# Start with a specific initial point
initial_point = [1.0, 2.0]

# Initialize manual optimizer with the initial point
optimizer = ManualGPMinimize(objective_function, dimensions, initial_point=initial_point)

# Now you can manually perform optimization steps
for _ in range(10):  # Perform 10 steps manually
    next_x = optimizer.step()  # Get next suggestion
    print(f"Next suggested point: {next_x}")
    
    # Here you can evaluate the cost function externally, e.g.:
    next_y = objective_function(next_x)  # Replace with your function call
    optimizer.update(next_x, next_y)  # Update the optimizer

'''