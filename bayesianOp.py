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
        initial_point=None,  # Accept an initial point
        n_initial_points=0,  # Set to 0 since we are giving an initial point
        initial_point_generator="random",
        acq_func="gp_hedge",
        acq_optimizer="auto",
        y0=None,
        random_state=None,
        verbose=False,
        callback=None,
        n_points=10000,
        n_restarts_optimizer=5,
        xi=0.01,
        kappa=1.96,
        noise="gaussian",
        n_jobs=1,
        model_queue_size=None,
        space_constraint=None,
    ):
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
        
        # Initialize optimizer with the provided initial point
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
        )
        
        # Check initial point
        self.x0 = []
        if initial_point is not None:
            self.x0 = [initial_point]  # Store the provided initial point
        
        # self.y0 = y0 if y0 is not None else []
        # if self.x0 and not self.y0:
        #     self.y0 = list(map(func, self.x0))
        
        # # Record provided point
        # if self.x0:
        #     self.optimizer.tell(self.x0, self.y0)
        
        self.best_x = None
        self.best_y = np.inf  # Initialize with a large value
        
        self.verbose = verbose
        self.callbacks = check_callback(callback)

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
        
        result.specs = {"args": locals(), "function": "base_minimize"}
        if eval_callbacks(self.callbacks, result):
            return result
        return None
    
    def getResult(self):
        """Retrieve the best result found during optimization."""
        return self.best_x, self.best_y

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