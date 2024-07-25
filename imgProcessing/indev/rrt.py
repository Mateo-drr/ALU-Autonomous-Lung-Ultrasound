# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:33:27 2024

@author: Mateo-drr
"""

'''
import filtering as filt

data  = filt.loadmat('C:/Users/Mateo-drr/Documents/SSD/ALU/d1')
'''

#'''
import numpy as np
from scipy.optimize import minimize

class RRT:
    def __init__(self, objective_function, bounds, max_iter=1000, step_size=0.1):
        self.objective_function = objective_function
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.dim = len(bounds)
        self.tree = []
        self.best_solution = None
        self.best_value = float('inf')

    def sample(self):
        return np.array([np.random.uniform(low, high) for low, high in self.bounds])

    def nearest(self, sample):
        return min(self.tree, key=lambda node: np.linalg.norm(node - sample))

    def steer(self, from_node, to_node):
        direction = to_node - from_node
        length = np.linalg.norm(direction)
        if length < self.step_size:
            return to_node
        direction = direction / length * self.step_size
        return from_node + direction

    def optimize(self):
        initial_node = self.sample()
        self.tree.append(initial_node)
        self.best_solution = initial_node
        self.best_value = self.objective_function(initial_node)

        for _ in range(self.max_iter):
            random_sample = self.sample()
            nearest_node = self.nearest(random_sample)
            new_node = self.steer(nearest_node, random_sample)
            self.tree.append(new_node)
            value = self.objective_function(new_node)

            if value < self.best_value:
                self.best_value = value
                self.best_solution = new_node

        # Refine with local optimization
        #result = minimize(self.objective_function, self.best_solution, bounds=self.bounds)
        #self.best_solution = result.x
        #self.best_value = result.fun

        return self.best_solution, self.best_value

# Example usage
def example_function(x):
    return np.sum(x**2)

bounds = [(-5, 5) for _ in range(2)]
rrt = RRT(example_function, bounds, max_iter=1000, step_size=0.1)
best_solution, best_value = rrt.optimize()

print("Best solution:", best_solution)
print("Best value:", best_value)
#'''