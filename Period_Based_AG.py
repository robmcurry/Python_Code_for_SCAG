import pyomo.environ as pyo
from pyomo.environ import value
import pandas as pd
import numpy as np
import xlrd
import xlwt
import xlsxwriter
import xlwings as xw
import random
import time
import math



def build_model1(I,V,V_i,T,M_t,A_all, A ,d,dhat,c):
    
#    Starting the wall time
    start_time = time.time()
    
#    Buidling the Model
    model = pyo.ConcreteModel()
    
#    The y-variables
#    Equal to 1 if attack graph i in I reaches node v in V at time t in T, and 0 otherwise
    model.y = pyo.Var(I, V, T, domain=pyo.Binary)
    
#    This is the time to reach all states
    model.Theta = pyo.Var(domain=pyo.NonNegativeReals)


#    Defining the objective function
    #Minimize the time to reach all nodes
    def obj_rule(model):
        return model.Theta
    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

#    Defines the constraint to create Theta
    def Theta_rule(model, i,v,t):
        return model.Theta >= t*model.y[i,v,t]
    model.Theta_constraint = pyo.Constraint(I,V,T, rule=Theta_rule)

#    Make sure not to exceed budget
    def budget_rule(model, i):
        return sum(c[i,u,v]*model.y[i,v,t] for (u,v) in A[i] for t in T) <= b[i]
    model.budget_constraint = pyo.Constraint(I, rule=budget_rule)

#    Make sure that all nodes are reach by some attacker
    def reach_node_rule(model,v):
        return sum(model.y[i,v,t] for t in T for i in I) == 1
    model.reach_node_constraint = pyo.Constraint(V, rule=reach_node_rule)


#    These are the precedence constraints to make sure that all incoming exploits are completed before a node is reached
    def precedence_rule(model, u, v, i, t):
        
#        Make sure that (u,v) is in A_i
        if (u,v) in A[i]:
            return model.y[i,v,t] <= sum(model.y[k,u,p] for k in I if k != i and (u,v) in A[k] for p in range(1,t-d[k,u,v]) ) + sum(model.y[i,u,p] for p in range(1,t-dhat[i,u,v]) )
#        This is really just a placeholder
        else:
            return model.y[i,v,t] >= 0
    model.precedence_constraint = pyo.Constraint(A_all, I, T,  rule=precedence_rule)

#    Declaring gurobi as the solver
    solver = pyo.SolverFactory('gurobi')
    
#    Sets the time limit to 30 minutes or 1800 seconds
    solver.options['TimeLimit'] = 1800
    

#    Solve the problem
    solver_result = solver.solve(model, tee=True)

#    This captures the end of the run for the time
    runtime = time.time() - start_time
    
#    Outputs the total value/time
    print("Value ", model.obj.expr());
    
#    outputs the solution
    for i in I:
        for v in V:
            for t in T:
                if model.y[i,v,t].value > 0:
                    print("Value of node ", v, " in graph ", i, " during time ", t)

    return model.obj.expr(), runtime


#The set of graphs/attackers
I = [0,1,2]

#The set of all arcs
A_all = [(1,3),(3,4),(2,3),(2,4),(3,5),(4,5)]


#The indexed set of arcs according to each graph
A = [[(1,3),(2,3),(3,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)]]

#The indexed set of nodes according to each graph
V_i = [[1,2,3,4,5],[2,3,4,5],[2,3,4,5]]

#The set of all nodes
V = [1,2,3,4,5]

#The list of all time periods
T = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#dictionary of times/durations
d = {}
#This is the updated value if you reach the node yourself
dhat = {}
#The cost of each arc in each graph
c = {}
for i in I:
    for (u,v) in A[i]:
        d[i,u,v] = random.randint(1, 7)
        dhat[i,u,v] = d[i,u,v] - random.randint(1, 2)
        c[i,u,v] = random.randint(1,3)

#The budget for each graph
b = {}
for i in I:
    b[i] = random.randint(5,10)

#Run the model here
objective_value, model1_time = build_model1(I, V, V_i, T, M_t, A_all, A, d, dhat, c)





