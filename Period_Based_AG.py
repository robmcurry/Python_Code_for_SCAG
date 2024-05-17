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


def build_model0(I,V,V_i,T,M_t,A_all, A ,d,dhat, c, dbar, b):
    
#    Starting the wall time
    start_time = time.time()
    
#    Buidling the Model
    model = pyo.ConcreteModel()
    
#the t variables
    model.t = pyo.Var(V, domain=pyo.NonNegativeReals)

    model.z = pyo.Var(I,V,domain=pyo.Binary)
    
    model.w = pyo.Var(I,V,V,domain=pyo.NonNegativeReals)

#    This is the time to reach all states
    model.Theta = pyo.Var(domain=pyo.NonNegativeReals)


#    Defining the objective function
    #Minimize the time to reach all nodes
    def obj_rule(model):
        return model.Theta
    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

#    Defines the constraint to create Theta
    def Theta_rule(model, v):
        return model.Theta >= model.t[v]
    model.Theta_constraint = pyo.Constraint(V, rule=Theta_rule)

#    Make sure not to exceed budget
    def budget_rule(model, i):
        return sum(c[i,u,v]*model.z[i,v] for (u,v) in A[i]) <= b[i]
    model.budget_constraint = pyo.Constraint(I, rule=budget_rule)

##    Make sure that all nodes are reach by some attacker
    def reach_node_rule(model,v):
        return sum(model.z[i,v] for i in I if v in V_i[i]) == 1
    model.reach_node_constraint = pyo.Constraint(V, rule=reach_node_rule)

#    These are the precedence constraints to make sure that all incoming exploits are completed before a node is reached
    def precedence_rule(model, u, v, i):

#        Make sure that (u,v) is in A_i
        if (u,v) in A[i]:

            return model.t[v] >= model.t[u] + d[i,u,v]*model.z[i,v]- 1*model.w[i,u,v]  - 1000*(1-model.z[i,v])
        else:
            return model.w[i,u,v] >= 0
    model.precedence_constraint = pyo.Constraint(A_all, I,  rule=precedence_rule)

#

    def w_rule_1(model, u,v, i):
        
#        Make sure that (u,v) is in A_i
        if (u,v) in A[i]:
            return model.w[i,u,v] <= model.z[i,u]
        else:
            return model.w[i,u,v] >= 0
    model.w_rule_1_constraint = pyo.Constraint(A_all, I, rule= w_rule_1)
    
    
    def w_rule_2(model, i, u,v):
        
#        Make sure that (u,v) is in A_i
        if (u,v) in A[i]:
            return model.w[i,u,v] <= model.z[i,v]
        else:
            return model.w[i,u,v] >= 0
    model.w_rule_2_constraint = pyo.Constraint(I, A_all, rule= w_rule_2)
    
    
    def w_rule_3(model, i, u,v):
        
#        Make sure that (u,v) is in A_i
        if (u,v) in A[i]:
            return model.w[i,u,v] >= model.z[i,u] +  model.z[i,v] - 1
        else:
            return model.w[i,u,v] >= 0
    model.w_rule_3_constraint = pyo.Constraint(I, A_all, rule= w_rule_3)
    
    
#    Declaring gurobi as the solver
    solver = pyo.SolverFactory('gurobi')
    
#    Sets the time limit to 30 minutes or 1800 seconds
    solver.options['TimeLimit'] = 1800
    

#    Solve the problem
    solver_result = solver.solve(model, tee=True)

#    This captures the end of the run for the time
    runtime = time.time() - start_time
    
#    Outputs the total value/time
    print("Value ", model.obj.expr())
    print("Time ", runtime)
##    outputs the solution
    for i in I:
        for v in V:
            if v in V_i[i]:
                if model.z[i,v].value > 0:
                    print("Value of node ", v, " in graph ", i, " at time ", model.t[v].value)

    return model.obj.expr(), runtime




def build_model1(I,V,V_i,T,M_t,A_all, A ,d,dhat,c):
    
#    Starting the wall time
    
#    Buidling the Model
    model = pyo.ConcreteModel()
    
    
    list_of_lists_of_lists = []
    for i in I:  # Create 2 sub-lists
        sub_list = []
        for v in V:  # Create 3 inner lists for each sub-list
            if v == 1 or v == 2:
                inner_list = [0]  # Example values for inner list
            if v == 3:
                inner_list = [1,2]
            if v == 4:
                inner_list = [1,2,3,4]
            if v == 5:
                inner_list = [1,2,3,4,5,6]
            sub_list.append(inner_list)
        list_of_lists_of_lists.append(sub_list)

    start_time = time.time()
    
    model.y_extra = pyo.Var(I,V,[k for i in I for v in V for k in list_of_lists_of_lists[i-1][v-1]], domain=pyo.Binary)

    model.Theta = pyo.Var(domain=pyo.NonNegativeReals)


#    Defining the objective function
    #Minimize the time to reach all nodes
    def obj_rule(model):
        return model.Theta
    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

    model.theta_constraints = pyo.ConstraintList()
#    Defines the constraint to create Theta
    for i in I:
        for v in V_i[i]:
            for t in list_of_lists_of_lists[i-1][v-1]:
                model.theta_constraints.add(model.Theta >= t*model.y_extra[i,v,t])
                
                
    model.budget_constraints = pyo.ConstraintList()
#    Make sure not to exceed budget
    for i in I:
        model.budget_constraints.add(sum(c[i,u,v]*model.y_extra[i,v,t] for (u,v) in A[i] for vv in V for t in list_of_lists_of_lists[i-1][vv-1] if v == vv) <= b[i])

#    Make sure that all nodes are reach by some attacker

    model.reach_node_constraints = pyo.ConstraintList()
    for v in V:
        model.reach_node_constraints.add(sum(model.y_extra[i,v,t] for i in I if v in V_i[i] for t in list_of_lists_of_lists[i-1][v-1] ) == 1)


    model.precedence_constraints = pyo.ConstraintList()



    for i in I:
        for (u,v) in A[i]:
            for t in list_of_lists_of_lists[i-1][v-1]:
                model.precedence_constraints.add(model.y_extra[i,v,t] <= sum(model.y_extra[k,u,p] for k in [i] if (u,v) in A[k] for p in list_of_lists_of_lists[k-1][u-1] if p <= t - dhat[i,u,v] ) +  sum(model.y_extra[k,u,p] for k in I if (u,v) in A[k] for p in list_of_lists_of_lists[k-1][u-1] if p <= t - d[k,u,v] ) )

#    Declaring gurobi as the solver
    solver = pyo.SolverFactory('gurobi')
    
#    Sets the time limit to 30 minutes or 1800 seconds
    solver.options['TimeLimit'] = 1800
    

#    Solve the problem
    solver_result = solver.solve(model, tee=True)

#    This captures the end of the run for the time
    runtime = time.time() - start_time
    
#    Outputs the total value/time
    print("Value ", model.obj.expr())
    print("Time ", runtime)
    
#    outputs the solution
    for i in I:
        for v in V:
            for t in list_of_lists_of_lists[i-1][v-1]:
                if model.y_extra[i,v,t].value is not None :
                    if model.y_extra[i,v,t].value > 0:
                        print("Value of node ", v, " in graph ", i, " during time ", t)

    return model.obj.expr(), runtime


#The set of graphs/attackers
I = [0,1,2,3,4,5,6,7,8,9]

#The set of all arcs
A_all = [(1,3),(3,4),(2,3),(2,4),(3,5),(4,5)]


#The indexed set of arcs according to each graph
A = [[(1,3),(2,3),(3,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,4),(1,4)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)],
    [(2,3),(2,4),(3,5),(4,5)]]

#The indexed set of nodes according to each graph
V_i = [[1,2,3,4,5],[2,3,4,5],[1,2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5]]

#The set of all nodes
V = [1,2,3,4,5]

#The list of all time periods
T = range(1,21)

#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#dictionary of times/durations
d = {}
#This is the updated value if you reach the node yourself
dhat = {}
#The cost of each arc in each graph
c = {}
for i in I:
    for (u,v) in A[i]:
        d[i,u,v] = 2
        dhat[i,u,v] = d[i,u,v] - 1
        c[i,u,v] = random.randint(1,3)

#The budget for each graph
b = {}
for i in I:
    b[i] = random.randint(30,60)

M_t = 10
#Run the model here
objective_value, model1_time = build_model1(I, V, V_i, T, M_t, A_all, A, d, dhat, c)

objective_value, model1_time = build_model0(I, V, V_i, T, M_t, A_all, A, d, dhat, c, dhat, b)


#def build_model0(I,V,V_i,T,M_t,A_all, A ,d,dhat, c, dbar, b):



