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


def parse_line_by_line(filename):
  """
  This function reads a file line by line, splits each line into words,
  and yields each line as a string.

  Args:
      filename: The name of the file to read.

  Yields:
      A string representing each line in the file.
  """
  with open(filename, 'r') as f:
    for line in f:
      yield line.strip()
#      print("line ", line[0])






#def parse_line_by_line(filename):
#  """
#  This function reads a file line by line, parses each line into words,
#  and yields a list of words for each line.
#
#  Args:
#      filename: The name of the file to read.
#
#  Yields:
#      A list of strings representing the words in each line.
#  """
#  A_test = []
#  linecount = 0;
#  with open(filename, 'r') as f:
#    for line in f:
#        linecount += 1
##        if linecount > 2:
#        # Split the line into words and remove trailing newline
#        words = line.strip().split()
##            A_test.append()
#        print(words[0])
#        yield words



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

            return model.t[v] >= model.t[u] + d[i,u,v]*model.z[i,v] - dbar[i,u,v]*model.w[i,u,v] - 1000*(1-model.z[i,v])
        else:
            return model.w[i,u,v] >= 0
    model.precedence_constraint = pyo.Constraint(A_all, I,  rule=precedence_rule)

             

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
    print("Value ", model.obj.expr());
    
##    outputs the solution
    for i in I:
        for v in V:
            if v in V_i[i]:
                if model.z[i,v].value > 0:
                    print("Value of node ", v, " in graph ", i, " at time ", model.t[v].value)

    return model.obj.expr(), runtime





def build_model1(I,V,V_i,T,M_t,A_all, A ,d,dhat,c,b):
    
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
        
            return model.y[i,v,t] <= sum(model.y[k,u,p] for k in I if k != i for p in range(0,t-d[i,u,v]+1) ) + sum(model.y[i,u,p] for p in range(0,t-dhat[i,u,v]+1) )
#        This is really just a placeholder
        else:
            return model.y[i,v,0] >= 0
    model.precedence_constraint = pyo.Constraint(V, V, I, T,  rule=precedence_rule)

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




##The set of graphs/attackers
#I = [0,1,2,3,4,5,6]
#
##The set of all arcs
#A_all = [(1,3),(3,4),(2,3),(2,4),(3,5),(4,5)]
#
#
##The indexed set of arcs according to each graph
#A = [[(1,3),(2,3),(3,4),(3,5),(4,5)],
#    [(3,4),(2,3),(2,4),(3,5),(4,5)],[(2,3),(2,4),(3,5),(4,5)],[(2,3),(2,4),(3,5),(4,5)],[(2,3),(2,4),(3,5),(4,5)],[(3,4),(2,3),(2,4),(3,5),(4,5)],[(2,3),(2,4),(3,5),(4,5)]]
#
##        ,
##    [(2,3),(2,4),(3,5),(4,5)]
#
##The indexed set of nodes according to each graph
#V_i = [[1,2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5]]
#
##,[2,3,4,5]
#
##The set of all nodes
#V = [1,2,3,4,5]
#
#The list of all time periods
T = range(0,95)
#
##dictionary of times/durations
#d = {}
##This is the updated value if you reach the node yourself
#dhat = {}
##This is the updated value if you reach the node yourself
#dbar = {}
##The cost of each arc in each graph
#c = {}
#for i in I:
#    for (u,v) in A[i]:
#        d[i,u,v] = random.randint(2, 3)
##        print("d value for arc (", u, ",", v, " in graph ", i, " is = ", d[i,u,v])
#        dbar[i,u,v] = 1
#        dhat[i,u,v] = d[i,u,v] - 1
#        c[i,u,v] = random.randint(1,3)
#
##The budget for each graph
##b = {}
##for i in I:
##    b[i] = 8
#
#M_t = 6



# Example usage
# Example usage
# Example usage
# Example usage
        
original_string = "mediumtest1.txt"
I_test = []
V_i_test = []
A_i_test = []
b_test = {}
V_test = []
A_test = []

#dictionary of times/durations
d_test = {}
#This is the updated value if you reach the node yourself
dhat_test = {}
#This is the updated value if you reach the node yourself
dbar_test = {}
#The cost of each arc in each graph
c_test = {}


for i in range(1,11):
#    print(original_string)  # Output: This was is a string

    I_test.append(int(i-1))
    V_i_test.append(int(i-1))
    A_i_test.append(int(i-1))
                
    V_i_test[int(i-1)] = []
    A_i_test[int(i-1)] = []
    linecount = 0
    for line in parse_line_by_line(original_string):
        linecount += 1
        if linecount == 2:
            b_test[int(i-1)] = int(line)
            
            
        if linecount > 2:
            
                        
            A_i_test[int(i-1)].append((int(line.split()[0]), int(line.split()[1])))
            
            if int(line.split()[0]) not in V_i_test[int(i-1)]:
                V_i_test[int(i-1)].append(int(line.split()[0]))
                
            if int(line.split()[1]) not in V_i_test[int(i-1)]:
                V_i_test[int(i-1)].append(int(line.split()[1]))
                
                            
            if int(line.split()[0]) not in V_test:
                V_test.append(int(line.split()[0]))
                
            if int(line.split()[1]) not in V_test:
                V_test.append(int(line.split()[1]))
            
            
            if (int(line.split()[0]), int(line.split()[1])) not in A_test:
                A_test.append((int(line.split()[0]), int(line.split()[1])))
            
#            print("ARC (", line.split()[0], ",", line.split()[1], " c value ", line.split()[2], " d value ", line.split()[3])
            d_test[int(i-1),int(line.split()[0]), int(line.split()[1])] = int(line.split()[3])
            c_test[int(i-1),int(line.split()[0]), int(line.split()[1])] = int(line.split()[2])
            dhat_test[int(i-1),int(line.split()[0]), int(line.split()[1])] = d_test[int(i-1),int(line.split()[0]), int(line.split()[1])] - 1
            dbar_test[int(i-1),int(line.split()[0]), int(line.split()[1])] = 1
    
    original_string = original_string.replace(str(i), str(i+1), 1)
    
#print(d_test)
#print(V_i_test)
#print(b_test)
##print(d_test)
#print(I_test)
#print(V_test)
    
#print(V_i_test)
#Run the model here
objective_value_continuous, model_time_continuous = build_model0(I_test, V_test, V_i_test, T, 130, A_test, A_i_test, d_test, dhat_test, c_test, dbar_test, b_test)

#Run the model here
objective_value_discrete, model_time_discrete = build_model1(I_test, V_test, V_i_test, T, 96, A_test, A_i_test, d_test, dhat_test, c_test,b_test)

print("discrete value = ", objective_value_discrete, " and time ", model_time_discrete)


print("continuous value = ", objective_value_continuous, " and time ", model_time_continuous)



