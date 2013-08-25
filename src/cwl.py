# -*- coding: utf-8 -*-
"""Module cwl.

Produces simulation calculation and figures for the Cooperate With/Without Looking project.

Usage:
	python cwl.py {recalculate?}

Examples:
    python cwl.py 					run using pre-calculated saved data
    python cwl.py recalculate		run with freshly calculated data


@author:	Maxim Zaslavsky <maxim@maximzaslavsky.com>
@author:	Erez Yoeli <eyoeli@gmail.com>

"""

### GENERAL

# system imports
import sys, os
import numpy as np
import matplotlib
matplotlib.use("pdf") # save as PDFs
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import defaultdict
from random import sample as random_sample
from math import floor
import cPickle as pickle

# Choose whether to recalculate or to used saved data

Calculate = False
if __name__ == "__main__":
    try:
        if sys.argv[1] == 'recalculate':
            Calculate = True

    except: # Interactive mode
        pass

output_dir = '../bin/'

print 'Welcome to the CW(O)L Simulations and Figures toolkit.'

#######################################################
#######################################################


# Game theory methods

def are_assumptions_valid(a, b, c1, c2, d, p, w):
    #P1 and P2 prefer a cooperative interaction to no interaction
    statement_1 = a > 0 and b > 0
    #P1 gets short-term gains from defection
    statement_2 = c1 > a and c2 > a
    #P2 2 doesn't want to interact with 1 if he expects 1 to defect in either game.
    statement_3 = b * p + d * (1 - p) < 0 and d * p + b * (1 - p) < 0
    #wlog it is more tempting to defect in state 2.
    statement_4 = c2 > c1
    #all of this must hold
    return statement_1 and statement_2 and statement_3 and statement_4


def get_game_population_1(a, b, c1, c2, d, p, w):
    """
    Game for population 1 of CWL
    """
    if not are_assumptions_valid(a, b, c1, c2, d, p, w):
        raise ValueError("This parameters do not comply with assumptions")
    A = np.empty(shape=(4, 3))
    A[0, 0] = (a * p + a * (1.0 - p)) / (1.0 - w)
    A[0, 1] = (a * p + a * (1.0 - p)) / (1.0 - w)
    A[0, 2] = (a * p + a * (1.0 - p))
    A[1, 0] = (a * p + a * (1.0 - p))
    A[1, 1] = (a * p + a * (1.0 - p)) / (1 - w)
    A[1, 2] = (a * p + a * (1.0 - p))
    A[2, 0] = (a * p + c2 * (1.0 - p))
    A[2, 1] = (a * p + c2 * (1.0 - p)) / (1 - p * w)
    A[2, 2] = (a * p + c2 * (1.0 - p))
    A[3, 0] = (c1 * p + c2 * (1.0 - p))
    A[3, 1] = (c1 * p + c2 * (1.0 - p))
    A[3, 2] = (c1 * p + c2 * (1.0 - p))
    return A


def get_game_population_2(a, b, c1, c2, d, p, w):
    """
    Game for population 2 of CWL
    """
    if not are_assumptions_valid(a, b, c1, c2, d, p, w):
        raise ValueError("This parameters do not comply with assumptions")
    B = np.empty(shape=(4, 3))
    B[0, 0] = (b * p + b * (1.0 - p)) / (1.0 - w)
    B[0, 1] = (b * p + b * (1.0 - p)) / (1.0 - w)
    B[0, 2] = (b * p + b * (1.0 - p))
    B[1, 0] = (b * p + b * (1.0 - p))
    B[1, 1] = (b * p + b * (1.0 - p)) / (1.0 - w)
    B[1, 2] = (b * p + b * (1.0 - p))
    B[2, 0] = (b * p + d * (1.0 - p))
    B[2, 1] = (b * p + d * (1.0 - p)) / (1.0 - p * w)
    B[2, 2] = (b * p + d * (1.0 - p))
    B[3, 0] = (d * p + d * (1.0 - p))
    B[3, 1] = (d * p + d * (1.0 - p))
    B[3, 2] = (d * p + d * (1.0 - p))
    return B.T

# replicator

def __replicator_equation_two_populations(x, t, game1, game2, number__of_strategies_population_1, number__of_strategies_population_2):
    """
    This auxiliary function codes the replicator dynamics step. Typically it is only called from replicator_trajectory_two_populations()
    Parameters
    ----------
    x: ndarray initial state (concatenated from the two populations)
    t: time
    game1: ndarray, game for population 1
    game2: ndarray, game for population 2
    number__of_strategies_population_1: int
    number__of_strategies_population_2: int
    Returns:
    out: ndarray next state (concatenated from the two populations)
    """
    x_population_1 = x[0:number__of_strategies_population_1]
    #the first piece of y corresponds to population 1
    x_population_2 = x[number__of_strategies_population_1:number__of_strategies_population_1 + number__of_strategies_population_2]  # the second piece of y corresponds to population 2
    #First Ay
    fitness_vector_1 = np.dot(game1, x_population_2)
    # and Bx (see equation above)
    fitness_vector_2 = np.dot(game2, x_population_1)
    #Now xAy
    average_fitness_1 = np.dot(x_population_1, fitness_vector_1)
    #And yBx
    average_fitness_2 = np.dot(x_population_2, fitness_vector_2)
    #the next lines correspond to equations 10.5 and 10.6 of Hofbauer and Sigmund (page 116)
    new_population_1 = x_population_1 * (fitness_vector_1 - average_fitness_1)
    new_population_2 = x_population_2 * (fitness_vector_2 - average_fitness_2)
    return np.array(new_population_1.tolist() + new_population_2.tolist())
	
def replicator_trajectory_two_populations(game_matrix_1, game_matrix_2, x_0, y_0, t_vector, **kwargs):
    """
    Computes a replicator trajectory for two populations, given two games, starting points and time vector.
    It uses scipy's odeint.
    Parameters
    ----------
    game_matrix_1: numpy matrix (for population 1)
    game_matrix_2: numpy matrix (for population 2)
    x_0: ndarray
    y_0: ndarray
    t_vector: time array
    Returns
    -------
    out: list

    Examples
    --------
    #TODO: Write examples
    """
    #join initial populations to fit signature of replicator_equation
    start = np.array(x_0.tolist() + y_0.tolist())
    number__of_strategies_population_1 = len(x_0)
    number__of_strategies_population_2 = len(y_0)
    #solve
    soln = odeint(__replicator_equation_two_populations, start, t_vector, args=(game_matrix_1, game_matrix_2, number__of_strategies_population_1, number__of_strategies_population_2), **kwargs)
    return [soln[:, i] for i in xrange(number__of_strategies_population_1 + number__of_strategies_population_2)]

	
def get_random_point_inside_simplex(dimension):
    """
    Returns a vector that sums up to one, where components have been uniformly chosen.
    Parameters:
    ----------
    dimension:int
    """
    exponencial = np.random.exponential(size=dimension)
    exponencial /= np.sum(exponencial, dtype=float)
    return exponencial

def adjusted_solution(a, b, c1, c2, d, p, w, x_0, y_0, max_t, **kwargs):
    """
    Returns a steady state, by ajusting dynamically the step size and total error.
    """
    tolerance = 1e-4
    added_factor_vector = [10.0, 20.0, 50.0, 100.0]
    game_1 = get_game_population_1(a, b, c1, c2, d, p, w)
    game_2 = get_game_population_2(a, b, c1, c2, d, p, w)
    t = np.linspace(0.0, max_t, 2000)
    if x_0 is None or y_0 is None:
        (x_0, y_0) = (get_random_point_inside_simplex(4), get_random_point_inside_simplex(3))
    for added_factor in added_factor_vector:
        sol = replicator_trajectory_two_populations(added_factor + game_1, added_factor + game_2, x_0, y_0, t, atol=tolerance, **kwargs)
        end_point = [sol[i][-1] for i in xrange(0, 7)]
        if np.allclose(sum(end_point), 2.0, atol=tolerance):
            return end_point
    raise ValueError("Numerics: x = {}, y = {}, a = {}, b = {}, c1 = {}, c2 = {}, d = {}, p = {}, w = {}".format(x_0.tolist(), y_0.tolist(), a, b, c1, c2, d, p, w))

	
def determine_outcome(solution):
    tolerance = 1e-3
    if not np.allclose(np.sum(solution), 2.0, atol=tolerance):
        raise ValueError("Probabilities don't add up: {} ".format(solution))
    elif player1_CWOL(solution, atol=tolerance) and player2_sometimes_exits_if_looks_or_defects(solution, atol=tolerance):
        return (1, solution)
    elif player1_alwaysD(solution, atol=tolerance) and (player2_pure_strategy(solution, atol=tolerance) or player2_mixes(solution, atol=tolerance)):
        return (2, solution)
    elif player2_exitifdefect(solution, atol=tolerance) and (player1_CWOL(solution, atol=tolerance) or player1_CWL(solution, atol=tolerance) or player1_CWOL_or_CWL(solution, atol=tolerance)):
        return (3, solution)
    else:
        return (4, solution)
		
def determine_random_outcome(a, b, c1, c2, d, p, w, max_t, **kwargs):
    """
    Starting in a random point tries to determine the outcome, given parameters.
    This is the main function to be called from montecarlo procedures
    """
    x_0 = get_random_point_inside_simplex(4)
    y_0 = get_random_point_inside_simplex(3)
    solution = adjusted_solution(a, b, c1, c2, d, p, w, x_0, y_0, max_t)
    return determine_outcome(solution)
	
def montecarlo(a, b, c1, c2, d, p, w, max_t=300, repetitions=5000):
    """
    Takes samples for a given point in the space. Counting the occurrences
    of different outcomes, and returns them in a dictionary with the
    following indexes:
    1 - Outcome 1
    2 - Outcome 2
    3 - Outcome 3
    4 - No categorized
    """
    ans = defaultdict(int)
    sum_of_solution = np.zeros(7)
    for i in xrange(0, repetitions):
        try:
            outcome, solution = determine_random_outcome(a, b, c1, c2, d, p, w, max_t)
            ans[outcome] = ans[outcome]+1
            sum_of_solution += solution 
        except ValueError, e:
            print e
            ans[5] = ans[5] + 1
    avg_of_solution = sum_of_solution/repetitions
    return (ans, sum_of_solution)

#--------- THEORY CHECKING FUNCTIONS ----------

def is_coop_wihtout_looking_an_equilibrium(a, b, c1, c2, d, p, w):
    return c1*p+c2*(1.0 - p) < a / (1.0 - w)


def is_coop_looking_an_equilibrium(a, b, c1, c2, d, p, w):
    return c2 < a / (1.0 - w)


def number_of_equlibria(a, b, c1, c2, d, p, w):
    CWOL = is_coop_wihtout_looking_an_equilibrium(a, b, c1, c2, d, p, w)
    CWL = is_coop_looking_an_equilibrium(a, b, c1, c2, d, p, w)
    if CWOL and CWL:
        return 3
    elif CWOL or CWOL:
        return 2
    else:
        return 1

#--- classifier functions


def player1_CWOL(solution, atol=1e-3):
    player1_plays_desired_pure_strategy = np.allclose(solution[0], 1.0, atol)
    return player1_plays_desired_pure_strategy


def player1_CWL(solution, atol=1e-3):
    player1_plays_desired_pure_strategy = np.allclose(solution[1], 1.0, atol)
    return player1_plays_desired_pure_strategy


def player1_Cin1(solution, atol=1e-3):
    player1_plays_desired_pure_strategy = np.allclose(solution[2], 1.0, atol)
    return player1_plays_desired_pure_strategy


def player1_alwaysD(solution, atol=1e-3):
    player1_plays_desired_pure_strategy = np.allclose(solution[3], 1.0, atol)
    return player1_plays_desired_pure_strategy


def player1_pure_strategy(solution, atol=1e-3):
    return (player1_CWOL(solution, atol) or player1_CWL(solution, atol) or player1_Cin1(solution, atol) or player1_alwaysD(solution, atol))


def player1_CWOL_or_CWL(solution, atol=1e-3):
    #solution[0:1] is now solution[0:2]
    player1_mixes_CWL_CWOL = np.allclose(np.sum(solution[0:2]), 1.0, atol)
    return player1_mixes_CWL_CWOL and not player1_pure_strategy(solution, atol)


def player1_mixes(solution, atol=1e-3):
    #solution[0:3] is now solution[0:4]
    player1_mixes = np.allclose(np.sum(solution[0:4]), 1.0, atol)
    return player1_mixes and not player1_pure_strategy(solution, atol)


def player2_exitiflook(solution, atol=1e-3):
    player2_plays_desired_pure_strategy = np.allclose(solution[4], 1.0, atol)
    return player2_plays_desired_pure_strategy


def player2_exitifdefect(solution, atol=1e-3):
    player2_plays_desired_pure_strategy = np.allclose(solution[5], 1.0, atol)
    return player2_plays_desired_pure_strategy


def player2_alwaysexit(solution, atol=1e-3):
    player2_plays_desired_pure_strategy = np.allclose(solution[6], 1.0, atol)
    return player2_plays_desired_pure_strategy


def player2_pure_strategy(solution, atol=1e-3):
    return (player2_exitifdefect(solution, atol=1e-3) or player2_exitiflook(solution, atol=atol) or player2_alwaysexit(solution, atol=atol))


def player2_mixes(solution, atol=1e-3):
    #solution[4:6] is now changed to solution[4:7], please verify.
    player2_mixes = np.allclose(np.sum(solution[4:7]), 1.0, atol)
    return player2_mixes and not player2_pure_strategy(solution, atol=atol)


def player2_sometimes_exits_if_looks_or_defects(solution, atol=1e-3):
    player2_sometimes_exits_if_looks = not np.allclose(solution[4], 0.0, atol)
    player2_sometimes_exits_if_defects = not np.allclose(solution[5], 0.0, atol)
    return player2_sometimes_exits_if_looks or player2_sometimes_exits_if_defects


# Additioanl plot beautifier functions:

def summarize_binary_list(lista):
    """
    #determines edges of sequences of 1's in a binary list
    """
    ans = []
    x_0 = None
    tamano = len(lista)
    for i in xrange(tamano):
        if lista[i] == 1 and x_0 is None:
            x_0 = i
        end_of_sequence = lista[i] == 0
        end_of_array = i == (tamano-1) and lista[i] == 1
        if (end_of_sequence or end_of_array) and x_0 is not None:
            if end_of_sequence:
                ans.append((x_0, i-1))
            if end_of_array:
                ans.append((x_0, i))
            x_0 = None
    return ans
	

	
#######################################################
#######################################################
	
	
### FIGURE 2 PREPARATION

def clear_past_figs():
    plt.close()
    plt.clf()
    plt.cla()
    plt.close()
    #del f, fig_all
    #gc.collect()

def export_graph(f_i, f_name):
    #f_i.savefig(output_dir+f_name+'.png',dpi=300)
    #f_i.savefig(output_dir+f_name+'.png',dpi=600)
    f_i.savefig(output_dir+f_name+'.pdf', dpi=600) # This one looks the best
    print f_name, 'exported as pdf at 600 dpi.' # 300dpi_png, 600dpi_png, 
	
# Figure 2B and 2C calculations:

print 'Calculating or loading values for Figure 2B and Figure 2C'

p = 0.5 + 0.01
b = 1.0
c1 = 4.0
c2 = 12.0
d = -10.0
w = 7.0/8.0 + 0.02
repetitions = 10000
number_of_points = 50

if Calculate:
    a_interval = np.linspace(0.0+0.1, 2.0, number_of_points, endpoint=False)
    a_interval_tight = np.linspace(0.0+0.1, 2.0, number_of_points) # TODO: change to 300?
    
    #lets plot the theory predictions first as a shade
    calculated_equilibria=[number_of_equlibria(a, b, c1, c2, d, p, w) for a in a_interval_tight]
    one_equilibrium_region = summarize_binary_list([ce == 1 for ce in calculated_equilibria])
    two_equilibria_region = summarize_binary_list([ce == 2 for ce in calculated_equilibria])
    three_equilibria_region = summarize_binary_list([ce == 3 for ce in calculated_equilibria])
    
    #first the sampling
    outcome_1 = []
    outcome_2 = []
    outcome_3 = []
    outcome_4 = []
    no_outcome = []
    strategy_1 = []
    strategy_2 = []
    strategy_3 = []
    strategy_4 = []
    strategy_5 = []
    strategy_6 = []
    strategy_7 = []
    for a in a_interval_tight: # TODO: should this be a_interval?
        diccionario, avg_strategy_frequency = montecarlo(a, b, c1, c2, d, p, w, repetitions=repetitions)
        outcome_1.append(diccionario[1])
        outcome_2.append(diccionario[2])
        outcome_3.append(diccionario[3])
        outcome_4.append(diccionario[4])
        no_outcome.append(diccionario[5])
        strategy_1.append(avg_strategy_frequency[0])
        strategy_2.append(avg_strategy_frequency[1])
        strategy_3.append(avg_strategy_frequency[2])
        strategy_4.append(avg_strategy_frequency[3])
        strategy_5.append(avg_strategy_frequency[4])
        strategy_6.append(avg_strategy_frequency[5])
        strategy_7.append(avg_strategy_frequency[6])
    stuff = [a_interval, a_interval_tight, one_equilibrium_region, two_equilibria_region, three_equilibria_region, outcome_1, outcome_2, outcome_3, outcome_4, no_outcome, strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6, strategy_7]
    pickle.dump( stuff, open( output_dir+"Figure 2_B and C_strategy frequency.saved_data", "wb" ) )
else:
    (a_interval, a_interval_tight, one_equilibrium_region, two_equilibria_region, three_equilibria_region, outcome_1, outcome_2, outcome_3, outcome_4, no_outcome, strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6, strategy_7) = pickle.load(open(output_dir+"Figure 2_B and C_strategy frequency.saved_data", "r"))
	

# Plotting:

clear_past_figs()

def process_ax(ax):
    '''
    Shades figure to correspond to equilibria regions.
    '''
    
    # hack to fill white space in the middle:
    midpoint = (a_interval_tight[one_equilibrium_region[0][1]] + a_interval_tight[two_equilibria_region[0][0]])/2
    midpoint1 = (a_interval_tight[two_equilibria_region[0][1]] + a_interval_tight[three_equilibria_region[0][0]])/2
    
    for dupla in one_equilibrium_region:
        #ax.axvspan(p_interval_tight[dupla[0]], p_interval_tight[dupla[1]], facecolor='red', alpha=0.2)
        ax.axvspan(a_interval_tight[dupla[0]], midpoint, facecolor='white', alpha=1) # red, alpha=0.2
        print 'one', dupla, a_interval_tight[dupla[0]], a_interval_tight[dupla[1]]
    for dupla in two_equilibria_region:
        #ax.axvspan(p_interval_tight[dupla[0]], p_interval_tight[dupla[1]], facecolor='blue', alpha=0.2)
        ax.axvspan(midpoint, midpoint1, facecolor='0.50', alpha=0.2) # blue or .80
        print 'two', dupla, a_interval_tight[dupla[0]], a_interval_tight[dupla[1]]
    for dupla in three_equilibria_region:
        ax.axvspan(midpoint1, a_interval_tight[dupla[1]], facecolor='0.10', alpha=0.2) # yellow or .20
        print 'three', dupla, a_interval_tight[dupla[0]], a_interval_tight[dupla[1]]
		
		


avoid_end = -1 # remove last 1 point

#######################################################
#######################################################

### PLOT FIGURE 2(B): Frequency vs. a-value

print 'Plotting Figure 2B'

clear_past_figs()
f = plt.figure(figsize=(10,10))
process_ax(f.gca())
plt.plot(a_interval[:avoid_end], (np.array(outcome_1)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'bo-', label='Cooperate without looking')
plt.plot(a_interval[:avoid_end], (np.array(outcome_2)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'ro-', label='Always defect')
plt.plot(a_interval[:avoid_end], (np.array(outcome_3)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'yo-', label='Cooperate with looking')
plt.plot(a_interval[:avoid_end], (np.array(outcome_4)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'ko-', label='Other')
plt.grid()
plt.legend(loc='best')
plt.ylim((-0.01, 1.01))
plt.xlim((a_interval[0]-0.01, a_interval[-1]+0.01))
plt.xlabel('a')
plt.ylabel('Frequency')
plt.title('Frequency vs a')
export_graph(f, 'Figure_2B')


#######################################################
#######################################################


### PLOT FIGURE 2(C): Average frequency of strategies for players 1 and 2

print 'Plotting Figure 2C'

clear_past_figs()
fig_all, (ax1, ax2) = plt.subplots(2,1, sharex=False, sharey=False) # make 2x1 grid of subplots
fig_all.set_size_inches(10, 15)

#plt.subplots_adjust(wspace=0.30, hspace=0.15)
#prepare plots
for ax in (ax1, ax2):
    ax.grid()
    ax.legend(loc='best')
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlim((a_interval[0]-0.01, a_interval[-1]+0.01))
    ax.set_xlabel('a')
    ax.set_ylabel('Frequency')
    process_ax(ax)
plt.tight_layout()

#player1
ax1.plot(a_interval[:avoid_end], (np.array(strategy_1)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'bo-', label='P1 CWOL')
ax1.plot(a_interval[:avoid_end], (np.array(strategy_2)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'ro-', label='P1 CWL')
ax1.plot(a_interval[:avoid_end], (np.array(strategy_3)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'yo-', label='P1 C in 1')
ax1.plot(a_interval[:avoid_end], (np.array(strategy_4)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'ko-', label='P1 All D')    
ax1.set_title('Average Frequency of Strategies - Player 1')
ax1.legend(loc='best')

#player2
ax2.plot(a_interval[:avoid_end], (np.array(strategy_5)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'co-', label='P2 Exit if Look')
ax2.plot(a_interval[:avoid_end], (np.array(strategy_6)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'mo-', label='P2 Exit if Defect')
ax2.plot(a_interval[:avoid_end], (np.array(strategy_7)/(float(repetitions)-np.array(no_outcome)))[:avoid_end], 'go-', label='P2 Always Exit')    
ax2.set_title('Average Frequency of Strategies - Player 2')
ax2.legend(loc='best')

fig_all.tight_layout()
export_graph(fig_all, 'Figure_2C')


#######################################################
#######################################################


### PLOT FIGURE 2(A): Player 1 and 2 strategy replicator trajectories from single simulation run

print 'Calculating or loading values for Figure 2A'

# Decide which a-values to use and plot.

def get_a_value_from_interval(bounds):
    for (bound_x, bound_y) in bounds:
        i_chosen = int(floor((bound_x+bound_y)/2.0))
        yield a_interval_tight[i_chosen]


a_selected = list(get_a_value_from_interval([one_equilibrium_region[0], two_equilibria_region[0], three_equilibria_region[0]]))
# This setup supports having multiple columns, i.e. one column for each a-value.
# The below is currently configured to hide all but the second column - however, we could easily disable this to return to all-column view, simply by commenting out the following line:
a_selected = a_selected[1:2]
print a_selected

# Randomly seed strategy frequencies:

if Calculate:
	tolerance_current=1e-2 # previously, 1e-3. arbitrary designation.
	x_0 = get_random_point_inside_simplex(4)  # random frequency
	y_0 = get_random_point_inside_simplex(3)  # random frequency
	t_vector = np.linspace(0.0, 30.0, 1000) # time values
	parameters_saved = [x_0, y_0, t_vector, tolerance_current, b, c1, c2, d, p, w] # a_selected is not necessary
	pickle.dump( parameters_saved, open( output_dir+"Figure 2_A_single simulation run of strategy replicator trajectories.saved_data", "wb" ) )
else: # load previous working version
	(x_0, y_0, t_vector, tolerance_current, b, c1, c2, d, p, w) = pickle.load(open(output_dir+"Figure 2_A_single simulation run of strategy replicator trajectories.saved_data", "r"))
	
# Begin plot:

print 'Plotting Figure 2A'

clear_past_figs()

fig_all, ax_arr = plt.subplots(2,len(a_selected), sharex=False, sharey=False, figsize=(10,20)) # make 2 rows x 3 columns grid of subplots; (30, 20) size when 3x2

for i in range(len(a_selected)):
    if len(a_selected) == 1: # Treat situation differently based on whether we are conmparing a-values or not.
        (ax_p1, ax_p2) = (ax_arr[0], ax_arr[1])
    else:
        (ax_p1, ax_p2) = (ax_arr[0,i], ax_arr[1,i])
    a_cur = a_selected[i]
    solution = replicator_trajectory_two_populations(get_game_population_1(a_cur, b, c1, c2, d, p, w), get_game_population_2(a_cur, b, c1, c2, d, p, w), x_0, y_0, t_vector, atol=tolerance_current)
    for ax in (ax_p1, ax_p2):
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim(0,10)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        ax.grid(True)
    ax_p1.plot(t_vector, solution[0], 'b-', label='P1 C wout looking', linewidth=2.0)
    ax_p1.plot(t_vector, solution[1], 'g-', label='P1 Observe and C', linewidth=2.0)
    ax_p1.plot(t_vector, solution[2], 'y-', label='P1 Observe and C only if 1 is chosen', linewidth=2.0)
    ax_p1.plot(t_vector, solution[3], 'r-', label='P1 ALLD', linewidth=2.0)
    ax_p2.plot(t_vector, solution[4], 'm--', label='P2 Continue iff P1 C wout looking', linewidth=2.0)
    ax_p2.plot(t_vector, solution[5], 'y--', label='P2 Continue iff P1 C', linewidth=2.0)
    ax_p2.plot(t_vector, solution[6], 'r--', label='P2 Exit', linewidth=2.0)
    ax_p1.set_title('Player 1 Strategies') # 'Player 1. a = '+str(a_cur)+'.'
    ax_p2.set_title('Player 2 Strategies') # 'Player 2. a = '+str(a_cur)+'.'
    ax_p1.legend(loc='best')
    ax_p2.legend(loc='best')

#fig_all.suptitle('Single simulation run, replicator trajectory; tolerance = '+str(tolerance_current)+'.', fontsize=24)
fig_all.tight_layout()
fig_all.subplots_adjust(top=0.85)
# fig_all.show()
export_graph(fig_all, 'Figure_2A')


#######################################################
#######################################################


print 'CW(O)L Simulation Calculations and Figures Complete.'
