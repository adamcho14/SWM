import sys
from numpy import *
import matplotlib.pyplot as plt
import math
import csv

# ----------------- MODEL SETUP --------------------------
# ----------------- Basic Parameters ---------------------
trial_no = "trial_1_" #set the trial number
n_e_or = 2048  # number of pyramidal cells used by Compte et al. (2000)
n_i_or = 512  # number of interneurons used by Compte et al. (2000)
factor = 4  # scaling factor for networks of different size
n_e = n_e_or/factor   # number of excitatory pyramidal cells - 2048 - 256 - 128
n_i = n_i_or/factor  # number of inhibitory interneurons - 512 - 64 - 32

fire = 30  # firing threshold

# ------- Simulation Period Lenghts ---------
t_0 = 100  # pre-trial
t_p = 250  # cue presentation
t_d = 350  # delay
t_s = 250  # response
t_e = 100  # post-trial

ms = t_0 + t_p + t_d + t_s + t_e # simulation time in ms - pre-presentation, presentation, delay, saccade, post-delay

timestep = 0.02 # simulation time step

time = int(float(ms) / float(timestep)) # number of simulation steps
print time
presented_cue = 180  # presented cue angle

# draw plots of the membrane potential development of these neurons:
close = int(presented_cue*n_e/360)  # neuron in the region close to the stimulation cue angle (can be changed)
far = int(((presented_cue-180)%360)*n_e/360)  # neuron in the region far from the stimulation cue angle (can be changed)
inter = n_e+1  # interneuron (can be changed)


v_poisson = 2  # poisson input current
i_transient = 30  # cue presentation period peak current value (alpha)
i_response = 4*i_transient  # response presentation period current value
ext_mu = 29  # delivery width
# ---------------- Architecture Parameters --------------
step_angle = 1  # step angle for the non-uniform distribution
cues = 360/step_angle  # number of cues in the non-uniform distribution

# ---------------- Channel Parameters--------------------
# synaptic reversal potential (Wacogne et al., 2012)
V_exc = 0  # excitatory synapse
V_inh = -70  # inhibitory synapse

# NMDAR channel parameters
G_NMDA = 0.002
Mg_ion = 1
alpha = 0.5
tau_decay = 100
tau_rise = 2

# GABAR channel parameters
G_GABA = 0.0075
tau_GABA = 10

# AMPAR channel parameters
G_AMPA = 0.0075
tau_AMPA = 2

# Connectivity footprint parameters
ex = -2  # exponent
G_EE = factor*0.381*(10**ex)
G_EI = factor*0.292*(10**ex)
G_IE = factor*1.336*(10**ex)
G_II = factor*1.024*(10**ex)

J_plus_EE = 1.62
J_plus_EI = 1.25
J_plus_IE = 1.4
J_plus_II = 1.5

sigma_EE = 18
sigma_EI = 18
sigma_IE = 32.4

# ------------- Functions --------------------------------


# initialize parameters of neuron
def init(a, b, c, d, v_0):
    return {"a": a, "b": b, "c": c, "d": d, "v_0": v_0}


# used to create an array of arrays of v's or u's, or inputs
def init_array(x, arange, is_Poisson):
    array = []
    for i in range(arange):
        if is_Poisson == 1:
            x = Poisson()
        array.append([x])
    return array


# random Poisson spikes with frequency 1.8 spikes/second per cell
def Poisson():
    if random.random() < float(1.8) / (1000/float(timestep)):
        return 1
    return 0


# update cell state according to Izhikevich
# pri update pouzijeme ako vstupne u, v na policku daneho casu t (kukni nizsie),
# co su vlaste aj posledne prvky v poliach u, v danej bunky

# new_v a new_u appendneme do poli v, u danej bunky
def update(a, b, c, d, v, u, input):
    new_v = v
    new_u = u
    output = 0

    if v < fire:
        new_v += timestep*((0.04 * v + 5) * v + 140 - u + input)
        new_u += timestep*(a * (b * v - u))
        if new_v >= fire:
            new_v = fire
            output = 1
    else:  # spike
        new_v = c
        new_u = u + d

    return [new_v, new_u, output]


# TRANSIENT CURRENT INPUT (Almeida et al., 2015; Edin et al., 2009)
def I_transient_current(v_transient, stim_cue, pref_cue):
    return v_transient * math.exp(
        ext_mu * (math.cos(2*math.pi/360*(pref_cue - stim_cue))-1))


# CHANNEL SIMULATION (Wang, Brunel & Wang, as by Compte, Wacogne et al.)
def I_channel_current(v, v_syn, g, s):
    return g * s * (v_syn - v)


mag = 1 #magnitude of jump when a neuron fires

# NMDAR channel dynamics
# gating variable s for NMDAR channel
def NMDAR_factor(v):
    return 1/(1 + Mg_ion * math.exp(-0.062 * v / 3.57))


def ds_NMDAR(s, x):
    return s + timestep * (-float(s)/float(tau_decay) + alpha * x * (1-s))


def dx_NMDAR(x, just_fired):
    return x + timestep * (-1/float(tau_rise) * x + mag * just_fired)


# GABAR channel dynamics
def ds_GABAR(s, just_fired):
    return s + timestep * (-float(s)/float(tau_GABA) + mag * just_fired)


# AMPAR channel dynamics
def ds_AMPAR(s, just_fired):
    return s + timestep * (-float(s)/float(tau_AMPA) + mag * just_fired)

# CONNECTIVITY FOOTPRINT - only excitatory-to-excitatory (Compte)
def W_integral(angle_1, sigma):
    my_sigma = math.sqrt(sigma**2)
    return 0.5*math.sqrt(math.pi)*my_sigma*\
           (math.erf(float(angle_1)/float(my_sigma)) - math.erf(float(angle_1-360)/float(my_sigma)))


def J_minus(angle_1, J_plus, sigma):
    return float(360 - J_plus*W_integral(angle_1, sigma))/float(360 - J_plus)


def W_conn_foot(angle_1, angle_2, J_plus, sigma):
    return J_minus(angle_1, J_plus, sigma) + (J_plus - J_minus(angle_1, J_plus, sigma)) * \
                                      math.exp(float(-(angle_1 - angle_2) ** 2)/float(2*sigma**2))


def preferred_cue(neuron_i, n):
    #  two options here - comment the one not used and uncomment the one you want to use
    return 360/float(n) * neuron_i  # uniform distribution
#  return (neuron_i % cues) * step_angle  #non-uniform distribution

# -------------------- Cell Initiation -----------------------------
# EXCITATION PYRAMIDAL CELL

exc = init(0.02, 0.2, -65, 8, -90)
inh = init(0.1, 0.2, -65, 2, -90)

# MEMBRANE POTENTIAL SETUP
v = [[]] * (n_e + n_i)

for i in range(n_e):
    v[i] = [0] * (time + 1)
    v[i][0] = exc["v_0"]
for i in range(n_e, n_e + n_i):
    v[i] = [0] * (time + 1)
    v[i][0] = inh["v_0"]

# MEMBRANE RECOVERY VARIABLE SETUP
u = [[]] * (n_e + n_i)

for i in range(n_e):
    u[i] = [0] * (time + 1)
    u[i][0] = exc["v_0"] * exc["b"]
for i in range(n_e, n_e + n_i):
    u[i] = [0] * (time + 1)
    u[i][0] = inh["v_0"] * inh["b"]

# INPUT SETUP
inp = [[]] * (n_e + n_i)
for i in range(n_e):
    inp[i] = [0] * time
for i in range(n_e, n_i + n_e):
    inp[i] = [0] * time

# OUTPUT SETUP
out = [0] * (n_e + n_i)

# NUMBER OF SPIKES SETUP
just_fired = [0] * (n_e + n_i)
num_of_firings = [0] * (n_e + n_i)

# NUMBER OF POISSON SPIKES INPUTS SETUP
num_of_poisson_spikes = [0] * (n_e + n_i)

# CONNECTION MATRIX SETUP - g[i][j] connection from neuron i to neuron j
g = [[]] * (n_e + n_i)
for i in range(n_e + n_i):
    g[i] = [0] * (n_e + n_i)

# uncomment the parts of equations which begin with " * W_conn_foot()" for enabling spatial tuning of connections
# comment those parts for disabling
# exc to ext
for i in range(n_e):
    for j in range(n_e):
        g[i][j] = G_EE * W_conn_foot(preferred_cue(i, n_e), preferred_cue(j, n_e), J_plus_EE, sigma_EE)

#exc to inh
for i in range(n_e):
    for j in range(n_e, n_e + n_i):
        g[i][j] = G_EI #* W_conn_foot(preferred_cue(i, n_e), preferred_cue(j-n_e, n_i), J_plus_EI, sigma_EI)#G_NMDA #* J_minus #* W_conn_foot(preferred_cue(i), preferred_cue(j))

#inh to exc,inh
for i in range(n_e, n_e+n_i):
    for j in range(n_e):
        g[i][j] = G_IE #* W_conn_foot(preferred_cue(i-n_e, n_i), preferred_cue(j,n_e), J_plus_IE, sigma_IE)
    for j in range(n_e, n_e + n_i):
        g[i][j] = G_II


#SYNAPTIC REVERSAL POTENTIAL SETUP
v_syn = [0] * (n_e + n_i)

for i in range(n_e):
    v_syn[i] = V_exc
for i in range(n_e, n_e + n_i):
    v_syn[i] = V_inh

#CHANNEL DYNAMICS VARIABLE SETUP
s = [1] * (n_e + n_i)
x = [1] * n_e

s_poisson = [1] * (n_e + n_i)

# ----------------- SIMULATION - here we start --------------------
# output files
with open(trial_no + 'experiment_log' + '.csv', mode='w') as logfile:
    log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(['time', 'input_close', 'v_close', 'u_close', 'input_far', 'v_far', 'u_far'])

for i in range(cues + 1):
    with open(trial_no + 'experiment_pyramid_' + str(i) + '.csv', mode='w') as exp:
        exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["time", "input", "membrane_potential", "variable"])
    with open(trial_no + 'number_of_firings_pyramid_' + str(i) + '.csv', mode='w') as exp:
        exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["time", "firings"])
for i in range(n_e):
    with open(trial_no + 'frequency_of_firings_pyramid_' + str(i) + '.csv', mode='w') as exp:
        exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["time", "firings"])
for i in range(n_e, n_e + n_i):
    with open(trial_no + 'experiment_interneuron_' + str(i) + '.csv', mode='w') as exp:
        exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["time", "input","membrane_potential", "variable"])
    with open(trial_no + 'number_of_firings_interneuron_' + str(i) + '.csv', mode='w') as exp:
        exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["time", "firings"])
    with open(trial_no + 'frequency_of_firings_interneuron_' + str(i) + '.csv', mode='w') as exp:
        exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["time", "firings"])
with open(trial_no + 'firings_log_pyramid.csv', mode='w') as logfile:
    log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(['time', 'neuron_fired'])
with open(trial_no + 'firings_log_interneuron.csv', mode='w') as logfile:
    log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(['time', 'neuron_fired'])
with open(trial_no + 'frequency_firings_log_pyramid.csv', mode='w') as logfile:
    log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(['neuron', 'frequency'])
with open(trial_no + 'frequency_firings_log_interneuron.csv', mode='w') as logfile:
    log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(['neuron', 'frequency'])

period_start = 0
period_end = 0
for t in range(time):
    # reset number of firings at the beginning of a new period
    if t == float(t_0) / float(timestep) or t == float(t_0 + t_p) / float(timestep) or t == float(
                            t_0 + t_p + t_d) / float(timestep):
        period_start = period_end
        period_end = t - 1
        for i in range(n_e + n_i):
            if i < n_e:
                with open(trial_no + 'frequency_of_firings_pyramid_' + str(i) + '.csv', mode='a') as exp:
                    exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    exp_writer.writerow([t-1, float(num_of_firings[i])/float(period_end - period_start)])
                if t == float(
                            t_0 + t_p + t_d) / float(timestep):
                    with open(trial_no + 'frequency_firings_log_pyramid.csv', mode='a') as logfile:
                        log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        log_writer.writerow([i, (1000/float(timestep))*(float(num_of_firings[i])/float(period_end - period_start))])
            else:
                with open(trial_no + 'frequency_of_firings_interneuron_' + str(i) + '.csv', mode='a') as exp:
                    exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    exp_writer.writerow([t-1, float(num_of_firings[i])/float(period_end - period_start)])
                if t == float(
                            t_0 + t_p + t_d) / float(timestep):
                    with open(trial_no + 'frequency_firings_log_interneuron.csv', mode='a') as logfile:
                        log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        log_writer.writerow([i, (1000/float(timestep))*(float(num_of_firings[i])/float(period_end - period_start))])
            num_of_firings[i] = 0

    # assign input to each cell
    for i in range(n_e + n_i):

        # external cortical areas activity
        poisson_spike = 0
        num_of_poisson_spikes[i] = 0
        if Poisson():
            num_of_poisson_spikes[i] = 1
            poisson_spike = v_poisson
            test_poisson = I_channel_current(poisson_spike, V_exc, G_AMPA, s_poisson[i])
            if math.isnan(test_poisson):
                print i, "poisson_nan" #  test for nan values
            inp[i][t] = test_poisson


        # presentation period - only to pyramidal cells close to stimulus
        if float(t_0) / float(timestep) <= t < float(t_0 + t_p) / float(timestep):
            if i < n_e:
                test_transient = I_transient_current(i_transient, presented_cue, preferred_cue(i,n_e))
                inp[i][t] += test_transient

        # post-delay period
        if float(t_0 + t_p + t_d) / float(timestep) <= t < float(t_0 + t_p + t_d + t_s) / float(timestep):
            inp[i][t] += i_response

        # channel currents
        for j in range(n_e + n_i):  # from presynaptic neuron j to postsynaptic neuron i
            if j != i:
                test_channel = I_channel_current(v[j][t], v_syn[j], g[j][i], s[j])
                if i == far and j == close:
                    print test_channel
                if j < n_e:  # from excitatory via NMDA
                    test_channel *= NMDAR_factor(v[j][t])
                if math.isnan(test_channel):
                    print i, j, ":", v[j][t], v_syn[j], g[j][i], s[j]
                    sys.exit(2)
                inp[i][t] += test_channel

    # Update and proceed to t+1
    # Excitatory
    for i in range(n_e):
        # update v and u
        res = update(exc["a"], exc["b"], exc["c"], exc["d"], v[i][t], u[i][t], inp[i][t])
        v[i][t+1] = res[0]
        u[i][t+1] = res[1]
        if res[0] < -1000000 or res[0] > 1000000:
            print "range error", i, res[0]
        just_fired[i] = res[2]
        if res[2] == 1:
            with open(trial_no + 'firings_log_pyramid.csv', mode='a') as logfile:
                log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                log_writer.writerow([t, i])
        num_of_firings[i] += res[2]
        if math.isnan(v[i][t+1]):
            print i, inp[i][t]

        # update channel
        x[i] = dx_NMDAR(x[i], just_fired[i])
        s[i] = ds_NMDAR(s[i], x[i])

    # Inhibitory
    for i in range(n_e, n_e+n_i):
        res = update(inh["a"], inh["b"], inh["c"], inh["d"], v[i][t], u[i][t], inp[i][t])
        v[i][t+1] = res[0]
        u[i][t+1] = res[1]
        if res[0] < -1000000 or res[0] > 1000000:
            print "range error", i, res[0]
        just_fired[i] = res[2]
        if res[2] == 1:
            with open(trial_no + 'firings_log_interneuron.csv', mode='a') as logfile:
                log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                log_writer.writerow([t, i])
        num_of_firings[i] += res[2]
        if math.isnan(v[i][t+1]):
            print i, inp[i][t]

        s[i] = ds_GABAR(s[i], just_fired[i])

# output files
    with open(trial_no + 'experiment_log' + '.csv', mode='a') as logfile:
        log_writer = csv.writer(logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow([t, inp[close][t], v[close][t+1], u[close][t+1], inp[far][t], v[far][t+1], u[far][t+1]])
    print t, ":", inp[close][t],"/", v[close][t+1], "/", u[close][t+1], " - ", \
        inp[far][t], "/", v[far][t+1], "/", u[far][t+1], " - ", \
        inp[inter][t], "/", v[inter][t+1], "/", u[inter][t+1]

    for i in range(cues + 1):
        with open(trial_no + 'experiment_pyramid_' + str(i) + '.csv', mode='a') as exp:
            exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            exp_writer.writerow([t, inp[i][t], v[i][t+1], u[i][t+1]])
        with open(trial_no + 'number_of_firings_pyramid_' + str(i) + '.csv', mode='a') as exp:
            exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            exp_writer.writerow([t, num_of_firings[i]])
    for i in range(n_e, n_e + n_i):
        with open(trial_no + 'experiment_interneuron_' + str(i) + '.csv', mode='a') as exp:
            exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            exp_writer.writerow([t, inp[i][t], v[i][t+1], u[i][t+1]])
        with open(trial_no + 'number_of_firings_interneuron_' + str(i) + '.csv', mode='a') as exp:
            exp_writer = csv.writer(exp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            exp_writer.writerow([t, num_of_firings[i]])

print "END!"

# draw plots

plt.plot(v[close])
plt.axvline(x=int(float(t_0)/float(timestep)), label='Close')
plt.axvline(x=int(float(t_0+t_p)/float(timestep)), label='P')
plt.axvline(x=int(float(t_0+t_p+t_d)/float(timestep)), label='D')
plt.axvline(x=int(float(t_0+t_p+t_d+t_s)/float(timestep)), label='S')
plt.legend()
plt.show()

plt.plot(v[far])
plt.axvline(x=int(float(t_0)/float(timestep)), label='Far')
plt.axvline(x=int(float(t_0+t_p)/float(timestep)), label='P')
plt.axvline(x=int(float(t_0+t_p+t_d)/float(timestep)), label='D')
plt.axvline(x=int(float(t_0+t_p+t_d+t_s)/float(timestep)), label='S')
plt.legend()
plt.show()

plt.plot(v[inter])
plt.axvline(x=int(float(t_0)/float(timestep)), label='Inter')
plt.axvline(x=int(float(t_0+t_p)/float(timestep)), label='P')
plt.axvline(x=int(float(t_0+t_p+t_d)/float(timestep)), label='D')
plt.axvline(x=int(float(t_0+t_p+t_d+t_s)/float(timestep)), label='S')
plt.legend()
plt.show()










