# ============================================================================ #
# Framing Time Pressure -- Attention Model
# Author: Ian D. Roberts
# Date: 05.20.2020
# ============================================================================ #


# SETUP ========================================================================

# import modules
import os
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
import scipy.stats as sps
import shelve

# define custom functions
def norm_pdf(x, mu, sigma):
    """Probability density function for normal distribution.

    Arguments:
        x {float} -- Value at which to calculate density
        mu {float} -- Mean of normal distribution
        sigma {float} -- Standard deviation of normal distribution

    Returns:
        float -- Probability density
    """
    x2 = (x-mu)/sigma
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*(x2**2))

def norm_cdf(x, mu, sigma):
    """Cumulative density function for normal distribution.

    Arguments:
        x {float} -- Value at which to calculate density
        mu {float} -- Mean of normal distribution
        sigma {float} -- Standard deviation of normal distribution

    Returns:
        float -- Cumulative density
    """
    x2 = (x-mu)/sigma
    if type(x2) is np.ndarray:
        res = np.full(len(x2), np.nan)
        for i in range(len(x2)):
            res[i] = 0.5 * (1 + math.erf(x2[i]/np.sqrt(2)))
    else:
        res = np.array([0.5 * (1 + math.erf(x2/np.sqrt(2)))])
    return res

def truncnorm_pdf(x, mus, sigmas, lb, ub):
    """Calculates probability density for even mixture of several truncated
    normal distributions. Because the framing task samples gamble probabilities
    from 1 of 3 truncated normal distributions with equal probability
    (i.e., 0.333), this is needed to calculate the overall probability of any
    given gamble probability.

    Arguments:
        x {numpy array; float} -- Value at which to calculate density
        mus {numpy array; float} -- Means of truncated normal distributions
        sigmas {numpy array; float} -- Standard deviations of truncated normal
            distributions
        lb {float} -- Lower bound for truncated normal distributions (same for
            all)
        ub {float} -- Upper bound for truncated normal distributions (same for
            all)

    Returns:
        float -- Probability density
    """
    probs = np.full(len(x), np.nan)
    for i in range(len(x)):
        numers = norm_pdf(x[i], mus, sigmas)
        denoms = norm_cdf(ub, mus, sigmas) - norm_cdf(lb, mus, sigmas)
        probs[i] = np.sum((numers / denoms) / sigmas) / len(mus)
    return probs

def entropy_norm(sd):
    """Calculates entropy of a normal distribution
    See: https://en.wikipedia.org/wiki/Normal_distribution#Maximum_entropy

    Arguments:
        sd {float} -- Standard deviation of distribution

    Returns:
        float -- Entropy (in nats)
    """
    return 0.5 * (1 + np.log(2 * np.pi * sd**2))

def entropy_discrete(probs):
    """Calculates entropy of a discrete distribution
    See: https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition

    Arguments:
        probs {numpy array; float} -- List of probabilities of each outcome
            (leave out probabilities of 0)

    Returns:
        float -- Entropy (in nats)
    """
    if probs.ndim == 1:
        ent = 0
        for p in probs:
            if p > 0:
                ent += p * np.log(1/p)
        np.array([ent])
    else:
        ent = np.full(probs.shape[0], np.nan)
        for e in np.arange(probs.shape[0]):
            ent[e] = 0
            for p in probs[e,:]:
                if p > 0:
                    ent[e] += p * np.log(1/p)

    return ent

def choice_probs_max(gamb, sure, fixGamb, sightBias):
    """Calculate the probabilities of choosing the two options

    Arguments:
        gamb {numpy array; float} -- Subjective value belief for the gamble.
            Column 0 is mean and column 1 is variance.
        sure {numpy array; float} -- Subjective value belief for the sure
            option. Column 0 is mean and column 1 is variance.
        fixGamb {numpy array; float} -- Array of current fixations for each
            simulation. 1 = gamble, 0 = sure, NaN = neither
        sightBias {float} -- Parameter determing how much advantage is given to
            the currently fixated option.

    Returns:
        numpy array; float -- Array of probabilities of choosing either option.
            Column 0 is the gamble and column 1 is the sure option.
    """
    # code sure option fixations as -1 and NaNs as 0
    # creates indicator variable for sightBias
    fixGamb[fixGamb == 0] = -1
    fixGamb[np.isnan(fixGamb)] = 0

    # take difference of priors
    diff_mean = (gamb[:,0] - sure[:,0]) + fixGamb*sightBias
    diff_var = gamb[:,1] + sure[:,1]

    # probability of choosing gamble
    probGamb = 1 - norm_cdf(0, diff_mean, np.sqrt(diff_var))
    return np.array([probGamb, 1-probGamb]).T

def update_prior(prior, kg, value, noise, iv):
    """Updates subjective value prior for a choice option.

    Arguments:
        prior {numpy array; float} -- Array of means (column 0) and variances
            (column 1)
        kg {numpy array; float} -- Kalman gain
        value {float} -- The subjective value of the option
        noise {float} -- The noisiness of the option's subjective value
        iv {float} -- Innovation variance

    Returns:
        numpy array; float -- Posterior array of means and variances
    """
    post = np.full(prior.shape, np.nan)  # initialize posterior array
    post[:,0] = prior[:,0] + kg*(value - prior[:,0])  # update mean
    post[:,1] = (1-kg) * (prior[:,1] + iv)  # update variance
    return post

def update_kalman_gain(var, iv, noise, kl):
    """Update the Kalman gain for a given choice option

    Arguments:
        var {numpy array; float} -- Current variance of the belief (i.e., prior)
        iv {float} -- Innovation variance
        noise {float} -- The noisiness of the option's subjective value
        kl {float} -- Kalman "laziness" (controls learning rate)

    Returns:
        float -- Updated Kalman gain
    """
    return kl * ((var + iv) / (var + iv + noise))


def pressure(t, d, de):
    """Calculates the subjective probability of being able to draw another
    sample without running out of decision time.

    Arguments:
        t {float} -- Current time point/sample
        d {float} -- Point at which probability becomes 0.5
        de {float} -- The degree of uncertainty around when time will run out.
            Alternatively can be thought of as the rate at which pressure builds.

    Returns:
        float -- Probability of being able to sample again safely
    """
    return 1 / (1 + np.exp(de*(t-d)))



def prob_sample(gamb, sure, fixGamb, sightBias, wInfo, sensSamp, attendPressure,
                switchCost, t, d, de):
    """Calculates the probability of continuing information search (vs
    terminating and making a choice)

    Arguments:
        gamb {numpy array; float} -- Subjective value belief for the gamble.
            Column 0 is mean and column 1 is variance.
        sure {numpy array; float} -- Subjective value belief for the sure
            option. Column 0 is mean and column 1 is variance.
        fixGamb {numpy array; float} -- Array of current fixations for each
            simulation. 1 = gamble, 0 = sure, NaN = neither
        sightBias {float} -- Parameter determing how much advantage is given to
            the currently fixated option.
        wInfo {float} -- Parameter controlling the weight given to option
            uncertainty
        sensSamp {float} -- Parameter controlling sensitivity to SampleValue
        attendPressure {float} -- Parameter controlling how much earlier
            attention pressure builds relative to decision pressure
        switchCost {float} -- Additional cost for switching fixation
        t {float} -- Current time point/sample
        d {float} -- Point at which probability becomes 0.5
        de {float} -- The degree of uncertainty around when time will run out.
            Alternatively can be thought of as the rate at which pressure builds.

    Returns:
        float -- Probability of drawing another sample (i.e., continuing
            information search)
    """
    
    # identify simulations currently fixating sure option
    fixSure = 1 - fixGamb
    fixGamb[np.isnan(fixGamb)] = 0  # if the current fix is nan, fixing nothing
    fixSure[np.isnan(fixSure)] = 0

    # calculate choice probabilities given current fixation
    cp = choice_probs_max(gamb, sure, fixGamb, sightBias)

    # calculate choice entropy
    ce = entropy_discrete(cp)

    # calculate option value entropy
    optSDs = np.full((gamb.shape[0], 2), np.nan)
    optSDs[:,0] = np.sqrt(gamb[:,1])
    optSDs[:,1] = np.sqrt(sure[:,1])

    oe = np.full(optSDs.shape, np.nan)
    oe[:,0] = entropy_norm(optSDs[:,0])
    oe[:,1] = entropy_norm(optSDs[:,1])
    # apply time pressure weighting
    oe[:,0] *= pressure(t + attendPressure + (1-fixGamb)*switchCost, d, de)
    oe[:,1] *= pressure(t + attendPressure + (1-fixSure)*switchCost, d, de)
    
    # calculate SampleValue
    weighted_ce = pressure(t, d, de) * ce
    weighted_oe = (1-np.exp(-ce))*wInfo*np.sum(oe, axis=1)
    sv = weighted_ce + weighted_oe

    return (2 / (1 + np.exp(-sensSamp*sv))) - 1


def fixation_probs(gamb, sure, gambNoise, sureNoise, fixGamb, sightBias, wInfo,
                   wNoise, sensAttend, sensPressure, attendPressure, switchCost,
                   t, d, de):
    """Calculates the probability of fixating the gamble

    Arguments:
        gamb {numpy array; float} -- Subjective value belief for the gamble.
            Column 0 is mean and column 1 is variance.
        sure {numpy array; float} -- Subjective value belief for the sure
            option. Column 0 is mean and column 1 is variance.
        gambNoise {float} -- Noisiness of learning gamble subjective value
        sureNoise {float} -- Noisiness of learning sure option subjective value
        fixGamb {numpy array; float} -- Array of current fixations for each
            simulation. 1 = gamble, 0 = sure, NaN = neither
        sightBias {float} -- Parameter determing how much advantage is given to
            the currently fixated option.
        wInfo {float} -- Parameter controlling the weight given to option
            uncertainty
        wNoise {float} -- Parameter controlling the weight given to the
            noisiness of learning an option's subjective value
        sensAttend {float} -- Starting value of sensitivity to attention value
        sensPressure {float} -- Degree to which time pressure increases
            sensitivity to attention value
        attendPressure {float} -- Parameter controlling how much earlier
            attention pressure builds relative to decision pressure
        switchCost {float} -- Additional cost for switching fixation
        t {float} -- Current time point/sample
        d {float} -- Point at which probability becomes 0.5
        de {float} -- The degree of uncertainty around when time will run out.
            Alternatively can be thought of as the rate at which pressure builds.

    Returns:
        float -- Probability of fixating gamble next
    """
    
    # identify simulations currently fixating sure option
    fixSure = 1 - fixGamb
    fixGamb[np.isnan(fixGamb)] = 0  # if the current fix is nan, fixing nothing
    fixSure[np.isnan(fixSure)] = 0

    # calculate choice probabilities given current fixation
    cp = choice_probs_max(gamb, sure, fixGamb, sightBias)

    # calculate choice entropy
    ce = entropy_discrete(cp)

    # calculate value of attention to information
    gambSD = np.sqrt(gamb[:,1])
    sureSD = np.sqrt(sure[:,1])

    ai_g = np.full(gambSD.shape, np.nan)
    ai_s = np.full(sureSD.shape, np.nan)
    ai_g = ce*wInfo*entropy_norm(gambSD)
    ai_s = ce*wInfo*entropy_norm(sureSD)
    ai_g *= pressure(t + attendPressure + (1-fixGamb)*switchCost, d, de)
    ai_s *= pressure(t + attendPressure + (1-fixSure)*switchCost, d, de)

    # calculate value of efficient information gathering
    an_g = -ce*wNoise*gambNoise
    an_s = -ce*wNoise*sureNoise
    an_g *= (1-pressure(t + attendPressure + (1-fixGamb)*switchCost, d, de))
    an_s *= (1-pressure(t + attendPressure + (1-fixSure)*switchCost, d, de))

    # calculate value of attending to subjective value
    av_g = np.exp(-wInfo*ai_g)*gamb[:,0]
    av_s = np.exp(-wInfo*ai_s)*sure[:,0]
    
    gv = ai_g + an_g + av_g
    sv = ai_s + an_s + av_s

    tmp = sensAttend * (1 + sensPressure*(1-pressure(t + attendPressure, d, de)))
    res = 1 / (1 + np.exp(-tmp*(gv-sv)))
    return res


# set simulation parameters -----------------------------------------
simLabel = "timePressure_freeAttend_max"  # for labelling output
saveResults = True

# plot settings
# plt.style.use('dark_background')
showPlots = True
savePlots = True
fileFormat = "pdf"

# whether to have simulations terminate in a choice
simChoices = True

# number of simulations to run for each trial
nSims = 1000

# set number of samples (fixation)
nSamps = 10

# if None, will simulate probabilistic sampling sequence
# otherwise, set a list for sampling sequence to simulate: 1 = gamble, 0 = sure
# (start sample sequence with a -1 for no initial fixation)
seqLabel = "none"
fix_seq = None #[-1] + [1]*9

# prospect theory parameters
r = 0.88  # risk aversion
l = 1.5  # loss aversion
g = 0.61  # probabilty weighting

sightBias = 5.0  # response bias to choose what's currently fixated
kl = 0.3  # kalman laziness
iv = 0.0  # innovation variance (keep at zero for now)
wInfo = 0.0  # weight given to option value uncertainties when directing attention
wNoise = 0.5  # weight given to learning noisiness of option values when directing attention
sensAttend = 0.1  # sensitivity to attention values
sensPressure = 0.5  # pressure's influence on sensAttend
attendPressure = 5.0  # degree to which attention pressure builds quicker than choice pressure
switchCost = 1  # cost of switching (leave fixed at 1 for now)
sensSamp = 3.0  # sensitivity to uncertainty when deciding to continue sampling
d = 5.0  # shift overall time pressure
de = 1.0  # time constraint uncertainty

# the true subjective value is noisily encoded
noise = {"gain": 10.0,
         "loss": 10.0,
         "gamb": 20.0}

# create trial stimuli ----------------------------------------------
# generate subjective values for probability * endowment combos
probs = np.arange(0.1, 0.91, 0.1)
endows = np.arange(20, 91, 10)
trial_svs = {"gain": np.full([len(endows), len(probs)], np.nan),
             "loss": np.full([len(endows), len(probs)], np.nan),
             "gamb": np.full([len(endows), len(probs)], np.nan)}
prob_pw = np.full([len(endows), len(probs)], 0)

# get weighting for frequency of each probability
# because gamble probabilities are drawn from truncated normal distributions
# where lower probabilities are more likely, need to weight these more heavily
# when generating priors later
pw = truncnorm_pdf(probs, np.array([0.28, 0.42, 0.56]),
                   np.array([0.2, 0.2, 0.2]), 0.1, 0.9)
pw /= np.sum(pw)
pw = np.int64(np.round(pw*1000, 0))

for i in range(len(endows)):
    for j in range(len(probs)):
        # calculate sure outcomes
        gainOut = int(endows[i]*probs[j])
        lossOut = int(endows[i] - gainOut)

        # weight probabiltiy
        p = (probs[j]**g) / (probs[j]**g + (1-probs[j])**g)**(1/g)

        # append subjective values
        trial_svs["gain"][i,j] = gainOut**r
        trial_svs["loss"][i,j] = endows[i]**r - l*(lossOut**r)
        trial_svs["gamb"][i,j] = (endows[i]**r) * p
        prob_pw[i,j] = pw[j]


# at the beginning of a trial, participant has a gaussian prior for each option 
# with the means set at the weighted median of the subjective values and the
# variance is the variance of the subjective values
sv_priors = {"gain": [np.median([i for i, w in zip(trial_svs["gain"].flatten(), prob_pw.flatten()) for j in range(w)]),
                      np.var([i for i, w in zip(trial_svs["gain"].flatten(), prob_pw.flatten()) for j in range(w)], ddof = 1)],
             "loss": [np.median([i for i, w in zip(trial_svs["loss"].flatten(), prob_pw.flatten()) for j in range(w)]),
                      np.var([i for i, w in zip(trial_svs["loss"].flatten(), prob_pw.flatten()) for j in range(w)], ddof = 1)],
             "gamb": [np.median([i for i, w in zip(trial_svs["gamb"].flatten(), prob_pw.flatten()) for j in range(w)]),
                      np.var([i for i, w in zip(trial_svs["gamb"].flatten(), prob_pw.flatten()) for j in range(w)], ddof = 1)]}


# TRIAL SIMULATION =============================================================

# initialize for simulations
# probabilities of choosing option with higher subjective value or gamble
prob_max = {"sv": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                   "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)},
            "gamb": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                     "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)}}
# choice entropy (sure vs gamble)
# total entropy (sure vs gamble vs resample vs switch)
entropy_max = {"choice": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                          "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)},
               "total": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                         "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)}}

# responses
resp_max = {"gain": {"gamb": np.full([len(endows), len(probs), nSims], np.nan),
                     "samps": np.full([len(endows), len(probs), nSims], np.nan)},
            "loss": {"gamb": np.full([len(endows), len(probs), nSims], np.nan),
                     "samps": np.full([len(endows), len(probs), nSims], np.nan)}}

# initialize sampling sequence
# switchs and fixate gamble
if not fix_seq:
    samp_max = {"switch": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                           "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)},
                "gamb": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                         "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)},
                "prob_switch": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                                "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)},
                "prob_gamb": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                              "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)}}
    
    # the first is always a switch
    samp_max["switch"]["gain"][:,:,0,:] = 1
    samp_max["switch"]["loss"][:,:,0,:] = 1
else:
    lead_samp = fix_seq[1:] + [fix_seq[-1]]
    switch_seq = [int(i!=j) for i,j in zip(fix_seq, lead_samp)]
    switchFix = np.repeat(np.array(switch_seq)[:, np.newaxis], nSims, axis=1)
    switchFix = np.repeat(switchFix[np.newaxis,:,:], len(probs), axis=0)
    switchFix = np.repeat(switchFix[np.newaxis,:,:,:], len(endows), axis=0)

    tmp = [np.nan if i == -1 else i for i in fix_seq]
    gambFix = np.repeat(np.array(tmp)[:, np.newaxis], nSims, axis=1)
    gambFix = np.repeat(gambFix[np.newaxis,:,:], len(probs), axis=0)
    gambFix = np.repeat(gambFix[np.newaxis,:,:,:], len(endows), axis=0)
    samp_max = {"switch": {"gain": switchFix.copy(),
                            "loss": switchFix.copy()},
                "gamb": {"gain": gambFix.copy(),
                            "loss": gambFix.copy()},
                "prob_switch": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                                "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)},
                "prob_gamb": {"gain": np.full([len(endows), len(probs), nSamps, nSims], np.nan),
                              "loss": np.full([len(endows), len(probs), nSamps, nSims], np.nan)}}


# loop through all trials and simulate sampling information
nTrials = len(probs) * len(endows)
counter = 1
for i in range(len(endows)):
    for j in range(len(probs)):
        # if counter % (nTrials//5) == 0:
        print(f"Simulating trial {counter} of {nTrials}", flush = True)
        counter += 1

        # get true subjective values
        true_svs = {"gain": trial_svs["gain"][i,j],
                    "loss": trial_svs["loss"][i,j],
                    "gamb_gain": trial_svs["gamb"][i,j],
                    "gamb_loss": trial_svs["gamb"][i,j]}

        # initialize priors for that trial
        max_p = {"gain": np.repeat(np.array(sv_priors["gain"])[np.newaxis,:], nSims, axis=0),
                 "loss": np.repeat(np.array(sv_priors["loss"])[np.newaxis,:], nSims, axis=0),
                 "gamb_gain": np.repeat(np.array(sv_priors["gamb"])[np.newaxis,:], nSims, axis=0),
                 "gamb_loss": np.repeat(np.array(sv_priors["gamb"])[np.newaxis,:], nSims, axis=0)}

        # initialize kalman gain
        max_kg = {"gain": update_kalman_gain(max_p["gain"][:,1], iv, noise["gain"], kl),
                  "loss": update_kalman_gain(max_p["loss"][:,1], iv, noise["loss"], kl),
                  "gamb_gain": update_kalman_gain(max_p["gamb_gain"][:,1], iv, noise["gamb"], kl),
                  "gamb_loss": update_kalman_gain(max_p["gamb_loss"][:,1], iv, noise["gamb"], kl)}

        # intialize for responses
        if simChoices:
            complete_max = {"gain": np.full(nSims, False),
                            "loss": np.full(nSims, False)}

        # draw samples
        for s in range(nSamps):

            for frame in ["gain", "loss"]:
                # no information gathered during a switch
                # flag simulations that are in the middle of a switch
                sw = samp_max["switch"][frame][i,j,s,:] == 1
                
                # whether fixating gamble or sure
                fixS = np.logical_and(samp_max["gamb"][frame][i,j,s,:] == 0, ~sw)
                fixG = np.logical_and(samp_max["gamb"][frame][i,j,s,:] == 1, ~sw)

                # update option values
                # sure
                max_p[frame][fixS,:] = update_prior(max_p[frame][fixS,:],
                                                    max_kg[frame][fixS],
                                                    true_svs[frame],
                                                    noise[frame], iv)
                # gamble
                max_p[f"gamb_{frame}"][fixG,:] = update_prior(max_p[f"gamb_{frame}"][fixG,:],
                                                                max_kg[frame][fixG],
                                                                true_svs[f"gamb_{frame}"],
                                                                noise["gamb"], iv)

                # update kalman gain
                # sure
                max_kg[frame][fixS] = update_kalman_gain(max_p[frame][fixS,1],
                                                         iv, noise[frame], kl)
                # gamble
                max_kg[f"gamb_{frame}"][fixG] = update_kalman_gain(max_p[f"gamb_{frame}"][fixG,1],
                                                                   iv, noise["gamb"], kl)
            
                # choice probabilities
                curr_probs_max = np.full(nSims, np.nan)
                curr_probs_max = choice_probs_max(max_p[f"gamb_{frame}"],
                                                  max_p[frame],
                                                  samp_max["gamb"][frame][i,j,s,:].copy(),
                                                  sightBias)

                # store probability of choosing gamble
                prob_max["gamb"][frame][i,j,s,:] = curr_probs_max[:,0]

                # store probability of choosing highest subjective value
                prob_max["sv"][frame][i,j,s,:] = np.nan
                if true_svs[frame] > true_svs[f"gamb_{frame}"]:
                    prob_max["sv"][frame][i,j,s,:] = curr_probs_max[:,1]
                else:
                    prob_max["sv"][frame][i,j,s,:] = curr_probs_max[:,0]

                # choice entropy
                entropy_max["choice"][frame][i,j,s,:] = entropy_discrete(curr_probs_max)
                                
                # fixation probabilities
                probFixGamb = fixation_probs(max_p[f"gamb_{frame}"].copy(),
                                             max_p[frame].copy(),
                                             noise["gamb"],
                                             noise[frame],
                                             samp_max["gamb"][frame][i,j,s,:].copy(),
                                             sightBias, wInfo, wNoise,
                                             sensAttend, sensPressure,
                                             attendPressure, switchCost, s, d, de)
                
                # store probability of fixating gamble and of switching
                if (s > 0) & (s < (nSamps-1)):
                    samp_max["prob_gamb"][frame][i,j,s+1,:] = probFixGamb
                    samp_max["prob_switch"][frame][i,j,s,fixS] = probFixGamb[fixS]
                    samp_max["prob_switch"][frame][i,j,s,fixG] = 1-probFixGamb[fixG]
                elif (s > 0) & (s == (nSamps-1)):
                    samp_max["prob_switch"][frame][i,j,s,fixS] = probFixGamb[fixS]
                    samp_max["prob_switch"][frame][i,j,s,fixG] = 1-probFixGamb[fixG]
                else:
                    samp_max["prob_gamb"][frame][i,j,s+1,:] = probFixGamb
                    samp_max["prob_switch"][frame][i,j,s,:] = np.nan

                # probability of making a decision
                prob_decide = 1-prob_sample(max_p[f"gamb_{frame}"].copy(),
                                            max_p[frame].copy(),
                                            samp_max["gamb"][frame][i,j,s,:].copy(),
                                            sightBias, wInfo, sensSamp,
                                            attendPressure, switchCost, s, d, de)
                
                # make decisions
                if simChoices:
                    done = complete_max[frame]  # flag already decided simulations
                    # choose whether to decide or keep sampling
                    chose = np.random.binomial(1, prob_decide, nSims) != 0
                    if np.any(np.logical_and(chose, ~sw)):
                        # make decisions
                        choseGamb = np.random.binomial(1, curr_probs_max[chose & ~sw & ~done,0])
                        # store decisions and "RT"
                        resp_max[frame]["gamb"][i,j,chose & ~sw & ~done] = choseGamb
                        resp_max[frame]["samps"][i,j,chose & ~sw & ~done] = s
                        complete_max[frame][chose] = True  # mark as completed
                else:
                    chose = np.full(nSims, False)
                
                if s > 0:
                    if np.any(np.logical_and(~chose, ~sw)):
                        # if didn't decide, keep sampling
                        if (s < (nSamps-1)) and not fix_seq:
                            # only do this step if there is still more time and
                            # the fixation sequence wasn't pre-set

                            # choose what to fixate
                            fixGamb = np.random.binomial(1, probFixGamb[~chose & ~sw])
                            samp_max["gamb"][frame][i,j,s+1,:] = samp_max["gamb"][frame][i,j,s,:]  # repeat previous fixation
                            samp_max["gamb"][frame][i,j,s+1,chose] = np.nan  # no more fixations if choice was made
                            samp_max["gamb"][frame][i,j,s+1,~chose & ~sw] = fixGamb  # input new fixations

                            # was it a switch?
                            currFix = samp_max["gamb"][frame][i,j,s,~chose & ~sw]
                            nextFix = samp_max["gamb"][frame][i,j,s+1,~chose & ~sw]
                            switches = np.int64(currFix != nextFix)
                            samp_max["switch"][frame][i,j,s+1,sw] = 0
                            samp_max["switch"][frame][i,j,s+1,~chose & ~sw] = switches
                elif (s == 0) and not fix_seq:
                    # choose what to fixate
                    fixGamb = np.random.binomial(1, probFixGamb)
                    samp_max["gamb"][frame][i,j,s+1,:] = fixGamb  # input new fixations


# PLOT RESULTS =================================================================

plotDir = os.path.join(os.getcwd(), 'kalman_filter', simLabel)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# get means across simulations
mean_prob_max = {"sv": {"gain": np.nanmean(prob_max["sv"]["gain"], 3),
                        "loss": np.nanmean(prob_max["sv"]["loss"], 3)},
                 "gamb": {"gain": np.nanmean(prob_max["gamb"]["gain"], 3),
                          "loss": np.nanmean(prob_max["gamb"]["loss"], 3)}}

mean_entropy_max = {"choice": {"gain": np.nanmean(entropy_max["choice"]["gain"], 3),
                               "loss": np.nanmean(entropy_max["choice"]["loss"], 3)},
                    "total": {"gain": np.nanmean(entropy_max["total"]["gain"], 3),
                              "loss": np.nanmean(entropy_max["total"]["loss"], 3)}}

mean_samp_max = {"switch": {"gain": np.nanmean(samp_max["switch"]["gain"], 3),
                            "loss": np.nanmean(samp_max["switch"]["loss"], 3)},
                 "gamb": {"gain": np.nanmean(samp_max["gamb"]["gain"], 3),
                          "loss": np.nanmean(samp_max["gamb"]["loss"], 3)},
                 "prob_switch": {"gain": np.nanmean(samp_max["prob_switch"]["gain"], 3),
                                 "loss": np.nanmean(samp_max["prob_switch"]["loss"], 3)},
                 "prob_gamb": {"gain": np.nanmean(samp_max["prob_gamb"]["gain"], 3),
                               "loss": np.nanmean(samp_max["prob_gamb"]["loss"], 3)}}

if saveResults:
    filename = os.path.join(plotDir, f'{simLabel}.out')
    my_shelf = shelve.open(filename,'n') # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

# plot results for both frames
for frame in ["gain", "loss"]:

    # relative subjective values -----------------------------------------------
    diff_matrix = trial_svs[frame] - trial_svs["gamb"]
    diff_df = pd.DataFrame({"values": diff_matrix.flatten(),
                            "endow": [i for i in endows for j in range(len(probs))],
                            "probs": list(np.round(probs,2))*len(endows)})
    diff_df = diff_df.pivot(index="endow", columns="probs", values="values")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.heatmap(diff_df, vmin = -1*np.max(np.abs(diff_matrix.flatten())),
                vmax = np.max(np.abs(diff_matrix.flatten())), center = 0,
                linewidths=0.0, rasterized=True, ax = ax, xticklabels=4,
                yticklabels=2)
    ax.set_ylabel("Endowment", fontsize=14)
    ax.set_xlabel("Gamble Probability", fontsize=14)
    ax.invert_yaxis()
    fig.tight_layout(rect=[0.03, 0.03, .98, 1])
    plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18)
    plt.subplots_adjust(top=0.88)
    fig.text(0.95, 0.5, 'Sure SV vs. Gamble SV', ha='center', va='center', rotation=270, fontsize=14)
    if savePlots:
        fn = os.path.join(plotDir, f"{frame}_sureVgamb_subjectiveValues_{simLabel}.{fileFormat}")
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if showPlots:
        plt.show()


    # CHOICE PROBABILITIES =====================================================

    # probability choose higher subjective value -------------------------------
    fig, axs = plt.subplots(int(np.ceil(nSamps/5)), 5,
                            figsize=(14.5, int(np.ceil(nSamps/5)*3)),
                            sharex = True, sharey = True)
    cbar_ax = fig.add_axes([.88, .3, .03, .4])
    for i, ax in enumerate(axs.flat):
        if i >= nSamps:
            break

        df = pd.DataFrame({"values": mean_prob_max["sv"][frame][:,:,i].flatten(),
                            "endow": [i for i in endows for j in range(len(probs))],
                            "probs": list(np.round(probs,2))*len(endows)})
        df = df.pivot(index="endow", columns="probs", values="values")

        sns.heatmap(df, vmin = 0, vmax = 1, center = 0.5,
                    cbar = i == 0, ax=ax, linewidths=0.0, rasterized=True,
                    cbar_ax = None if i else cbar_ax, xticklabels=4,
                    yticklabels=2)
        ax.invert_yaxis()
        ax.set_title(f"Sample {i}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.tight_layout(rect=[0.03, 0.03, .96, 1])
    plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18, x=0.465)
    plt.subplots_adjust(top=0.88, right=0.87)
    fig.text(0.465, 0.02, 'Gamble Probability', ha='center', va='center', fontsize=14)
    fig.text(0.95, 0.5, 'Probability of Choosing Higher SV', ha='center', va='center', rotation=270, fontsize=14)
    fig.text(0.02, 0.5, 'Endowment', ha='center', va='center', rotation="vertical", fontsize=14)
    if savePlots:
        fn = os.path.join(plotDir, f"{frame}_max_probSV_{simLabel}.{fileFormat}")
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if showPlots:
        plt.show()

    # probability choose gamble ------------------------------------------------
    fig, axs = plt.subplots(int(np.ceil(nSamps/5)), 5,
                            figsize=(14.5, int(np.ceil(nSamps/5)*3)),
                            sharex = True, sharey = True)
    cbar_ax = fig.add_axes([.88, .3, .03, .4])
    # tick_locator = ticker.MaxNLocator(5)
    for i, ax in enumerate(axs.flat):
        if i >= nSamps:
            break
            
        df = pd.DataFrame({"values": mean_prob_max["gamb"][frame][:,:,i].flatten(),
                            "endow": [i for i in endows for j in range(len(probs))],
                            "probs": list(np.round(probs,2))*len(endows)})
        df = df.pivot(index="endow", columns="probs", values="values")

        sns.heatmap(df, vmin = 0, vmax = 1, center = 0.5,
                    cbar = i == 0, ax=ax, linewidths=0.0, rasterized=True,
                    cbar_ax = None if i else cbar_ax, xticklabels=4,
                    yticklabels=2)
        ax.invert_yaxis()
        ax.set_title(f"Sample {i}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.tight_layout(rect=[0.03, 0.03, .96, 1])
    plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18, x=0.465)
    plt.subplots_adjust(top=0.88, right=0.87)
    fig.text(0.465, 0.02, 'Gamble Probability', ha='center', va='center', fontsize=14)
    fig.text(0.95, 0.5, 'Probability of Choosing Gamble', ha='center', va='center', rotation=270, fontsize=14)
    fig.text(0.02, 0.5, 'Endowment', ha='center', va='center', rotation="vertical", fontsize=14)
    if savePlots:
        fn = os.path.join(plotDir, f"{frame}_max_probGamb_{simLabel}.{fileFormat}")
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if showPlots:
        plt.show()


    # ENTROPY ==================================================================

    # choice entropy -----------------------------------------------------------
    fig, axs = plt.subplots(int(np.ceil(nSamps/5)), 5,
                            figsize=(14.5, int(np.ceil(nSamps/5)*3)),
                            sharex = True, sharey = True)
    cbar_ax = fig.add_axes([.88, .3, .03, .4])
    # tick_locator = ticker.MaxNLocator(5)
    for i, ax in enumerate(axs.flat):
        if i >= nSamps:
            break

        df = pd.DataFrame({"values": mean_entropy_max["choice"][frame][:,:,i].flatten(),
                            "endow": [i for i in endows for j in range(len(probs))],
                            "probs": list(np.round(probs,2))*len(endows)})
        df = df.pivot(index="endow", columns="probs", values="values")

        sns.heatmap(df, vmin = 0, vmax = entropy_discrete(np.array([0.5, 0.5])),
                    cbar = i == 0, ax=ax, linewidths=0.0, rasterized=True,
                    cbar_ax = None if i else cbar_ax, xticklabels=4,
                    yticklabels=2)
        ax.invert_yaxis()
        ax.set_title(f"Sample {i}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.tight_layout(rect=[0.03, 0.03, .96, 1])
    plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18, x=0.465)
    plt.subplots_adjust(top=0.88, right=0.87)
    fig.text(0.465, 0.02, 'Gamble Probability', ha='center', va='center', fontsize=14)
    fig.text(0.95, 0.5, 'Choice Entropy', ha='center', va='center', rotation=270, fontsize=14)
    fig.text(0.02, 0.5, 'Endowment', ha='center', va='center', rotation="vertical", fontsize=14)
    if savePlots:
        fn = os.path.join(plotDir, f"{frame}_max_choiceEntropy_{simLabel}.{fileFormat}")
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if showPlots:
        plt.show()

    # total entropy ------------------------------------------------------------
    # fig, axs = plt.subplots(int(np.ceil(nSamps/5)), 5,
    #                         figsize=(14.5, int(np.ceil(nSamps/5)*3)),
    #                         sharex = True, sharey = True)
    # cbar_ax = fig.add_axes([.88, .3, .03, .4])
    # # tick_locator = ticker.MaxNLocator(5)
    # for i, ax in enumerate(axs.flat):
    #     if i >= nSamps:
    #         break

    #     sns.heatmap(mean_entropy_max["total"][frame][:,:,i], vmin = 0, vmax = 1, center = 0.5,
    #                 cbar = i == 0, ax=ax, linewidths=0.0, rasterized=True,
    #                 cbar_ax = None if i else cbar_ax)
    #     ax.set_yticklabels(endows[0:len(endows)], size = 10)
    #     # ax.yaxis.set_major_locator(tick_locator)
    #     ax.set_xticklabels([str(x.round(1)) if (i % 2) == 0 else "" for i, x in enumerate(probs)], size = 10)
    #     # ax.xaxis.set_major_locator(tick_locator)
    #     ax.invert_yaxis()
    #     ax.set_title(f"Sample {i}")

    # fig.tight_layout(rect=[0.03, 0.03, .96, 1])
    # plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18, x=0.465)
    # plt.subplots_adjust(top=0.88, right=0.87)
    # fig.text(0.465, 0.02, 'Gamble Probability', ha='center', va='center', fontsize=14)
    # fig.text(0.95, 0.5, 'Total Entropy (Choice & Fixation)', ha='center', va='center', rotation=270, fontsize=14)
    # fig.text(0.02, 0.5, 'Endowment', ha='center', va='center', rotation="vertical", fontsize=14)
    # if savePlots:
    #     fn = os.path.join(plotDir, f"{frame}_max_totalEntropy_deadline[{deadline[0]}_{deadline[1]}]_fixTemp[{fix_temp}]_sightBias[{sightBias}].{fileFormat}")
    #     plt.savefig(fn, dpi=300, bbox_inches='tight')
    # if showPlots:
    #     plt.show()


    # SAMPLING PROBABILITIES ===================================================

    # probability of sampling gamble -------------------------------------------
    fig, axs = plt.subplots(int(np.ceil(nSamps/5)), 5,
                            figsize=(14.5, int(np.ceil(nSamps/5)*3)),
                            sharex = True, sharey = True)
    cbar_ax = fig.add_axes([.88, .3, .03, .4])
    # tick_locator = ticker.MaxNLocator(5)
    for i, ax in enumerate(axs.flat):
        if i >= nSamps:
            break

        df = pd.DataFrame({"values": mean_samp_max["prob_gamb"][frame][:,:,i].flatten(),
                            "endow": [i for i in endows for j in range(len(probs))],
                            "probs": list(np.round(probs,2))*len(endows)})
        df = df.pivot(index="endow", columns="probs", values="values")

        sns.heatmap(df, vmin = 0, vmax = 1, center = 0.5,
                    cbar = i == 0, ax=ax, linewidths=0.0, rasterized=True,
                    cbar_ax = None if i else cbar_ax, xticklabels=4,
                    yticklabels=2)
        ax.invert_yaxis()
        ax.set_title(f"Sample {i}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.tight_layout(rect=[0.03, 0.03, .96, 1])
    plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18, x=0.465)
    plt.subplots_adjust(top=0.88, right=0.87)
    fig.text(0.465, 0.02, 'Gamble Probability', ha='center', va='center', fontsize=14)
    fig.text(0.95, 0.5, 'Probability of Fixating Gamble', ha='center', va='center', rotation=270, fontsize=14)
    fig.text(0.02, 0.5, 'Endowment', ha='center', va='center', rotation="vertical", fontsize=14)
    if savePlots:
        fn = os.path.join(plotDir, f"{frame}_max_probSampGamb_{simLabel}.{fileFormat}")
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if showPlots:
        plt.show()


    # probability of switching sampling ----------------------------------------
    fig, axs = plt.subplots(int(np.ceil(nSamps/5)), 5,
                            figsize=(14.5, int(np.ceil(nSamps/5)*3)),
                            sharex = True, sharey = True)
    cbar_ax = fig.add_axes([.88, .3, .03, .4])
    # tick_locator = ticker.MaxNLocator(5)
    for i, ax in enumerate(axs.flat):
        if i >= nSamps:
            break

        df = pd.DataFrame({"values": mean_samp_max["prob_switch"][frame][:,:,i].flatten(),
                            "endow": [i for i in endows for j in range(len(probs))],
                            "probs": list(np.round(probs,2))*len(endows)})
        df = df.pivot(index="endow", columns="probs", values="values")

        sns.heatmap(df, vmin = 0, vmax = 1, center = 0.5,
                    cbar = i == 0, ax=ax, linewidths=0.0, rasterized=True,
                    cbar_ax = None if i else cbar_ax, xticklabels=4,
                    yticklabels=2)
        ax.invert_yaxis()
        ax.set_title(f"Sample {i}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.tight_layout(rect=[0.03, 0.03, .96, 1])
    plt.suptitle(f"{frame.capitalize()} Frame", fontsize=18, x=0.465)
    plt.subplots_adjust(top=0.88, right=0.87)
    fig.text(0.465, 0.02, 'Gamble Probability', ha='center', va='center', fontsize=14)
    fig.text(0.95, 0.5, 'Probability of Switching Fixation', ha='center', va='center', rotation=270, fontsize=14)
    fig.text(0.02, 0.5, 'Endowment', ha='center', va='center', rotation="vertical", fontsize=14)
    if savePlots:
        fn = os.path.join(plotDir, f"{frame}_max_probSampSwitch_{simLabel}.{fileFormat}")
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if showPlots:
        plt.show()