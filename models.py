#
#  models.py
#  
#  Joint MPA and Remote unfished biomass model
# 
#  Created by M. Aaron MacNeil on 11/06/14.
#  Copyright (c) 2014 AIMS. All rights reserved.
#
import sys
import os
from data import *
from pymc import *
import numpy as np
import pdb

#------------------------------------------------ Models
"""
Various models for the paper can be re-constructed here, flagging out various components to end up with the models M0 to M2 below. The basic hierarchical model is the same, with additional nussiance parameters to control for potential biasing factors.

M0 = Hierarchical model with no depth, atoll, mpa size, or hard coral covariates present
M1 = M0 with depth, atoll, mpa size, and hard coral covariates present
M2 = M2 data provider random effect included
"""

#------------------------------------------------ Global priors
# Uninformed
gamma_0 = Uniform('unfished_biomass', lower=1, upper=15, value=6)
# Global intrisic biomass levels - priors from PNAS paper
#gamma_0 = Normal('unfished_biomass', mu=pnas_prior, tau=pnas_tau)
sd_0 = Uniform('gbiom_sd', lower=0., upper=100., value=1.)
tau_0 = Lambda('gbiom_tau', lambda sd=sd_0: sd**-2)

#"""
# Nussiance parameters
beta_1 = Normal('hard_coral', mu=0.0, tau=0.001, value=0.0)
beta_2 = Normal('depth', mu=0.0, tau=0.001, value=0.0)
beta_3 = Normal('atoll', mu=0.0, tau=0.001, value=0.0)
beta_4 = Normal('mres_size', mu=0.0, tau=0.001, value=0.0)
beta_5 = Normal('productivity', mu=0.0, tau=0.001, value=0.0)
#"""

#"""
# Data-provider effect
delta_0 = Normal('delta0', mu=0.0, tau=0.001, value=0.0)
sd_d0 = Uniform('provider_sd', lower=0., upper=100., value=1.)
tau_d0 = Lambda('provider_tau', lambda sd=sd_d0: sd**-2)
rho0 = Normal('provider', mu=delta_0, tau=tau_d0, value=np.ones(nprovider))
rho_0 = Lambda('rho_0', lambda r0=rho0: r0-np.mean(r0))
#"""

#------------------------------------------------ Marine reserve recovery submodel
# Minimum biomass 
gamma_1 = Normal('min_biomass', mu=4.599065, tau=0.01, value=4.599065)
# Rate of intrinsic biomass growth within reserves
gamma_2 = Uniform('intrinsic_growth', lower=-1, upper=3, value=.1)

# Mean model
mu = Lambda('mu', lambda K=gamma_0, r=gamma_2, mb=gamma_1, b1=beta_1, b2=beta_2, b3=beta_3, b4=beta_4, b5=beta_5, r0=rho_0[Imp]: K/(1+((K-mb)/mb)*np.exp(-r*mreserve_age)) + b1*mhard_coral + b2*mdepth + b3*matoll + b4*mreserve_size + b5*mproductivity + r0)
#mu = Lambda('mu', lambda K=gamma_0, r=gamma_2, mb=gamma_1, b1=beta_1, b2=beta_2, b3=beta_3, b4=beta_4, b5=beta_5: K/(1+((K-mb)/mb)*np.exp(-r*mreserve_age)) + b1*mhard_coral + b2*mdepth + b3*matoll + b4*mreserve_size + b5*mproductivity)
#mu = Lambda('mu', lambda K=gamma_0, r=gamma_2, mb=gamma_1: K/(1+((K-mb)/mb)*np.exp(-r*mreserve_age)))

# 'Observation' error
bio_sd = Uniform('bio_sd', lower=0, upper=1000, value=2.) 
bio_tau = Lambda('bio_tau', lambda sd=bio_sd: sd**-2) 
# Likelihood
Yi = Normal('Yi', mu=mu, tau=bio_tau, value=m_lbiomass, observed=True)


#------------------------------------------------ Remote location submodel
# Local variation
sd_lr = Uniform('lr_sd', lower=0., upper=1000., value=1.)
tau_lr = Lambda('lr_tau', lambda sd=sd_lr: sd**-2)
# Location-scale estimates
eta_0 = Normal('eta0', mu=gamma_0, tau=tau_lr, value=np.ones(nrlocation)*6.)
eta0c = [Lambda('eta0__%s'%(rlocation[i]), lambda e0=eta_0[i]: e0) for i in xrange(nrlocation)]

# Mean model 
mu_r = Lambda('mu_r', lambda e0=eta_0[Irl], b1=beta_1, b2=beta_2, b3=beta_3, b5=beta_5, r0=rho_0[Irp]: e0 + b1*rhard_coral + b2*rdepth + b3*ratoll + b5*rproductivity + r0)
#mu_r = Lambda('mu_r', lambda e0=eta_0[Irl], b1=beta_1, b2=beta_2, b3=beta_3, b5=beta_5: e0 + b1*rhard_coral + b2*rdepth + b3*ratoll + b5*rproductivity)
#mu_r = Lambda('mu_r', lambda e0=eta_0[Irl]: e0 )

# 'Observation' error
bior_sd = Uniform('bior_sd', lower=0, upper=1000, value=2.) 
bior_tau = Lambda('bior_tau', lambda sd=bior_sd: sd**-2) 
# Likelihood
Zi = Normal('Zi', mu=mu_r, tau=bior_tau, value=r_lbiomass, observed=True)


#------------------------------------------------ Management type submodel
# Management parent priors
#res_mu= Normal('res_mu', mu=6, tau=0.001, value=6.1)

# Location variability
#sd_resmu = Uniform('resmu_sd', lower=0., upper=1000., value=1.)
#tau_resmu = Lambda('resmu_tau', lambda sd=sd_resmu: sd**-2)
# Location-scale estimates
res_l = Normal('res_location', mu=6., tau=0.01, value=np.ones(nrslocation)*6)
res_lx = [Lambda('res_location__%s'%(res_location[i]), lambda b0=res_l[i], B0=gamma_0: np.exp(b0)/np.exp(B0)) for i in xrange(nrslocation)]

# Mean model
res_mu = Lambda('res_mu', lambda b0=res_l[Irsl], b1=beta_1, b2=beta_2, b3=beta_3, b5=beta_5, r0=rho_0[Izp]: b0 + b1*res_hard_coral + b2*res_depth + b3*res_atoll + b5*res_productivity + r0)
#res_mu = Lambda('res_mu', lambda b0=res_l[Irsl], b1=beta_1, b2=beta_2, b3=beta_3, b5=beta_5: b0 + b1*res_hard_coral + b2*res_depth + b3*res_atoll + b5*res_productivity)
#res_mu = Lambda('res_mu', lambda b0=res_l[Irsl]: b0)

# 'Observation' error
res_sd = Uniform('res_sd', lower=0, upper=1000, value=2.) 
res_tau = Lambda('res_tau', lambda sd=res_sd: sd**-2)
# Likelihood
Gi = Normal('Gi', mu=res_mu, tau=res_tau, value=res_lbiomass, observed=True)


#------------------------------------------------ Fished submodel

# Average fished prior
#zeta_0 = Normal('fished_average', mu=3, tau=0.001, value=3.1)

# Local variation
#sd_lf = Uniform('lf_sd', lower=0., upper=1000., value=1.)
#tau_lf = Lambda('lf_tau', lambda sd=sd_lf: sd**-2)
# Location-scale estimates
kappa_0 = Normal('kappa_0', mu=4.0, tau=0.01, value=np.ones(nfl)*4.0)
Kappa = [Lambda('Kappa__%s'%(flocation[i]), lambda k=kappa_0[i]: k) for i in xrange(nfl)]

# Mean model
mu_k = Lambda('mu_k', lambda k0=kappa_0[Ifl], b1=beta_1, b2=beta_2, b3=beta_3, b5=beta_5, r0=rho_0[Ifp]: k0 + b1*fhard_coral + b2*fdepth + b3*fatoll + b5*fproductivity + r0)
#mu_k = Lambda('mu_k', lambda k0=kappa_0[Ifl], b1=beta_1, b2=beta_2, b3=beta_3, b5=beta_5: k0 + b1*fhard_coral + b2*fdepth + b3*fatoll + b5*fproductivity)
#mu_k = Lambda('mu_k', lambda k0=kappa_0[Ifl]: k0)

# 'Observation' error
fc_sd = Uniform('fc_sd', lower=0, upper=1000, value=10.) 
fc_tau = Lambda('fc_tau', lambda sd=fc_sd: sd**-2)
# Likelihood
Ki = Normal('Ki', mu=mu_k, tau=fc_tau, value=flbiomass, observed=True)


#======================================================= POSTERIORS

#------------------------------------------------ Proportions of B0
# Management
res_mgmtx = [Lambda('mgmt__%s'%(res_mgmt[i]), lambda b0=res_l[Imresl==i], B0=gamma_0: np.exp(np.mean(b0))/np.exp(B0)) for i in xrange(nmgmt)]
# Fished
F_ari = Lambda('F_ari', lambda B0=gamma_0, k0=kappa_0: np.exp(k0)/np.exp(B0), trace=False)
F = [Lambda('F__%s'%(flocation[i]), lambda F=F_ari[i]: F) for i in xrange(nfl)]
# Restricted
R_ari = Lambda('R_ari', lambda B0=gamma_0, l0=res_l: np.exp(l0)/np.exp(B0), trace=False)
R = [Lambda('R__%s'%(res_location[i]), lambda R=R_ari[i]: R) for i in xrange(nrslocation)]

#------------------------------------------------ Recovery times
# Time to recovery from initial biomass
@deterministic(plot=False)
def AR_90(K=gamma_0, r=gamma_2, mb=gamma_1):
    # Log-scale equivalient of 90% of arithmetic B0
    k0 = np.log(.9*np.exp(K))
    return np.log(((K/k0)-1)/((K-mb)/mb))/-r

# Virtual reserve age - fished
@deterministic(plot=False)
def VAf(K=gamma_0, r=gamma_2, mb=gamma_1, K0=kappa_0):
    # Keep things sane incase of random zeros
    k0 = array([min(max(mb,k),K) for k in K0])
    return np.log(((K/k0)-1)/((K-mb)/mb))/-r

# Virtual reserve age - restricted
@deterministic(plot=False)
def VAr(K=gamma_0, r=gamma_2, mb=gamma_1, K0=res_l):
    # Keep things sane incase of random zeros
    k0 = array([min(max(mb,k),K) for k in K0])
    return np.log(((K/k0)-1)/((K-mb)/mb))/-r

# Times to recovery - fished
TR90_f = Lambda('TR90_f', lambda t90=AR_90, vage=VAf: t90-vage)
TR90x = [Lambda('fTR90__%s'%(flocation[i]), lambda tr=TR90_f[i]: tr) for i in xrange(nfl)]
# Times to recovery - restricted
TR90_res = Lambda('TR90_res', lambda t90=AR_90, vage=VAr: t90-vage)
TR90y = [Lambda('rTR90__%s'%(res_location[i]), lambda tr=TR90_res[i]: tr) for i in xrange(nrslocation)]

#======================================================= PLOTTING POSTERIORS
#"""
# Marine reserve recovery
global_model = Lambda('global_model', lambda K=gamma_0, r=gamma_2, mb=gamma_1: 
K/(1+((K-mb)/mb)*np.exp(-r*pred_x)))
# Unfished biomass - arithmetic
global_B0 = Lambda('global_B0', lambda b0=gamma_0: np.exp(b0))
#"""










