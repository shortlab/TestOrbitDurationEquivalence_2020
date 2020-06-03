#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys
#!{sys.executable} -m pip install --upgrade uncertainties
#!{sys.executable} -m pip install --upgrade periodictable
#!{sys.executable} -m pip install --upgrade brewer2mpl
#!{sys.executable} -m pip install PAScual
import matplotlib.gridspec as gridspec
import numpy.random
import scipy
import scipy.optimize
import scipy.constants
import math
import numpy 
import uncertainties
import uncertainties.unumpy 

import matplotlib.axes
import matplotlib.pyplot
import scipy.misc
import scipy.special
import scipy.stats
from scipy.constants import *
import random as random
from scipy.special import xlogy
import sys
import scipy.stats 
from scipy.interpolate import interp1d

import csv
 import seaborn               


import pandas as pd

from scipy.optimize import curve_fit


from math import *

from scipy.optimize import curve_fit

import periodictable

from periodictable import elements as periodictable_elements

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl

a = matplotlib.cm.get_cmap(name='tab10')
colors = a.colors
colors = numpy.concatenate((colors, colors))

a2 = matplotlib.cm.get_cmap(name='tab20')
colors2 = a2.colors

styles = ['-', '--', '-.', ':']


material_densities =     {"CdTe": 5.85, "ZnS": 4.09, "ZnSe": 5.27, "ZnTe": 6.34, "InSb": 5.78, "In5Ga5As": 5.5, "In5Ga5Sb":  5.7, "Hg5Cd5Te": 8, "GaN":6.15, "ZnO":5.7, "MgO":3.58,  "GaP":4.14, "InAs":5.67, "InP":4.81, "Si":2.32}


# In[2]:


# Read in energy deposited per 10 micron nuclear and electronic for all energy bins in each spectrum

fn = 'nuclear_E_loss_vsEin_vsDistance_Al_shielded.csv'
shielded_Ebin_edep_n = pd.read_csv(fn, sep=',', names=['material', 'particle', 'E_bin_MeV', 'dist_um', 'E_dep_MeV'], engine='python')
fn = 'electronic_E_loss_vsEin_vsDistance_Al_shielded.csv'
shielded_Ebin_edep_e = pd.read_csv(fn, sep=',', names=['material', 'particle', 'E_bin_MeV', 'dist_um', 'E_dep_MeV'], engine='python')
fn = 'nuclear_E_loss_vsEin_vsDistance_unshielded.csv'
unshielded_Ebin_edep_n = pd.read_csv(fn, sep=',', names=['material', 'particle', 'E_bin_MeV', 'dist_um', 'E_dep_MeV'], engine='python')
fn = 'electronic_E_loss_vsEin_vsDistance_unshielded.csv'
unshielded_Ebin_edep_e = pd.read_csv(fn, sep=',', names=['material', 'particle', 'E_bin_MeV', 'dist_um', 'E_dep_MeV'], engine='python')
fn = 'nuclear_E_loss_vsEin_vsDistance_test.csv'
test_Ebin_edep_n = pd.read_csv(fn, sep=',', names=['material', 'particle', 'E_bin_MeV', 'dist_um', 'E_dep_MeV'], engine='python')
fn = 'electronic_E_loss_vsEin_vsDistance_test.csv'
test_Ebin_edep_e = pd.read_csv(fn, sep=',', names=['material', 'particle', 'E_bin_MeV', 'dist_um', 'E_dep_MeV'], engine='python')

# goes from here to the end, this is used to distinguish the sources of He
startHeE = 4.0*1.0000E-01
stopHeE =  4.0*4.5000E+02

# distinguish for solar and gcr He, all H summed for all sources anyway as are the spectra so it doesn't matter

nn = unshielded_Ebin_edep_n[(unshielded_Ebin_edep_n.E_bin_MeV == startHeE) & (unshielded_Ebin_edep_n.particle == 'He')]
solarHe_startIndex = nn.index[0]
nn = unshielded_Ebin_edep_n[(unshielded_Ebin_edep_n.E_bin_MeV == stopHeE) & (unshielded_Ebin_edep_n.particle == 'He')]
solarHe_stopIndex = nn.index[-1]


# In[3]:


# READ IN THE SPECTRA

mission_time_s = 10*365*24*60*60

filenames = {
'LEO_600':'LEO_600.txt',
'LEO_800':'LEO_800.txt',
'LEO_ISS':'LEO_ISS.txt',
'MEO_MNY':'MEO_MNY.txt',
'MEO_GPS':'MEO_GPS.txt',
'HEO_IBEX':'HEO_IBEX.txt',
'HEO_GEOSTAT':'HEO_GEOSTAT.txt'
}

orbits_legend = {
'LEO_600': "LEO Polar Sun-synchronous 600km",
'LEO_800': "LEO Polar Sun-synchronous 800km",
'LEO_ISS': "LEO Inclined Nonpolar ISS",
'MEO_MNY': "MEO Molniya",
'MEO_GPS': "MEO Semi-synchronous GPS",
'HEO_IBEX': "HEO Highly-eccentric IBEX",
'HEO_GEOSTAT': "HEO Geostationary"
}

# Unshielded Trapped spectra - protons and electrons

unshielded_trapped_protons = {}
unshielded_trapped_electrons = {}
filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Trapped/'
for keyvalue, filename in filenames.items():
    fn = filepath + filename
    
    proton_spectrum = pd.read_csv(fn, sep=',', skiprows=75, skipfooter=107, names=['Energy_MeV', 'IFlux_cm2s', 'DFlux_MeVcm2s'], engine='python')
    E_width = numpy.diff(proton_spectrum.Energy_MeV, prepend=0)
    proton_spectrum['DFlux_cm2s'] = proton_spectrum.DFlux_MeVcm2s*E_width
    unshielded_trapped_protons[keyvalue] = proton_spectrum
    
    electron_spectrum = pd.read_csv(fn, sep=',', skiprows=180, skipfooter=1, names=['Energy_MeV', 'IFlux_cm2s', 'DFlux_MeVcm2s'], engine='python')
    E_width = numpy.diff(electron_spectrum.Energy_MeV, prepend=0)
    electron_spectrum['DFlux_cm2s'] = electron_spectrum.DFlux_MeVcm2s*E_width
    unshielded_trapped_electrons[keyvalue] = electron_spectrum

# Unshielded no magnetosphere shielding solar spectra - protons

unshielded_solar_protons_noMagnetosphere = {}
filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Solar/WITHOUT_MAGNETOSPHERE_SHIELDING/'
for keyvalue, filename in filenames.items():
    fn = filepath + filename    
    proton_spectrum = pd.read_csv(fn, sep=',', skiprows=54, skipfooter=129, names=['Energy_MeV', 'IFlux_cm2', 'DFlux_MeVcm2', 'AttenFactor', 'Exposure_Hrs'], engine='python')
    E_width = numpy.diff(proton_spectrum.Energy_MeV, prepend=0)
    proton_spectrum['DFlux_cm2'] = proton_spectrum.DFlux_MeVcm2*E_width
    proton_spectrum['DFlux_MeVcm2s'] = proton_spectrum.DFlux_MeVcm2/mission_time_s
    proton_spectrum['DFlux_cm2s'] = proton_spectrum['DFlux_cm2']/mission_time_s
    unshielded_solar_protons_noMagnetosphere[keyvalue] = proton_spectrum
    
# Unshielded yes magnetosphere shielding solar spectra - protons
    
unshielded_solar_protons_yesMagnetosphere = {}
unshielded_solar_He_yesMagnetosphere = {}
filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Solar/WITH_MAGNETOSPHERE_SHIELDING/'
for keyvalue, filename in filenames.items():
    fn = filepath + filename    
    proton_spectrum = pd.read_csv(fn, sep=',', skiprows=54, skipfooter=129, names=['Energy_MeV', 'IFlux_cm2', 'DFlux_MeVcm2', 'AttenFactor', 'Exposure_Hrs'], engine='python')
    E_width = numpy.diff(proton_spectrum.Energy_MeV, prepend=0)
    proton_spectrum['DFlux_cm2'] = proton_spectrum.DFlux_MeVcm2*E_width
    proton_spectrum['DFlux_MeVcm2s'] = proton_spectrum.DFlux_MeVcm2/mission_time_s
    proton_spectrum['DFlux_cm2s'] = proton_spectrum['DFlux_cm2']/mission_time_s
    unshielded_solar_protons_yesMagnetosphere[keyvalue] = proton_spectrum
    
    ion_spectrum = pd.read_csv(fn, sep=',', skiprows=182, skipfooter=2, names=['Energy_MeV_n', "H_IFlux_cm2", "He_IFlux_cm2", "Li_IFlux_cm2", "Be_IFlux_cm2", "B_IFlux_cm2", "C_IFlux_cm2", "N_IFlux_cm2", "O_IFlux_cm2", "F_IFlux_cm2", "H_DFlux_cm2MeV_n", "He_DFlux_cm2MeV_n", "Li_DFlux_cm2MeV_n", "Be_DFlux_cm2MeV_n", "B_DFlux_cm2MeV_n", "C_DFlux_cm2MeV_n", "N_DFlux_cm2MeV_n", "O_DFlux_cm2MeV_n", "F_DFlux_cm2MeV_n"], engine='python')
    # For He
    n = 4
    energy_data = {'Energy_MeV': numpy.array(ion_spectrum["Energy_MeV_n"])*n}
    He_spectrum = pd.DataFrame(energy_data)
    E_width = numpy.diff(He_spectrum['Energy_MeV'], prepend=0)
    He_spectrum['DFlux_cm2'] = ion_spectrum.He_DFlux_cm2MeV_n*(1/n)*E_width
    He_spectrum['DFlux_MeVcm2s'] = ion_spectrum.He_DFlux_cm2MeV_n*(1.0/mission_time_s)*(1/n)
    He_spectrum['DFlux_cm2s'] = He_spectrum['DFlux_cm2']/mission_time_s  
    unshielded_solar_He_yesMagnetosphere[keyvalue] = He_spectrum
 
# All unshielded, no magnetosphere GCR ions
                                                                             
unshielded_H_galactic_noMagnetosphere = {}
unshielded_He_galactic_noMagnetosphere = {}
unshielded_C_galactic_noMagnetosphere = {}
unshielded_O_galactic_noMagnetosphere = {}
unshielded_N_galactic_noMagnetosphere = {}
unshielded_Ne_galactic_noMagnetosphere = {}
unshielded_Mg_galactic_noMagnetosphere = {}
unshielded_Fe_galactic_noMagnetosphere = {}
unshielded_Si_galactic_noMagnetosphere = {}

filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/GCR/WITHOUT_MAGNETOSPHERE_SHIELDING_CREME96/'
for keyvalue, filename in filenames.items():
    fn = filepath + filename    
    all_spectrum = pd.read_csv(fn, sep=',', skiprows=89, skipfooter=1, names=['Energy_MeV', "H_IFlux_m2sSr", "He_IFlux_m2sSr", "Li_IFlux_m2sSr", "Be_IFlux_m2sSr", "B_IFlux_m2sSr", "C_IFlux_m2sSr", "N_IFlux_m2sSr", "O_IFlux_m2sSr", "F_IFlux_m2sSr", "Ne_IFlux_m2sSr", "Na_IFlux_m2sSr", "Mg_IFlux_m2sSr", "Al_IFlux_m2sSr", "Si_IFlux_m2sSr", "P_IFlux_m2sSr", "S_IFlux_m2sSr", "Cl_IFlux_m2sSr", "Ar_IFlux_m2sSr", "K_IFlux_m2sSr", "Ca_IFlux_m2sSr", "Sc_IFlux_m2sSr", "Ti_IFlux_m2sSr", "V_IFlux_m2sSr", "Cr_IFlux_m2sSr", "Mn_IFlux_m2sSr", "Fe_IFlux_m2sSr", "Co_IFlux_m2sSr", "Ni_IFlux_m2sSr", "Cu_IFlux_m2sSr", "Zn_IFlux_m2sSr", "Ga_IFlux_m2sSr", "Ge_IFlux_m2sSr", "As_IFlux_m2sSr", "Se_IFlux_m2sSr", "Br_IFlux_m2sSr", "Kr_IFlux_m2sSr", "Rb_IFlux_m2sSr", "Sr_IFlux_m2sSr", "Y_IFlux_m2sSr", "Zr_IFlux_m2sSr", "Nb_IFlux_m2sSr", "Mo_IFlux_m2sSr", "Tc_IFlux_m2sSr", "Ru_IFlux_m2sSr", "Rh_IFlux_m2sSr", "Pd_IFlux_m2sSr", "Ag_IFlux_m2sSr", "Cd_IFlux_m2sSr", "In_IFlux_m2sSr", "Sn_IFlux_m2sSr", "Sb_IFlux_m2sSr", "Te_IFlux_m2sSr", "I_IFlux_m2sSr", "Xe_IFlux_m2sSr", "Cs_IFlux_m2sSr", "Ba_IFlux_m2sSr", "La_IFlux_m2sSr", "Ce_IFlux_m2sSr", "Pr_IFlux_m2sSr", "Nd_IFlux_m2sSr", "Pm_IFlux_m2sSr", "Sm_IFlux_m2sSr", "Eu_IFlux_m2sSr", "Gd_IFlux_m2sSr", "Tb_IFlux_m2sSr", "Dy_IFlux_m2sSr", "Ho_IFlux_m2sSr", "Er_IFlux_m2sSr", "Tm_IFlux_m2sSr", "Yb_IFlux_m2sSr", "Lu_IFlux_m2sSr", "Hf_IFlux_m2sSr", "Ta_IFlux_m2sSr", "W_IFlux_m2sSr", "Re_IFlux_m2sSr", "Os_IFlux_m2sSr", "Ir_IFlux_m2sSr", "Pt_IFlux_m2sSr", "Au_IFlux_m2sSr", "Hg_IFlux_m2sSr", "Tl_IFlux_m2sSr", "Pb_IFlux_m2sSr", "Bi_IFlux_m2sSr", "Po_IFlux_m2sSr", "At_IFlux_m2sSr", "Rn_IFlux_m2sSr", "Fr_IFlux_m2sSr", "Ra_IFlux_m2sSr", "Ac_IFlux_m2sSr", "Th_IFlux_m2sSr", "Pa_IFlux_m2sSr", "U_IFlux_m2sSr", "H_DFlux_m2sSrMeVu", "He_DFlux_m2sSrMeVu", "Li_DFlux_m2sSrMeVu", "Be_DFlux_m2sSrMeVu", "B_DFlux_m2sSrMeVu", "C_DFlux_m2sSrMeVu", "N_DFlux_m2sSrMeVu", "O_DFlux_m2sSrMeVu", "F_DFlux_m2sSrMeVu", "Ne_DFlux_m2sSrMeVu", "Na_DFlux_m2sSrMeVu", "Mg_DFlux_m2sSrMeVu", "Al_DFlux_m2sSrMeVu", "Si_DFlux_m2sSrMeVu", "P_DFlux_m2sSrMeVu", "S_DFlux_m2sSrMeVu", "Cl_DFlux_m2sSrMeVu", "Ar_DFlux_m2sSrMeVu", "K_DFlux_m2sSrMeVu", "Ca_DFlux_m2sSrMeVu", "Sc_DFlux_m2sSrMeVu", "Ti_DFlux_m2sSrMeVu", "V_DFlux_m2sSrMeVu", "Cr_DFlux_m2sSrMeVu", "Mn_DFlux_m2sSrMeVu", "Fe_DFlux_m2sSrMeVu", "Co_DFlux_m2sSrMeVu", "Ni_DFlux_m2sSrMeVu", "Cu_DFlux_m2sSrMeVu", "Zn_DFlux_m2sSrMeVu", "Ga_DFlux_m2sSrMeVu", "Ge_DFlux_m2sSrMeVu", "As_DFlux_m2sSrMeVu", "Se_DFlux_m2sSrMeVu", "Br_DFlux_m2sSrMeVu", "Kr_DFlux_m2sSrMeVu", "Rb_DFlux_m2sSrMeVu", "Sr_DFlux_m2sSrMeVu", "Y_DFlux_m2sSrMeVu", "Zr_DFlux_m2sSrMeVu", "Nb_DFlux_m2sSrMeVu", "Mo_DFlux_m2sSrMeVu", "Tc_DFlux_m2sSrMeVu", "Ru_DFlux_m2sSrMeVu", "Rh_DFlux_m2sSrMeVu", "Pd_DFlux_m2sSrMeVu", "Ag_DFlux_m2sSrMeVu", "Cd_DFlux_m2sSrMeVu", "In_DFlux_m2sSrMeVu", "Sn_DFlux_m2sSrMeVu", "Sb_DFlux_m2sSrMeVu", "Te_DFlux_m2sSrMeVu", "I_DFlux_m2sSrMeVu", "Xe_DFlux_m2sSrMeVu", "Cs_DFlux_m2sSrMeVu", "Ba_DFlux_m2sSrMeVu", "La_DFlux_m2sSrMeVu", "Ce_DFlux_m2sSrMeVu", "Pr_DFlux_m2sSrMeVu", "Nd_DFlux_m2sSrMeVu", "Pm_DFlux_m2sSrMeVu", "Sm_DFlux_m2sSrMeVu", "Eu_DFlux_m2sSrMeVu", "Gd_DFlux_m2sSrMeVu", "Tb_DFlux_m2sSrMeVu", "Dy_DFlux_m2sSrMeVu", "Ho_DFlux_m2sSrMeVu", "Er_DFlux_m2sSrMeVu", "Tm_DFlux_m2sSrMeVu", "Yb_DFlux_m2sSrMeVu", "Lu_DFlux_m2sSrMeVu", "Hf_DFlux_m2sSrMeVu", "Ta_DFlux_m2sSrMeVu", "W_DFlux_m2sSrMeVu", "Re_DFlux_m2sSrMeVu", "Os_DFlux_m2sSrMeVu", "Ir_DFlux_m2sSrMeVu", "Pt_DFlux_m2sSrMeVu", "Au_DFlux_m2sSrMeVu", "Hg_DFlux_m2sSrMeVu", "Tl_DFlux_m2sSrMeVu", "Pb_DFlux_m2sSrMeVu", "Bi_DFlux_m2sSrMeVu", "Po_DFlux_m2sSrMeVu", "At_DFlux_m2sSrMeVu", "Rn_DFlux_m2sSrMeVu", "Fr_DFlux_m2sSrMeVu", "Ra_DFlux_m2sSrMeVu", "Ac_DFlux_m2sSrMeVu", "Th_DFlux_m2sSrMeVu", "Pa_DFlux_m2sSrMeVu", "U_DFlux_m2sSrMeVu"], engine='python')
    E_width = numpy.diff(all_spectrum.Energy_MeV, prepend=0)
    all_spectrum["H_DFlux_m2sSr"] = numpy.array(all_spectrum['H_DFlux_m2sSrMeVu'])*E_width
    all_spectrum["He_DFlux_m2sSr"] = numpy.array(all_spectrum['He_DFlux_m2sSrMeVu'])*E_width*2.0
    all_spectrum["C_DFlux_m2sSr"] = numpy.array(all_spectrum['C_DFlux_m2sSrMeVu'])*E_width*6.0
    all_spectrum["O_DFlux_m2sSr"] = numpy.array(all_spectrum['O_DFlux_m2sSrMeVu'])*E_width*8.0
    energy_data = {'Energy_MeV': numpy.array(all_spectrum["Energy_MeV"])}
    H_spectrum = pd.DataFrame(energy_data)
    H_spectrum['H_DFlux_m2sSrMeVu'] = all_spectrum['H_DFlux_m2sSrMeVu']
    H_spectrum["H_DFlux_m2sSr"] = all_spectrum["H_DFlux_m2sSr"]
    H_spectrum["H_IFlux_m2sSr"] = all_spectrum["H_IFlux_m2sSr"]
    He_spectrum = pd.DataFrame(energy_data)
    He_spectrum['He_DFlux_m2sSrMeVu'] = all_spectrum['He_DFlux_m2sSrMeVu']
    He_spectrum["He_DFlux_m2sSr"] = all_spectrum["He_DFlux_m2sSr"]
    He_spectrum["He_IFlux_m2sSr"] = all_spectrum["He_IFlux_m2sSr"]
    O_spectrum = pd.DataFrame(energy_data)
    O_spectrum['O_DFlux_m2sSrMeVu'] = all_spectrum['O_DFlux_m2sSrMeVu']
    O_spectrum["O_DFlux_m2sSr"] = all_spectrum["O_DFlux_m2sSr"]
    O_spectrum["O_IFlux_m2sSr"] = all_spectrum["O_IFlux_m2sSr"]
    C_spectrum = pd.DataFrame(energy_data)
    C_spectrum['C_DFlux_m2sSrMeVu'] = all_spectrum['C_DFlux_m2sSrMeVu']
    C_spectrum["C_DFlux_m2sSr"] = all_spectrum["C_DFlux_m2sSr"]
    C_spectrum["C_IFlux_m2sSr"] = all_spectrum["C_IFlux_m2sSr"]    
    unshielded_H_galactic_noMagnetosphere[keyvalue] = H_spectrum
    unshielded_He_galactic_noMagnetosphere[keyvalue] = He_spectrum
    unshielded_C_galactic_noMagnetosphere[keyvalue] = C_spectrum
    unshielded_O_galactic_noMagnetosphere[keyvalue] = O_spectrum
    
# All unshielded, yes magnetosphere GCR ions

unshielded_H_galactic_yesMagnetosphere = {}
unshielded_He_galactic_yesMagnetosphere = {}
unshielded_C_galactic_yesMagnetosphere = {}
unshielded_O_galactic_yesMagnetosphere = {}
unshielded_N_galactic_yesMagnetosphere = {}
unshielded_Ne_galactic_yesMagnetosphere = {}
unshielded_Mg_galactic_yesMagnetosphere = {}
unshielded_Fe_galactic_yesMagnetosphere = {}
unshielded_Si_galactic_yesMagnetosphere = {}

filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/GCR/WITH_MAGNETOSPHERE_SHIELDING_ISO_15390/'
for keyvalue, filename in filenames.items():
    fn = filepath + filename   
    # note energy is energy MeV per nucleon and differential flux is per MeV per nucleon, must correct
    all_spectrum = pd.read_csv(fn, sep=',', skiprows=89, skipfooter=1, names=['Energy_MeV', "H_IFlux_m2sSr", "He_IFlux_m2sSr", "Li_IFlux_m2sSr", "Be_IFlux_m2sSr", "B_IFlux_m2sSr", "C_IFlux_m2sSr", "N_IFlux_m2sSr", "O_IFlux_m2sSr", "F_IFlux_m2sSr", "Ne_IFlux_m2sSr", "Na_IFlux_m2sSr", "Mg_IFlux_m2sSr", "Al_IFlux_m2sSr", "Si_IFlux_m2sSr", "P_IFlux_m2sSr", "S_IFlux_m2sSr", "Cl_IFlux_m2sSr", "Ar_IFlux_m2sSr", "K_IFlux_m2sSr", "Ca_IFlux_m2sSr", "Sc_IFlux_m2sSr", "Ti_IFlux_m2sSr", "V_IFlux_m2sSr", "Cr_IFlux_m2sSr", "Mn_IFlux_m2sSr", "Fe_IFlux_m2sSr", "Co_IFlux_m2sSr", "Ni_IFlux_m2sSr", "Cu_IFlux_m2sSr", "Zn_IFlux_m2sSr", "Ga_IFlux_m2sSr", "Ge_IFlux_m2sSr", "As_IFlux_m2sSr", "Se_IFlux_m2sSr", "Br_IFlux_m2sSr", "Kr_IFlux_m2sSr", "Rb_IFlux_m2sSr", "Sr_IFlux_m2sSr", "Y_IFlux_m2sSr", "Zr_IFlux_m2sSr", "Nb_IFlux_m2sSr", "Mo_IFlux_m2sSr", "Tc_IFlux_m2sSr", "Ru_IFlux_m2sSr", "Rh_IFlux_m2sSr", "Pd_IFlux_m2sSr", "Ag_IFlux_m2sSr", "Cd_IFlux_m2sSr", "In_IFlux_m2sSr", "Sn_IFlux_m2sSr", "Sb_IFlux_m2sSr", "Te_IFlux_m2sSr", "I_IFlux_m2sSr", "Xe_IFlux_m2sSr", "Cs_IFlux_m2sSr", "Ba_IFlux_m2sSr", "La_IFlux_m2sSr", "Ce_IFlux_m2sSr", "Pr_IFlux_m2sSr", "Nd_IFlux_m2sSr", "Pm_IFlux_m2sSr", "Sm_IFlux_m2sSr", "Eu_IFlux_m2sSr", "Gd_IFlux_m2sSr", "Tb_IFlux_m2sSr", "Dy_IFlux_m2sSr", "Ho_IFlux_m2sSr", "Er_IFlux_m2sSr", "Tm_IFlux_m2sSr", "Yb_IFlux_m2sSr", "Lu_IFlux_m2sSr", "Hf_IFlux_m2sSr", "Ta_IFlux_m2sSr", "W_IFlux_m2sSr", "Re_IFlux_m2sSr", "Os_IFlux_m2sSr", "Ir_IFlux_m2sSr", "Pt_IFlux_m2sSr", "Au_IFlux_m2sSr", "Hg_IFlux_m2sSr", "Tl_IFlux_m2sSr", "Pb_IFlux_m2sSr", "Bi_IFlux_m2sSr", "Po_IFlux_m2sSr", "At_IFlux_m2sSr", "Rn_IFlux_m2sSr", "Fr_IFlux_m2sSr", "Ra_IFlux_m2sSr", "Ac_IFlux_m2sSr", "Th_IFlux_m2sSr", "Pa_IFlux_m2sSr", "U_IFlux_m2sSr", "H_DFlux_m2sSrMeVu", "He_DFlux_m2sSrMeVu", "Li_DFlux_m2sSrMeVu", "Be_DFlux_m2sSrMeVu", "B_DFlux_m2sSrMeVu", "C_DFlux_m2sSrMeVu", "N_DFlux_m2sSrMeVu", "O_DFlux_m2sSrMeVu", "F_DFlux_m2sSrMeVu", "Ne_DFlux_m2sSrMeVu", "Na_DFlux_m2sSrMeVu", "Mg_DFlux_m2sSrMeVu", "Al_DFlux_m2sSrMeVu", "Si_DFlux_m2sSrMeVu", "P_DFlux_m2sSrMeVu", "S_DFlux_m2sSrMeVu", "Cl_DFlux_m2sSrMeVu", "Ar_DFlux_m2sSrMeVu", "K_DFlux_m2sSrMeVu", "Ca_DFlux_m2sSrMeVu", "Sc_DFlux_m2sSrMeVu", "Ti_DFlux_m2sSrMeVu", "V_DFlux_m2sSrMeVu", "Cr_DFlux_m2sSrMeVu", "Mn_DFlux_m2sSrMeVu", "Fe_DFlux_m2sSrMeVu", "Co_DFlux_m2sSrMeVu", "Ni_DFlux_m2sSrMeVu", "Cu_DFlux_m2sSrMeVu", "Zn_DFlux_m2sSrMeVu", "Ga_DFlux_m2sSrMeVu", "Ge_DFlux_m2sSrMeVu", "As_DFlux_m2sSrMeVu", "Se_DFlux_m2sSrMeVu", "Br_DFlux_m2sSrMeVu", "Kr_DFlux_m2sSrMeVu", "Rb_DFlux_m2sSrMeVu", "Sr_DFlux_m2sSrMeVu", "Y_DFlux_m2sSrMeVu", "Zr_DFlux_m2sSrMeVu", "Nb_DFlux_m2sSrMeVu", "Mo_DFlux_m2sSrMeVu", "Tc_DFlux_m2sSrMeVu", "Ru_DFlux_m2sSrMeVu", "Rh_DFlux_m2sSrMeVu", "Pd_DFlux_m2sSrMeVu", "Ag_DFlux_m2sSrMeVu", "Cd_DFlux_m2sSrMeVu", "In_DFlux_m2sSrMeVu", "Sn_DFlux_m2sSrMeVu", "Sb_DFlux_m2sSrMeVu", "Te_DFlux_m2sSrMeVu", "I_DFlux_m2sSrMeVu", "Xe_DFlux_m2sSrMeVu", "Cs_DFlux_m2sSrMeVu", "Ba_DFlux_m2sSrMeVu", "La_DFlux_m2sSrMeVu", "Ce_DFlux_m2sSrMeVu", "Pr_DFlux_m2sSrMeVu", "Nd_DFlux_m2sSrMeVu", "Pm_DFlux_m2sSrMeVu", "Sm_DFlux_m2sSrMeVu", "Eu_DFlux_m2sSrMeVu", "Gd_DFlux_m2sSrMeVu", "Tb_DFlux_m2sSrMeVu", "Dy_DFlux_m2sSrMeVu", "Ho_DFlux_m2sSrMeVu", "Er_DFlux_m2sSrMeVu", "Tm_DFlux_m2sSrMeVu", "Yb_DFlux_m2sSrMeVu", "Lu_DFlux_m2sSrMeVu", "Hf_DFlux_m2sSrMeVu", "Ta_DFlux_m2sSrMeVu", "W_DFlux_m2sSrMeVu", "Re_DFlux_m2sSrMeVu", "Os_DFlux_m2sSrMeVu", "Ir_DFlux_m2sSrMeVu", "Pt_DFlux_m2sSrMeVu", "Au_DFlux_m2sSrMeVu", "Hg_DFlux_m2sSrMeVu", "Tl_DFlux_m2sSrMeVu", "Pb_DFlux_m2sSrMeVu", "Bi_DFlux_m2sSrMeVu", "Po_DFlux_m2sSrMeVu", "At_DFlux_m2sSrMeVu", "Rn_DFlux_m2sSrMeVu", "Fr_DFlux_m2sSrMeVu", "Ra_DFlux_m2sSrMeVu", "Ac_DFlux_m2sSrMeVu", "Th_DFlux_m2sSrMeVu", "Pa_DFlux_m2sSrMeVu", "U_DFlux_m2sSrMeVu"], engine='python')
    # this is MeV/n with n = number of nucleons
    
    energy_data = {'Energy_MeV': numpy.array(all_spectrum["Energy_MeV"])}
    n = 1
    H_spectrum = pd.DataFrame(energy_data)*n
    H_spectrum['H_DFlux_m2sSrMeV'] = all_spectrum['H_DFlux_m2sSrMeVu']*(1/n)
    H_spectrum["H_IFlux_m2sSr"] = all_spectrum["H_IFlux_m2sSr"]
    n = 4
    He_spectrum = pd.DataFrame(energy_data)*n
    He_spectrum['He_DFlux_m2sSrMeV'] = all_spectrum['He_DFlux_m2sSrMeVu']*(1/n)
    He_spectrum["He_IFlux_m2sSr"] = all_spectrum["He_IFlux_m2sSr"]
    n = 12
    C_spectrum = pd.DataFrame(energy_data)*n
    C_spectrum['C_DFlux_m2sSrMeV'] = all_spectrum['C_DFlux_m2sSrMeVu']*(1/n)
    C_spectrum["C_IFlux_m2sSr"] = all_spectrum["C_IFlux_m2sSr"]
    n = 14
    N_spectrum = pd.DataFrame(energy_data)*n
    N_spectrum['N_DFlux_m2sSrMeV'] = all_spectrum['N_DFlux_m2sSrMeVu']*(1/n)
    N_spectrum["N_IFlux_m2sSr"] = all_spectrum["N_IFlux_m2sSr"]
    n = 16
    O_spectrum = pd.DataFrame(energy_data)*n
    O_spectrum['O_DFlux_m2sSrMeV'] = all_spectrum['O_DFlux_m2sSrMeVu']*(1/n)
    O_spectrum["O_IFlux_m2sSr"] = all_spectrum["O_IFlux_m2sSr"]
    
    n = 20
    Ne_spectrum = pd.DataFrame(energy_data)*n
    Ne_spectrum['Ne_DFlux_m2sSrMeV'] = all_spectrum['Ne_DFlux_m2sSrMeVu']*(1/n)
    Ne_spectrum["Ne_IFlux_m2sSr"] = all_spectrum["Ne_IFlux_m2sSr"]
    n = 24
    Mg_spectrum = pd.DataFrame(energy_data)*n
    Mg_spectrum['Mg_DFlux_m2sSrMeV'] = all_spectrum['Mg_DFlux_m2sSrMeVu']*(1/n)
    Mg_spectrum["Mg_IFlux_m2sSr"] = all_spectrum["Mg_IFlux_m2sSr"]
    n = 28
    Si_spectrum = pd.DataFrame(energy_data)*n
    Si_spectrum['Si_DFlux_m2sSrMeV'] = all_spectrum['Si_DFlux_m2sSrMeVu']*(1/n)
    Si_spectrum["Si_IFlux_m2sSr"] = all_spectrum["Si_IFlux_m2sSr"]
    n = 56
    Fe_spectrum = pd.DataFrame(energy_data)*n
    Fe_spectrum['Fe_DFlux_m2sSrMeV'] = all_spectrum['Fe_DFlux_m2sSrMeVu']*(1/n)
    Fe_spectrum["Fe_IFlux_m2sSr"] = all_spectrum["Fe_IFlux_m2sSr"]
      
      
    
    unshielded_H_galactic_yesMagnetosphere[keyvalue] = H_spectrum
    unshielded_He_galactic_yesMagnetosphere[keyvalue] = He_spectrum
    unshielded_C_galactic_yesMagnetosphere[keyvalue] = C_spectrum
    unshielded_N_galactic_yesMagnetosphere[keyvalue] = N_spectrum
    unshielded_O_galactic_yesMagnetosphere[keyvalue] = O_spectrum

    unshielded_Ne_galactic_yesMagnetosphere[keyvalue] = Ne_spectrum
    unshielded_Mg_galactic_yesMagnetosphere[keyvalue] = Mg_spectrum
    unshielded_Si_galactic_yesMagnetosphere[keyvalue] = Si_spectrum
    unshielded_Fe_galactic_yesMagnetosphere[keyvalue] = Fe_spectrum
    
# Put into an array
                                                                               
unshielded_ion_galactic_yesMagnetosphere = {}                                                                    
unshielded_ion_galactic_yesMagnetosphere['H'] = unshielded_H_galactic_yesMagnetosphere
unshielded_ion_galactic_yesMagnetosphere['He'] = unshielded_He_galactic_yesMagnetosphere                                                                  
unshielded_ion_galactic_yesMagnetosphere['C'] = unshielded_C_galactic_yesMagnetosphere
unshielded_ion_galactic_yesMagnetosphere['N'] = unshielded_N_galactic_yesMagnetosphere                                                                        
unshielded_ion_galactic_yesMagnetosphere['O'] = unshielded_O_galactic_yesMagnetosphere                                                                       
unshielded_ion_galactic_yesMagnetosphere['Ne'] = unshielded_Ne_galactic_yesMagnetosphere                                                                   
unshielded_ion_galactic_yesMagnetosphere['Mg'] = unshielded_Mg_galactic_yesMagnetosphere
unshielded_ion_galactic_yesMagnetosphere['Si'] = unshielded_Si_galactic_yesMagnetosphere                                                                      
unshielded_ion_galactic_yesMagnetosphere['Fe'] = unshielded_Fe_galactic_yesMagnetosphere
                                                                                                                                  
    
Al_shielding_gcm2 = [0.1, 0.5, 1.0, 2.0, 5.0]
Al_shielding_gcm2_names = ['0p1', '0p5', '1p0', '2p0', '5p0']


Alshielded_protons_yesMagnetosphere = {}
Alshielded_electrons_yesMagnetosphere = {}
Alshielded_neutrons_yesMagnetosphere = {}
Alshielded_solarHe_yesMagnetosphere = {}
Alshielded_GCRH_yesMagnetosphere = {}

# Read in trapped, neutron, solar H with Al shielding from SPENVIS and 
# solar He and GCR with Al shielding from GEANT4
# all with magnetosphere shielding

filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded/WITH_MAGNETOSPHERE_SHIELDING/'
for Al_thickness in Al_shielding_gcm2_names:
    for orbit_keyvalue, filename in filenames.items():
        fn = filepath + Al_thickness + '/' + filename
        keyvalue = orbit_keyvalue + '_' + Al_thickness

        # this include trapped + solar H
        proton_spectrum = pd.read_csv(fn, sep=',', skiprows=13, skipfooter=121, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (proton_spectrum['Energy_keV_high'] - proton_spectrum['Energy_keV_low'])/1000.0
        proton_spectrum['DFlux_MeVcm2s'] = (proton_spectrum.Flux_cm2bin/E_width)/mission_time_s
        proton_spectrum['Energy_MeV'] = (0.5/1000.0)*(proton_spectrum['Energy_keV_low']+proton_spectrum['Energy_keV_high'])
        
        electron_spectrum = pd.read_csv(fn, sep=',', skiprows=73, skipfooter=62, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (electron_spectrum['Energy_keV_high'] - electron_spectrum['Energy_keV_low'])/1000.0
        electron_spectrum['DFlux_MeVcm2s'] = (electron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        electron_spectrum['Energy_MeV'] = (0.5/1000.0)*(electron_spectrum['Energy_keV_low']+electron_spectrum['Energy_keV_high'])
        Alshielded_electrons_yesMagnetosphere[keyvalue] = electron_spectrum
        
        neutron_spectrum = pd.read_csv(fn, sep=',', skiprows=133, skipfooter=1, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (neutron_spectrum['Energy_keV_high'] - neutron_spectrum['Energy_keV_low'])/1000.0
        neutron_spectrum['DFlux_MeVcm2s'] = (neutron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_neutrons_yesMagnetosphere[keyvalue] = neutron_spectrum
        
        # from GEANT4, Al shielded solar He
        fn = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded_GEANT4_SolarHe/' + orbit_keyvalue + '_SolarHeSpectrum_' + Al_thickness + '_Al.txt' 
        solarHe_spectrum = pd.read_csv(fn, sep = ' ', names=['E_MeV', 'DFlux_MeVcm2s'])
        Alshielded_solarHe_yesMagnetosphere[keyvalue] = solarHe_spectrum 

        # from GEANT4, Al shielded GCR H, add to other protons to propogate through together
        fn = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded_GEANT4_GCRH/' + orbit_keyvalue + '_GCRHSpectrum_' + Al_thickness + '_Al.txt' 
        GCRH_spectrum = pd.read_csv(fn, sep = ' ', names=['E_MeV', 'DFlux_MeVcm2s'])
        # add GCR H to all other H from trapped and solar
        proton_spectrum['DFlux_MeVcm2s'] = proton_spectrum['DFlux_MeVcm2s'] + GCRH_spectrum['DFlux_MeVcm2s']
        Alshielded_GCRH_yesMagnetosphere[keyvalue] = GCRH_spectrum   
        
        Alshielded_protons_yesMagnetosphere[keyvalue] = proton_spectrum

    
Alshielded_protons_noMagnetosphere = {}
Alshielded_electrons_noMagnetosphere = {}
Alshielded_neutrons_noMagnetosphere = {}

# Read in trapped, neutron, solar H with Al shielding from SPENVIS and 
# all without magnetosphere shielding, we don't use these

filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded/WITHOUT_MAGNETOSPHERE_SHIELDING/'
for Al_thickness in Al_shielding_gcm2_names:
    for keyvalue, filename in filenames.items():
        fn = filepath + Al_thickness + '/' + filename

        proton_spectrum = pd.read_csv(fn, sep=',', skiprows=13, skipfooter=121, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')    
        E_width = (proton_spectrum['Energy_keV_high'] - proton_spectrum['Energy_keV_low'])/1000.0
        proton_spectrum['DFlux_MeVcm2s'] = (proton_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_protons_noMagnetosphere[keyvalue] = proton_spectrum

        electron_spectrum = pd.read_csv(fn, sep=',', skiprows=73, skipfooter=62, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (electron_spectrum['Energy_keV_high'] - electron_spectrum['Energy_keV_low'])/1000.0
        electron_spectrum['DFlux_MeVcm2s'] = (electron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_electrons_noMagnetosphere[keyvalue] = electron_spectrum
        
        neutron_spectrum = pd.read_csv(fn, sep=',', skiprows=133, skipfooter=1, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (neutron_spectrum['Energy_keV_high'] - neutron_spectrum['Energy_keV_low'])/1000.0
        neutron_spectrum['DFlux_MeVcm2s'] = (neutron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_neutrons_noMagnetosphere[keyvalue] = neutron_spectrum
        
# this contains all of the proton energies to propogate through for all orbits and shieldings, MeV, average
protons_Al_shielded_bin_energies_all_yesMagnetosphere = (0.5/1000.0)*(Alshielded_protons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_low']+Alshielded_protons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_high'])

# this contains all of the electron energies to propogate through for all orbits and shieldings, MeV
electrons_Al_shielded_bin_energies_all_yesMagnetosphere = (0.5/1000.0)*(Alshielded_electrons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_low']+Alshielded_electrons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_high'])


# In[4]:


# sum up all protons for without Al shielding, with shielding were summed above by default

# this contains all of the proton energies to propogate through for all orbits and with no shielding
protons_unshielded_bin_energies_all_yesMagnetosphere = numpy.append(numpy.array(unshielded_solar_protons_yesMagnetosphere['LEO_600']['Energy_MeV']), [5.5e+02, 6.0e+02, 6.5e+02, 7.0e+02, 7.5e+02, 8.0e+02, 8.5e+02, 9.0e+02, 9.5e+02, 1.0e+03])
protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s = {}

# for each orbit, all have same energy bins
for keyvalue, filename in filenames.items():
    this_sum = numpy.zeros(len(protons_unshielded_bin_energies_all_yesMagnetosphere))
    # for each energy in the orbit
    for num, E_current in enumerate(protons_unshielded_bin_energies_all_yesMagnetosphere):
        # flux sum for this energy bin from all proton sources (trapped, solar, GCR)
        this_E_sum = 0       
        orbit_function = interp1d(unshielded_trapped_protons[keyvalue].Energy_MeV, unshielded_trapped_protons[keyvalue].DFlux_MeVcm2s, kind='quadratic', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + max(orbit_function(E_current),0.0)
        orbit_function = interp1d(unshielded_solar_protons_yesMagnetosphere[keyvalue].Energy_MeV, unshielded_solar_protons_yesMagnetosphere[keyvalue].DFlux_MeVcm2s*unshielded_solar_protons_yesMagnetosphere[keyvalue].AttenFactor, kind='quadratic', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + max(orbit_function(E_current),0.0)
        orbit_function = interp1d(unshielded_H_galactic_yesMagnetosphere[keyvalue].Energy_MeV, (1.0/10000.0)*math.pi*unshielded_H_galactic_yesMagnetosphere[keyvalue].H_DFlux_m2sSrMeV, kind='quadratic', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + max(orbit_function(E_current),0.0) 
        # write the total proton flux for this energy bin
        this_sum[num] = this_E_sum
    # write the total proton fluxes for all energy bins
    protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue] = this_sum

# put electrons in the same format, only one source (trapped so nothing to sum)
electrons_unshielded_bin_energies_all_yesMagnetosphere = numpy.array(unshielded_trapped_electrons['LEO_600']['Energy_MeV'])
electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s = {}
for keyvalue, filename in filenames.items():
    electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue] = numpy.array(unshielded_trapped_electrons[keyvalue].DFlux_MeVcm2s)



# In[5]:


# Sum up the energy deposited for each of the orbits / shieldings, include all particles


# Compute 3 times including a subset of GCR ions to see which ones matter
inc = ['noIon', 'justSolarHe', 'allIon']
for inclusion in inc:

    edep_vs_Dist_n = {}
    edep_vs_Dist_e = {}

    ### NO ALUMINUM SHIELDING
    zero_Al_thickness = {'0.0':0.0}
    # for each aluminum thickness == 0
    for Al_num, Al_thickness in enumerate(zero_Al_thickness):
        # for each material
        for mat_num, sm in enumerate(material_densities):
            # make the start of the legend, the aluminum thickness and semiconductor
            kv1 = Al_thickness + '_' + sm                
            # for each of the orbits
            for orbit_keyvalue, orbit_legvalue in orbits_legend.items():
                    # this is the total key to get the flux
                    kv2 = kv1 + '_' + orbit_keyvalue
                    print(kv2)

                    # wrote the sum for every 10 microns, 2000 microns in total (2 mm)
                    cum_edep_n = numpy.zeros(200)
                    cum_edep_e = numpy.zeros(200)

                    ##  1 ## PROTONS
                    proton_flux = protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[orbit_keyvalue]
                    proton_energy = protons_unshielded_bin_energies_all_yesMagnetosphere
                    proton_energy_widths = numpy.diff(proton_energy, prepend=0)  
                    
                    # get edep, nuclear and electronic, for this semi and this particle
                    edep_this_n = unshielded_Ebin_edep_n[unshielded_Ebin_edep_n.material == sm]
                    edep_this_n = edep_this_n[edep_this_n.particle == 'p']
                    edep_this_e = unshielded_Ebin_edep_e[unshielded_Ebin_edep_e.material == sm]
                    edep_this_e = edep_this_e[edep_this_e.particle == 'p']
                    # for edep, nuclear and electronic, for each semi/particle/flux incident energy bin
                    for num_E, eBin in enumerate(proton_energy):
                        edep_this_nn = edep_this_n[edep_this_n.E_bin_MeV == eBin]
                        edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])
                        edep_this_ee = edep_this_e[edep_this_e.E_bin_MeV == eBin]
                        edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])
                        
                        # add the flux weighted energy deposition as a function of distance
                        cum_edep_n = cum_edep_n + edep_this_nn*proton_flux[num_E]*proton_energy_widths[num_E]
                        cum_edep_e = cum_edep_e + edep_this_ee*proton_flux[num_E]*proton_energy_widths[num_E]

                    ## 2 ## ELECTRONS 
                    electron_flux = electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[orbit_keyvalue]
                    electron_energy = electrons_unshielded_bin_energies_all_yesMagnetosphere
                    electron_energy_widths = numpy.diff(electron_energy, prepend=0)  
                    
                    # get edep, nuclear and electronic, for this semi and this particle
                    edep_this_n = unshielded_Ebin_edep_n[unshielded_Ebin_edep_n.material == sm]
                    edep_this_n = edep_this_n[edep_this_n.particle == 'e']
                    edep_this_e = unshielded_Ebin_edep_e[unshielded_Ebin_edep_e.material == sm]
                    edep_this_e = edep_this_e[edep_this_e.particle == 'e']
                    # for edep, nuclear and electronic, for each semi/particle/flux incident energy bin
                    for num_E, eBin in enumerate(electron_energy):
                        edep_this_nn = edep_this_n[edep_this_n.E_bin_MeV == eBin]
                        edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])
                        edep_this_ee = edep_this_e[edep_this_e.E_bin_MeV == eBin]
                        edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])

                        # add the flux weighted energy deposition as a function of distance
                        cum_edep_n = cum_edep_n + edep_this_nn*electron_flux[num_E]*electron_energy_widths[num_E]
                        cum_edep_e = cum_edep_e + edep_this_ee*electron_flux[num_E]*electron_energy_widths[num_E]
                     
                    # if including solar He or all ions
                    if (inclusion == 'justSolarHe') or (inclusion == 'allIon'):

                        ## 3 ## SOLAR HE
                        solarHe_flux = unshielded_solar_He_yesMagnetosphere[orbit_keyvalue].DFlux_MeVcm2s
                        solarHe_energy = unshielded_solar_He_yesMagnetosphere[orbit_keyvalue].Energy_MeV
                        solarHe_energy_widths = numpy.diff(solarHe_energy, prepend=0)  
                        # get just the solar He (not GCR)
                        edep_this_n = unshielded_Ebin_edep_n[(unshielded_Ebin_edep_n.material == sm) & (unshielded_Ebin_edep_n.index >= solarHe_startIndex)]
                        edep_this_n = edep_this_n[edep_this_n.particle == 'He']
                        edep_this_e = unshielded_Ebin_edep_e[(unshielded_Ebin_edep_e.material == sm) & (unshielded_Ebin_edep_n.index >= solarHe_startIndex)]
                        edep_this_e = edep_this_e[edep_this_e.particle == 'He']
                        for num_E, eBin in enumerate(solarHe_energy):
                            edep_this_nn = edep_this_n[edep_this_n.E_bin_MeV == eBin]
                            edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])
                            edep_this_ee = edep_this_e[edep_this_e.E_bin_MeV == eBin]
                            edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])
        
                            # add the flux weighted energy deposition as a function of distance
                            cum_edep_n = cum_edep_n + edep_this_nn*solarHe_flux[num_E]*solarHe_energy_widths[num_E]
                            cum_edep_e = cum_edep_e + edep_this_ee*solarHe_flux[num_E]*solarHe_energy_widths[num_E]

                        # if including all ions
                        if (inclusion == 'allIon'):
                        
                            ## ALL GCR IONS
                            # convert to per cm2 s MeV
                            gcrHe_spectrum = unshielded_He_galactic_yesMagnetosphere[orbit_keyvalue]['He_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrHe_energies = unshielded_He_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrC_spectrum = unshielded_C_galactic_yesMagnetosphere[orbit_keyvalue]['C_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrC_energies = unshielded_C_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrO_spectrum = unshielded_O_galactic_yesMagnetosphere[orbit_keyvalue]['O_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrO_energies = unshielded_O_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrN_spectrum = unshielded_N_galactic_yesMagnetosphere[orbit_keyvalue]['N_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrN_energies = unshielded_N_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrNe_spectrum = unshielded_Ne_galactic_yesMagnetosphere[orbit_keyvalue]['Ne_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrNe_energies = unshielded_Ne_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrMg_spectrum = unshielded_Mg_galactic_yesMagnetosphere[orbit_keyvalue]['Mg_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrMg_energies = unshielded_Mg_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrSi_spectrum = unshielded_Si_galactic_yesMagnetosphere[orbit_keyvalue]['Si_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrSi_energies = unshielded_Si_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']
                            gcrFe_spectrum = unshielded_Fe_galactic_yesMagnetosphere[orbit_keyvalue]['Fe_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi
                            gcrFe_energies = unshielded_Fe_galactic_yesMagnetosphere[orbit_keyvalue]['Energy_MeV']

                            # put together so that you can loop through
                            extra_ion_spectra = [gcrHe_spectrum, gcrC_spectrum, gcrO_spectrum, gcrN_spectrum, gcrNe_spectrum, gcrMg_spectrum, gcrSi_spectrum, gcrFe_spectrum]
                            extra_ion_energies = [gcrHe_energies, gcrC_energies, gcrO_energies, gcrN_energies, gcrNe_energies, gcrMg_energies, gcrSi_energies, gcrFe_energies]
                            extra_ions = ['He', 'C', 'O', 'N', 'Ne', 'Mg', 'Si', 'Fe']
                            # for each GCR ion
                            for ion_num, ion in enumerate(extra_ions):
                                # incident energies from this orbit - ion
                                ion_energy = extra_ion_energies[ion_num] 
                                ion_flux = extra_ion_spectra[ion_num]       
                                ion_energy_widths = numpy.diff(ion_energy, prepend=0)  
                                # because solar He is the last thing written, they must have smaller indices to distinguish GCR He from solar He
                                edep_this_n = unshielded_Ebin_edep_n[(unshielded_Ebin_edep_n.material == sm) & (unshielded_Ebin_edep_n.index <= solarHe_startIndex) & (unshielded_Ebin_edep_n.particle  == ion)]
                                edep_this_e = unshielded_Ebin_edep_e[(unshielded_Ebin_edep_e.material == sm) & (unshielded_Ebin_edep_e.index <= solarHe_startIndex) & (unshielded_Ebin_edep_e.particle  == ion)]

                                for num_E, eBin in enumerate(ion_energy):
                                    # because of slight rounding errors in what was written
                                    eBin = round(eBin,2)
                                    edep_this_nn = edep_this_n[numpy.round(edep_this_n.E_bin_MeV,2) == eBin]
                                    edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])
                                    edep_this_ee = edep_this_e[numpy.round(edep_this_e.E_bin_MeV,2) == eBin]
                                    edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])

                                    # add the flux weighted energy deposition as a function of distance
                                    cum_edep_n = cum_edep_n + edep_this_nn*ion_flux[num_E]*ion_energy_widths[num_E]
                                    cum_edep_e = cum_edep_e + edep_this_ee*ion_flux[num_E]*ion_energy_widths[num_E]


                    # append edep as a function of distance to the dictionary by key
                    # works with distinguished inclusion because we just summed the applicable
                    # ones to this for this loop so distinguish by which loop you are on
                    # key includes - Al shielding, material, orbit
                    edep_vs_Dist_n[kv2] = cum_edep_n
                    edep_vs_Dist_e[kv2] = cum_edep_e

    ### YES ALUMINUM SHIELDING
    Al_shielding_gcm2_dict = {'0.0':0.0, '0p1':0.1, '0p5':0.5, '1p0':1.0, '2p0':2.0, '5p0':5.0}
    for Al_num, Al_thickness in enumerate(Al_shielding_gcm2_names):
        # for each material
        for mat_num, sm in enumerate(material_densities):
            # make the start of the legend, the aluminum thickness and semiconductor
            kv1 = Al_thickness + '_' + sm                
            # for each of the orbits
            for orbit_keyvalue, orbit_legvalue in orbits_legend.items():

                    kv2 = kv1 + '_' + orbit_keyvalue
                    print(kv2)

                    # wrote the sum for every 10 microns, 2 mm total
                    cum_edep_n = numpy.zeros(200)
                    cum_edep_e = numpy.zeros(200)
                    ## 1 ## PROTONS - incudes all (trapped, solar, GCR)
                    proton_flux = Alshielded_protons_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].DFlux_MeVcm2s
                    # this is the average for the bin
                    proton_energy = Alshielded_protons_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].Energy_MeV
                    # this is upper energy to get accurate bin widths
                    proton_energy_upper = (0.001)*Alshielded_protons_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].Energy_keV_high
                    proton_energy_widths = numpy.diff(proton_energy_upper, prepend=0)   
                    # get energy deposition as a function of distance for semi and particle
                    edep_this_n = shielded_Ebin_edep_n[shielded_Ebin_edep_n.material == sm]
                    edep_this_n = edep_this_n[edep_this_n.particle == 'p']
                    edep_this_e = shielded_Ebin_edep_e[shielded_Ebin_edep_e.material == sm]
                    edep_this_e = edep_this_e[edep_this_e.particle == 'p']
                    # and for incident energy of the flux bin
                    for num_E, eBin in enumerate(proton_energy):
                        edep_this_nn = edep_this_n[edep_this_n.E_bin_MeV == eBin]
                        edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])
                        edep_this_ee = edep_this_e[edep_this_e.E_bin_MeV == eBin]
                        edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])
                        # add edep as a function of distance to the total weighted by the flux of that incident energy
                        cum_edep_n = cum_edep_n + edep_this_nn*proton_flux[num_E]*proton_energy_widths[num_E]
                        cum_edep_e = cum_edep_e + edep_this_ee*proton_flux[num_E]*proton_energy_widths[num_E]
                    ## 2 ## ELECTRONS - trapped
                    electron_flux = Alshielded_electrons_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].DFlux_MeVcm2s
                    # average energy of the flux bin
                    electron_energy = Alshielded_electrons_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].Energy_MeV
                    # max energy of the flux bin (so acccurate widths)
                    electron_energy_upper = (0.001)*Alshielded_electrons_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].Energy_keV_high               
                    electron_energy_widths = numpy.diff(electron_energy_upper, prepend=0) 
                    # get energy deposition as a function of distance for semi and particle
                    edep_this_n = shielded_Ebin_edep_n[shielded_Ebin_edep_n.material == sm]
                    edep_this_n = edep_this_n[edep_this_n.particle == 'e']
                    edep_this_e = shielded_Ebin_edep_e[shielded_Ebin_edep_e.material == sm]
                    edep_this_e = edep_this_e[edep_this_e.particle == 'e']
                    # and for incident energy of the flux bin
                    for num_E, eBin in enumerate(electron_energy):
                        edep_this_nn = edep_this_n[edep_this_n.E_bin_MeV == eBin]
                        edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])
                        edep_this_ee = edep_this_e[edep_this_e.E_bin_MeV == eBin]
                        edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])
                        # add edep as a function of distance to the total weighted by the flux of that incident energy
                        cum_edep_n = cum_edep_n + edep_this_nn*electron_flux[num_E]*electron_energy_widths[num_E]
                        cum_edep_e = cum_edep_e + edep_this_ee*electron_flux[num_E]*electron_energy_widths[num_E]
                    ## 3 ## HE - only solar matters from above comp of trapped
                    solarHe_flux = Alshielded_solarHe_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].DFlux_MeVcm2s
                    solarHe_energy = Alshielded_solarHe_yesMagnetosphere[orbit_keyvalue+'_'+Al_thickness].E_MeV
                    solarHe_energy_widths = numpy.diff(numpy.concatenate((solarHe_energy,[0])))
                    # get energy deposition as a function of distance for semi and particle
                    edep_this_n = shielded_Ebin_edep_n[shielded_Ebin_edep_n.material == sm]
                    edep_this_n = edep_this_n[edep_this_n.particle == 'He']
                    edep_this_e = shielded_Ebin_edep_e[shielded_Ebin_edep_e.material == sm]
                    edep_this_e = edep_this_e[edep_this_e.particle == 'He']
                    # and for incident energy of the flux bin
                    for num_E, eBin in enumerate(solarHe_energy):
                        edep_this_nn = edep_this_n[edep_this_n.E_bin_MeV == eBin]
                        edep_this_nn = numpy.array(edep_this_nn['E_dep_MeV'])[0:200]
                        edep_this_ee = edep_this_e[edep_this_e.E_bin_MeV == eBin]
                        edep_this_ee = numpy.array(edep_this_ee['E_dep_MeV'])[0:200]
                        # add edep as a function of distance to the total weighted by the flux of that incident energy
                        cum_edep_n = cum_edep_n + edep_this_nn*solarHe_flux[num_E]*solarHe_energy_widths[num_E]
                        cum_edep_e = cum_edep_e + edep_this_ee*solarHe_flux[num_E]*solarHe_energy_widths[num_E]

                    edep_vs_Dist_n[kv2] = cum_edep_n
                    edep_vs_Dist_e[kv2] = cum_edep_e

    # write which ions are included so that we can later make the comparison
    if inclusion == 'noIon':
        orbit_Edep_percm2Second_all_shield_nuclear_no_ion = edep_vs_Dist_n
        orbit_Edep_percm2Second_all_shield_electronic_no_ion = edep_vs_Dist_e
    if inclusion == 'justSolarHe':
        orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe = edep_vs_Dist_n
        orbit_Edep_percm2Second_all_shield_electronic_just_solarHe = edep_vs_Dist_e
    if inclusion == 'allIon':
        orbit_Edep_percm2Second_all_shield_nuclear_all_ion = edep_vs_Dist_n
        orbit_Edep_percm2Second_all_shield_electronic_all_ion = edep_vs_Dist_e



# In[7]:


# Calculate the normal equivalence (non-iso)
# real distance in microns, recall wrote every 10 microns before
dist = numpy.arange(0,2000,10)
# nice spacing of distance values to record, don't want to make it too big
max_D_um_vals = numpy.concatenate((numpy.arange(10,101,10), numpy.arange(120,250,20), numpy.arange(250,1000,50), numpy.arange(1000,2001,100)))


# MeV
test_Energies = [ 10., 15., 25., 35., 45., 63.,  70., 100., 200., 230.]
# number/cm2
test_Fluence = 7.5*10**11

max_D_um = numpy.concatenate((numpy.arange(10,101,10), numpy.arange(120,250,20), numpy.arange(250,1000,50), numpy.arange(1000,2000,100)))

# for each level of ion inclusion
for inclusion in inc:    
    if inclusion == 'noIon':
        orbit_Edep_percm2Second_all_shield_nuclear_this = orbit_Edep_percm2Second_all_shield_nuclear_no_ion
        orbit_Edep_percm2Second_all_shield_electronic_this = orbit_Edep_percm2Second_all_shield_electronic_no_ion
    if inclusion == 'justSolarHe':
        orbit_Edep_percm2Second_all_shield_nuclear_this = orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe
        orbit_Edep_percm2Second_all_shield_electronic_this = orbit_Edep_percm2Second_all_shield_electronic_just_solarHe
    if inclusion == 'allIon':
        orbit_Edep_percm2Second_all_shield_nuclear_this = orbit_Edep_percm2Second_all_shield_nuclear_all_ion
        orbit_Edep_percm2Second_all_shield_electronic_this = orbit_Edep_percm2Second_all_shield_electronic_all_ion

    
    
    Al_thicks = []
    mats = []
    orbits = []
    testEs = []
    times_s_n = []
    times_s_e = []
    max_distances = []
    # for all shieldings
    for num, Al_thickness in enumerate(Al_shielding_gcm2_dict):
        # for each material
        for num, sm in enumerate(material_densities):
            # make the start of the legend, the aluminum thickness and semiconductor
            kv1 = Al_thickness + '_' + sm                
            for orbit_keyvalue, orbit_legvalue in orbits_legend.items():
                # Key = AlThickness_Orbit_Material
                kv2 = kv1 + '_' + orbit_keyvalue
                print(kv2)
                # preset material thicknesses to consider in micron
                for max_D_um in max_D_um_vals:
                    # arg not equal to um because only wrote every 10
                    max_arg = numpy.argmin(abs(dist - max_D_um)) + 1
                    # sum the flux weighted energy deposition for the thickness of interest
                    eDepOrbit_cm2s_n = sum(orbit_Edep_percm2Second_all_shield_nuclear_this[kv2][0:max_arg])
                    eDepOrbit_cm2s_e = sum(orbit_Edep_percm2Second_all_shield_electronic_this[kv2][0:max_arg])
                    # for each test energy
                    for testE_num, testE in enumerate(test_Energies):
                        # get edep for this test energy and material
                        this_test_edep_n = test_Ebin_edep_n[test_Ebin_edep_n.E_bin_MeV == testE]
                        this_test_edep_n = this_test_edep_n[this_test_edep_n.material == sm]
                        # sum the fluence weighted energy deposition for the thickness of interest
                        eDepTest_cm2_n = sum(test_Fluence*this_test_edep_n['E_dep_MeV'][0:max_arg])
                        # get edep for this test energy and material
                        this_test_edep_e = test_Ebin_edep_e[test_Ebin_edep_e.E_bin_MeV == testE]
                        this_test_edep_e = this_test_edep_e[this_test_edep_e.material == sm]
                        # sum the fluence weighted energy deposition for the thickness of interest
                        eDepTest_cm2_e = sum(test_Fluence*this_test_edep_e['E_dep_MeV'][0:max_arg])
                        # divide fluence by flux weighted edep to get time
                        time_s_n = eDepTest_cm2_n/eDepOrbit_cm2s_n
                        time_s_e = eDepTest_cm2_e/eDepOrbit_cm2s_e
                        # write the indices
                        Al_thicks.append(Al_shielding_gcm2_dict[Al_thickness])
                        mats.append(sm)
                        orbits.append(orbit_legvalue)
                        testEs.append(testE)
                        times_s_n.append(time_s_n)
                        times_s_e.append(time_s_e)
                        max_distances.append(max_D_um)


                        
    if inclusion == 'noIon':
        All_equivalence_data_normal_no_ion = {'Sample_thickness_um':numpy.array(max_distances), 'Al_Shielding_Thickness_gcm2':numpy.array(Al_thicks), 'Material':numpy.array(mats), 'Orbit':numpy.array(orbits), 'Test_Energy_MeV':numpy.array(testEs), 'Electronic_Equivalence_s': numpy.array(times_s_e), 'Nuclear_Equivalence_s':numpy.array(times_s_n) }
        All_equivalence_normal_no_ion = pd.DataFrame(All_equivalence_data_normal_no_ion) 
    
    if inclusion == 'justSolarHe':
        All_equivalence_data_normal_just_solarHe = {'Sample_thickness_um':numpy.array(max_distances), 'Al_Shielding_Thickness_gcm2':numpy.array(Al_thicks), 'Material':numpy.array(mats), 'Orbit':numpy.array(orbits), 'Test_Energy_MeV':numpy.array(testEs), 'Electronic_Equivalence_s': numpy.array(times_s_e), 'Nuclear_Equivalence_s':numpy.array(times_s_n) }
        All_equivalence_normal_just_solarHe = pd.DataFrame(All_equivalence_data_normal_just_solarHe) 

    if inclusion == 'allIon':
        All_equivalence_data_normal_all_ion = {'Sample_thickness_um':numpy.array(max_distances), 'Al_Shielding_Thickness_gcm2':numpy.array(Al_thicks), 'Material':numpy.array(mats), 'Orbit':numpy.array(orbits), 'Test_Energy_MeV':numpy.array(testEs), 'Electronic_Equivalence_s': numpy.array(times_s_e), 'Nuclear_Equivalence_s':numpy.array(times_s_n) }
        All_equivalence_normal_all_ion = pd.DataFrame(All_equivalence_data_normal_all_ion) 

    #All_equivalence_normal.to_csv('All_equivalence_7p5E11_F_normal.csv')


# In[8]:


# Plot the ion inclusion sensitivity equivalent durations

st_um = 500
mat = 'Si'
E_MeV = 63.0

# get the equivalences for the different ion inclusions
data = All_equivalence_normal_no_ion
no_ion_E = data.Electronic_Equivalence_s
no_ion_N = data.Nuclear_Equivalence_s
data = All_equivalence_normal_just_solarHe
just_solarHe_E = data.Electronic_Equivalence_s
just_solarHe_N = data.Nuclear_Equivalence_s
data = All_equivalence_normal_all_ion
all_ion_E = data.Electronic_Equivalence_s
all_ion_N = data.Nuclear_Equivalence_s

# fractional error - duration longer if fewer ions included, which is why we need this order to be +
err_no_ion_E = (-all_ion_E+no_ion_E)/no_ion_E
err_no_ion_N = (-all_ion_N+no_ion_N)/no_ion_N
# fractional error - duration longer if fewer ions included, which is why we need this order to be +
err_just_solarHe_E = (-all_ion_E+just_solarHe_E)/just_solarHe_E
err_just_solarHe_N = (-all_ion_N+just_solarHe_N)/just_solarHe_N

# record errors
All_equivalence_normal_all_ion['err_no_ion_E'] = err_no_ion_E
All_equivalence_normal_all_ion['err_no_ion_N'] = err_no_ion_N

All_equivalence_normal_all_ion['err_just_solarHe_E'] = err_just_solarHe_E
All_equivalence_normal_all_ion['err_just_solarHe_N'] = err_just_solarHe_N

# thickness, just showing one
t = 500

# plot errors in nuclear equivalent duration, recall only calculated in no Al shielding case
fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))
legend_strings = []
ax = fig.add_subplot(2,2,1)
seaborn.stripplot(x='Test_Energy_MeV', y='err_no_ion_N', hue='Orbit', data=All_equivalence_normal_all_ion[(All_equivalence_normal_all_ion.Al_Shielding_Thickness_gcm2 == 0.0) & (All_equivalence_normal_all_ion.Sample_thickness_um == t)])
matplotlib.pyplot.ylim([-0.01,0.35])
#matplotlib.pyplot.xlabel('Test Energy [MeV]')
matplotlib.pyplot.ylabel(r'[dE/dx]$_n$ Fractional Difference in Equivalence')
matplotlib.pyplot.title('(No Z>1 Ions) vs (All Major Ions)')
ax.get_legend().remove()
matplotlib.pyplot.xlabel('')
matplotlib.pyplot.tick_params(axis='x', labelleft=False)
matplotlib.pyplot.grid(b=None, which='major', axis='y', color='grey', linestyle='-', linewidth=.2)
ax = fig.add_subplot(2,2,2)
seaborn.stripplot(x='Test_Energy_MeV', y='err_just_solarHe_N', hue='Orbit', data=All_equivalence_normal_all_ion[(All_equivalence_normal_all_ion.Al_Shielding_Thickness_gcm2 == 0.0) & (All_equivalence_normal_all_ion.Sample_thickness_um == t)])
matplotlib.pyplot.title('(Only Solar He Ions) vs (All Major Ions)')
matplotlib.pyplot.ylim([-0.01,0.35])
matplotlib.pyplot.ylabel('')
matplotlib.pyplot.xlabel('')
matplotlib.pyplot.tick_params(axis='y', labelleft=False)
matplotlib.pyplot.tick_params(axis='x', labelleft=False)
matplotlib.pyplot.grid(b=None, which='major', axis='y', color='grey', linestyle='-', linewidth=.2)

# plot errors in electronic equivalent duration, recall only calculated in no Al shielding case
ax = fig.add_subplot(2,2,3)
seaborn.stripplot(x='Test_Energy_MeV', y='err_no_ion_E', hue='Orbit', data=All_equivalence_normal_all_ion[(All_equivalence_normal_all_ion.Al_Shielding_Thickness_gcm2 == 0.0) & (All_equivalence_normal_all_ion.Sample_thickness_um == t)])
matplotlib.pyplot.ylim([-0.01,0.1])
matplotlib.pyplot.xlabel('Test Energy [MeV]')
matplotlib.pyplot.ylabel(r'[dE/dx]$_e$ Fractional Difference in Equivalence')
matplotlib.pyplot.title('(No Z>1 Ions) vs (All Major Ions)')
matplotlib.pyplot.grid(b=None, which='major', axis='y', color='grey', linestyle='-', linewidth=.2)
ax.get_legend().remove()
ax = fig.add_subplot(2,2,4)
seaborn.stripplot(x='Test_Energy_MeV', y='err_just_solarHe_E', hue='Orbit', data=All_equivalence_normal_all_ion[(All_equivalence_normal_all_ion.Al_Shielding_Thickness_gcm2 == 0.0) & (All_equivalence_normal_all_ion.Sample_thickness_um == t)])
matplotlib.pyplot.title('(Only Solar He Ions) vs (All Major Ions)')
matplotlib.pyplot.ylim([-0.01,0.1])
matplotlib.pyplot.ylabel('')
matplotlib.pyplot.tick_params(axis='y', labelleft=False)
matplotlib.pyplot.grid(b=None, which='major', axis='y', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.show()


# In[9]:


# Calculate the isotropic conversion


# to store isotropic results in
orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe_iso = {}
orbit_Edep_percm2Second_all_shield_electronic_just_solarHe_iso = {}

dist = numpy.arange(0,200,1)
# 0 to 1 to sample from the CDF
probs = numpy.random.uniform(0, 1, 100000)
# drawn thetas from the PDF associated with the CFF
thetas = numpy.arcsin(probs**(1/2))

# for each Al thickness
for num, Al_thickness in enumerate(Al_shielding_gcm2_dict):
    # for each material
    for num, sm in enumerate(material_densities):
        # make the start of the legend, the aluminum thickness and semiconductor
        kv1 = Al_thickness + '_' + sm                
        for orbit_keyvalue, orbit_legvalue in orbits_legend.items():
            # Key = AlThickness_Orbit_Material
            kv2 = kv1 + '_' + orbit_keyvalue
            
            # this is the normal incidence energy deposition - nuclear and electronic
            # previously determined that just solar He was enough
            this_n = orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe[kv2]
            this_e = orbit_Edep_percm2Second_all_shield_electronic_just_solarHe[kv2]
            
            # to store the iso
            this_n_iso = numpy.zeros(200)
            this_e_iso = numpy.zeros(200)
            

            # these are the indices of the edep when projected normal to the surface so add to the normal distance
            # edep is the energy deposited along the straight line of the incident particle in some direction
            # for each isotropic angle of origin
            for theta in thetas:
                dist_proj = numpy.floor(dist*numpy.cos(theta)).astype('int')    
                # add projection
                numpy.add.at(this_n_iso, dist_proj, this_n)
                numpy.add.at(this_e_iso, dist_proj, this_e)
                
            # append results, convert back to initial energy by dividing by the number of sums taken
            orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe_iso[kv2] = this_n_iso/len(thetas)
            orbit_Edep_percm2Second_all_shield_electronic_just_solarHe_iso[kv2] = this_e_iso/len(thetas)
            
     


# In[10]:


# Calculate the isotropic equivalence - for special fluence
dist = numpy.arange(0,2000,10)
# calculate the equivalence for this subset of material thicknesses
max_D_um_vals = numpy.concatenate((numpy.arange(10,101,10), numpy.arange(120,250,20), numpy.arange(250,1000,50), numpy.arange(1000,2001,100)))


# MeV
test_Energies = [ 10., 15., 25., 35., 45., 63.,  70., 100., 200., 230.]
# number/cm2
test_Fluence = 7.5*10**11

max_D_um = numpy.concatenate((numpy.arange(10,101,10), numpy.arange(120,250,20), numpy.arange(250,1000,50), numpy.arange(1000,2000,100)))

Al_thicks = []
mats = []
orbits = []
testEs = []
times_s_n = []
times_s_e = []
max_distances = []

# for each Al thickness
for num, Al_thickness in enumerate(Al_shielding_gcm2_dict):
    # for each material
    for num, sm in enumerate(material_densities):
        # make the start of the legend, the aluminum thickness and semiconductor
        kv1 = Al_thickness + '_' + sm                
        for orbit_keyvalue, orbit_legvalue in orbits_legend.items():
            # Key = AlThickness_Orbit_Material
            kv2 = kv1 + '_' + orbit_keyvalue
            print(kv2)
            # for each relevant thickness        
            for max_D_um in max_D_um_vals:
                # get index of that thicknesss because written every 10 microns
                max_arg = numpy.argmin(abs(dist - max_D_um)) + 1
                # get the energy deposition as a function of distance for this shielding, orbit, material, sum for thickness of interest
                eDepOrbit_cm2s_n = sum(orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe_iso[kv2][0:max_arg])
                eDepOrbit_cm2s_e = sum(orbit_Edep_percm2Second_all_shield_electronic_just_solarHe_iso[kv2][0:max_arg])
                
                # for each test energy
                for testE_num, testE in enumerate(test_Energies):
                    # get edep as a function of distance for this test energy and material and sum the desired distance
                    this_test_edep_n = test_Ebin_edep_n[test_Ebin_edep_n.E_bin_MeV == testE]
                    this_test_edep_n = this_test_edep_n[this_test_edep_n.material == sm]
                    eDepTest_cm2_n = sum(test_Fluence*this_test_edep_n['E_dep_MeV'][0:max_arg])

                    this_test_edep_e = test_Ebin_edep_e[test_Ebin_edep_e.E_bin_MeV == testE]
                    this_test_edep_e = this_test_edep_e[this_test_edep_e.material == sm]
                    eDepTest_cm2_e = sum(test_Fluence*this_test_edep_e['E_dep_MeV'][0:max_arg])
                    # divide fluence weigted by flux weighted to get time
                    time_s_n = eDepTest_cm2_n/eDepOrbit_cm2s_n
                    time_s_e = eDepTest_cm2_e/eDepOrbit_cm2s_e
                    # store keys
                    Al_thicks.append(Al_shielding_gcm2_dict[Al_thickness])
                    mats.append(sm)
                    orbits.append(orbit_legvalue)
                    testEs.append(testE)
                    times_s_n.append(time_s_n)
                    times_s_e.append(time_s_e)
                    max_distances.append(max_D_um)

                
# write - recall this is for 7.5E11 fluence, isotropic            
All_equivalence_data_iso = {'Sample_thickness_um':numpy.array(max_distances), 'Al_Shielding_Thickness_gcm2':numpy.array(Al_thicks), 'Material':numpy.array(mats), 'Orbit':numpy.array(orbits), 'Test_Energy_MeV':numpy.array(testEs), 'Electronic_Equivalence_s': numpy.array(times_s_e), 'Nuclear_Equivalence_s':numpy.array(times_s_n) }
All_equivalence_iso = pd.DataFrame(All_equivalence_data_iso)             
All_equivalence_iso.to_csv('All_equivalence_7p5E11_F_iso.csv')


# In[13]:


# Calculate isotropic equivalence per 10**11 p/cm2 fluence (slope)
# number/cm2

# diff = 10^11
fluences = [ 1*10**11, 2*10**11, 3*10**11]

# real distance in microns, recall wrote every 10 microns before
dist = numpy.arange(0,2000,10)
# nice spacing of distance values to record, don't want to make it too big
max_D_um_vals = numpy.concatenate((numpy.arange(10,101,10), numpy.arange(120,250,20), numpy.arange(250,1000,50), numpy.arange(1000,2001,100)))

# MeV
test_Energies = [ 10., 15., 25., 35., 45., 63.,  70., 100., 200., 230.]


Al_thicks = []
mats = []
orbits = []
testEs = []
max_distances = []
slope_n = []
slope_e = []


for num, Al_thickness in enumerate(Al_shielding_gcm2_dict):
    # for each material
    for num, sm in enumerate(material_densities):
        # make the start of the legend, the aluminum thickness and semiconductor
        kv1 = Al_thickness + '_' + sm                
        for orbit_keyvalue, orbit_legvalue in orbits_legend.items():
            # Key = AlThickness_Orbit_Material
            kv2 = kv1 + '_' + orbit_keyvalue
            # for subset of material thickesses      
            for max_D_um in max_D_um_vals:
                # get arg for that distance in um
                max_arg = numpy.argmin(abs(dist - max_D_um)) + 1
                # get isotropic orbit edep, sum for range of interest
                eDepOrbit_cm2s_n = sum(orbit_Edep_percm2Second_all_shield_nuclear_just_solarHe_iso[kv2][0:max_arg])
                eDepOrbit_cm2s_e = sum(orbit_Edep_percm2Second_all_shield_electronic_just_solarHe_iso[kv2][0:max_arg])
                # for each test E
                for testE_num, testE in enumerate(test_Energies):
                    # test energy deposition for proper energy, material
                    this_test_edep_n = test_Ebin_edep_n[test_Ebin_edep_n.E_bin_MeV == testE]
                    this_test_edep_n = this_test_edep_n[this_test_edep_n.material == sm]
                    
                    this_test_edep_e = test_Ebin_edep_e[test_Ebin_edep_e.E_bin_MeV == testE]
                    this_test_edep_e = this_test_edep_e[this_test_edep_e.material == sm]
                    
                    # sum for desired distance, weight by fluence
                    test_fluence = 1*10**11
                    eDepTest_cm2_n = sum(test_fluence*this_test_edep_n['E_dep_MeV'][0:max_arg])
                    eDepTest_cm2_e =  sum(test_fluence*this_test_edep_e['E_dep_MeV'][0:max_arg])

                    time_s_n1 = eDepTest_cm2_n/eDepOrbit_cm2s_n
                    time_s_e1 = eDepTest_cm2_e/eDepOrbit_cm2s_e
                    # sum for desired distance, weight by fluence
                    test_fluence = 2*10**11
                    eDepTest_cm2_n = sum(test_fluence*this_test_edep_n['E_dep_MeV'][0:max_arg])
                    eDepTest_cm2_e = sum(test_fluence*this_test_edep_e['E_dep_MeV'][0:max_arg])

                    time_s_n2 = eDepTest_cm2_n/eDepOrbit_cm2s_n
                    time_s_e2 = eDepTest_cm2_e/eDepOrbit_cm2s_e
                    
                    # write keys
                    Al_thicks.append(Al_shielding_gcm2_dict[Al_thickness])
                    mats.append(sm)
                    orbits.append(orbit_legvalue)
                    testEs.append(testE)
                    max_distances.append(max_D_um)
                    
                    # years per 10**11 p/cm2, note could have just done 0 to E11, but thought that I would check the two
                    slope_n.append((time_s_n2-time_s_n1)/(60*60*24*365))
                    slope_e.append((time_s_e2-time_s_e1)/(60*60*24*365))
                    
                
# store results           
All_equivalence_data_iso_fluence_slope = {'Sample_thickness_um':numpy.array(max_distances), 'Al_Shielding_Thickness_gcm2':numpy.array(Al_thicks), 'Material':numpy.array(mats), 'Orbit':numpy.array(orbits), 'Test_Energy_MeV':numpy.array(testEs), 'Slope_Year_1E11p_cm2_n': numpy.array(slope_n), 'Slope_Year_1E11p_cm2_e': numpy.array(slope_e)}
All_equivalence_iso_fluence_slope = pd.DataFrame(All_equivalence_data_iso_fluence_slope)             
# write results
All_equivalence_iso_fluence_slope.to_csv('All_equivalence_Y_PerE11_TestFluence_iso_slope.csv')


# THE REST IS JUST PLOTTING OF RESULTS

# In[14]:


orbits_labels = ['LEO Polar \nSun-sync 600km', 'LEO Polar \nSun-sync 800km', 'LEO Inclined \nNonpolar ISS', 'MEO Molniya', 'MEO Semi \nSync GPS', 'HEO Highly \nEccentric IBEX', 'HEO \nGeostationary']
#seaborn.set(font_scale=1.2)
seaborn.set_style("whitegrid")
s = 18
import matplotlib.pylab as pylab
params = {'legend.fontsize': s-1,
          'figure.figsize': (15, 5),
         'axes.labelsize': s,
         'axes.titlesize':  s,
         'xtick.labelsize': s-2,
         'ytick.labelsize': s-2}
pylab.rcParams.update(params)


# Al shielding, spread due to: material thickness, test energy, material
# E
matplotlib.pyplot.figure()
df_subset = All_equivalence_iso_fluence_slope.rename(columns={"Al_Shielding_Thickness_gcm2": "Al_shielding"})
g = seaborn.factorplot(x="Orbit", y='Slope_Year_1E11p_cm2_e', hue="Al_shielding", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g._legend.set_title("Al Shielding \n[g/cm$^2$]")
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()
# N
matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y='Slope_Year_1E11p_cm2_n', hue="Al_shielding", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g._legend.set_title("Al Shielding \n[g/cm$^2$]")
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Nuclear")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()

# Material thickness, spread due to: Al shielding, test energy, material
# E
df_subset = All_equivalence_iso_fluence_slope[(All_equivalence_iso_fluence_slope.Sample_thickness_um == 10) | (All_equivalence_iso_fluence_slope.Sample_thickness_um == 100) | (All_equivalence_iso_fluence_slope.Sample_thickness_um == 500) | (All_equivalence_iso_fluence_slope.Sample_thickness_um == 1000) | (All_equivalence_iso_fluence_slope.Sample_thickness_um == 2000)]
df_subset = df_subset.rename(columns={"Sample_thickness_um": "Sample_thick"})

matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y='Slope_Year_1E11p_cm2_e', hue="Sample_thick", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g._legend.set_title('Sample \nThickness [$\\mu$m]')
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()
# N
matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y='Slope_Year_1E11p_cm2_n', hue="Sample_thick", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g._legend.set_title('Sample \nThickness [$\\mu$m]')
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Nuclear")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()



# Test energy, spread due to: Al shielding, material thickness, material
# E
df_subset = All_equivalence_iso_fluence_slope[(All_equivalence_iso_fluence_slope.Test_Energy_MeV == 15) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 45) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 63) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 100) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 200)]

matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y='Slope_Year_1E11p_cm2_e', hue="Test_Energy_MeV", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g._legend.set_title('Test \nEnergy [MeV]')
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()
# N
matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y='Slope_Year_1E11p_cm2_n', hue="Test_Energy_MeV", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g._legend.set_title('Test \nEnergy [MeV]')
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Nuclear")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()


# In[15]:


# Test energy, spread due to: Al shielding, material thickness, material

seaborn.set_style("whitegrid")
df_subset = All_equivalence_iso_fluence_slope[(All_equivalence_iso_fluence_slope.Test_Energy_MeV == 45) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 63) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 100) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 200) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 230)]
df_subset = df_subset[df_subset.Al_Shielding_Thickness_gcm2 == 1.0]
#df_subset = df_subset[df_subset.Sample_thickness_um == 500.0]
# E
matplotlib.pyplot.figure()
g = seaborn.factorplot(y="Orbit", x='Slope_Year_1E11p_cm2_e', hue="Test_Energy_MeV", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True, orient='h')
g._legend.set_title('Test \nEnergy [MeV]')
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.7)
matplotlib.pyplot.show()
# N
matplotlib.pyplot.figure()
g = seaborn.factorplot(y="Orbit", x='Slope_Year_1E11p_cm2_n', hue="Test_Energy_MeV", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True, orient='h')
g._legend.set_title('Test \nEnergy [MeV]')
g._legend.get_title().set_fontsize(s-2)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Nuclear")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.7)
matplotlib.pyplot.show()


# In[17]:


orbits_labels_new = ['LEO Polar\nSun-sync\n600km', 'LEO Polar\nSun-sync\n800km', 'LEO Inclined\nNonpolar ISS', 'MEO Molniya', 'MEO Semi\nSync GPS', 'HEO Highly\nEccentric\nIBEX', 'HEO\nGeostationary']


seaborn.set_style("ticks")

s = 10
params = {'legend.fontsize': s-1,
         'axes.labelsize': s,
         'axes.titlesize':  s-8,
         'xtick.labelsize': s,
         'ytick.labelsize': s,
         "lines.linewidth": 3.0}

pylab.rcParams.update(params)


fig = matplotlib.pyplot.figure(dpi=200, figsize=(7,5))
gs2 = gridspec.GridSpec(1,2)
gs2.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 

df_subset = All_equivalence_iso_fluence_slope[(All_equivalence_iso_fluence_slope.Test_Energy_MeV == 45) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 63) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 100) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 200) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 230)]
df_subset = df_subset[df_subset.Al_Shielding_Thickness_gcm2 == 1.0]

axv = plt.subplot(gs2[0])
#g = seaborn.factorplot(y="Orbit", x='Slope_Year_1E11p_cm2_e', hue="Test_Energy_MeV", data=df_subset, saturation=5, aspect=3, kind="box", ax = axv)
g = seaborn.boxplot(y="Orbit", x='Slope_Year_1E11p_cm2_e', hue="Test_Energy_MeV", data=df_subset, linewidth=0.5, fliersize=1)   

matplotlib.pyplot.grid(b=None, which='major', axis='x', color='grey', linestyle='-', linewidth=.7)
for y_val in [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    axv.axhline(y_val, color='grey', linestyle='-', linewidth=.7)

matplotlib.pyplot.title('Electronic', size=s)
g.set_yticklabels(orbits_labels_new)

g.set(xlabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence")
g.set(xscale="log")
#g.tick_params(axis='x', labelbottom=False)
#ax.set_ylabel('')
#ax.set_xlabel('')
#matplotlib.pyplot.ylabel("Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic", size=s)
#g.set(yscale="log")
#seaborn.despine()

axv.set_ylabel('')

 
axv = plt.subplot(gs2[1])
g = seaborn.boxplot(y="Orbit", x='Slope_Year_1E11p_cm2_n', hue="Test_Energy_MeV", data=df_subset, linewidth=0.5, fliersize=1)   

matplotlib.pyplot.grid(b=None, which='major', axis='x', color='grey', linestyle='-', linewidth=.7)
for y_val in [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    axv.axhline(y_val, color='grey', linestyle='-', linewidth=.7)

matplotlib.pyplot.title('Nuclear', size=s)
matplotlib.pyplot.tick_params(axis='y', labelleft=False)
g.set(xlabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence")
g.set(xscale="log")
#ax.set_xlabel('')
#matplotlib.pyplot.ylabel("Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic", size=s)
#g.set(yscale="log")
#seaborn.despine()

axv.set_ylabel('')
axv.legend_.remove()

matplotlib.pyplot.show()


# In[18]:


Al_shielding_gcm2_dict_subset = {'0.0': 0.0, '0p5': 0.5, '5p0': 5.0}
seaborn.set_style("ticks")


s = 16
params = {'legend.fontsize': s-1,
         'axes.labelsize': s+4,
         'axes.titlesize':  s-5,
         'xtick.labelsize': s,
         'ytick.labelsize': s,
         "lines.linewidth": 3.0}

pylab.rcParams.update(params)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(20,10))
gs1 = gridspec.GridSpec(2,3)
gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 


df_subset_energies = All_equivalence_iso_fluence_slope[(All_equivalence_iso_fluence_slope.Test_Energy_MeV == 45) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 63) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 100) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 200) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 230)]


for num_Al, Al_thickness in enumerate(Al_shielding_gcm2_dict_subset):
    ax = plt.subplot(gs1[num_Al])
    
    df_subset = df_subset_energies[df_subset_energies.Al_Shielding_Thickness_gcm2 == Al_shielding_gcm2_dict_subset[Al_thickness]]
    g = seaborn.lineplot(x="Sample_thickness_um", y='Slope_Year_1E11p_cm2_e', hue="Orbit", data=df_subset, ci="sd" , legend=False)
    
    
    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.7)    
    matplotlib.pyplot.title('Al thickness ' + str(Al_shielding_gcm2_dict[Al_thickness]) + r' g/cm$^2$', size=s+2)
    g.tick_params(axis='x', labelbottom=False)
    
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    if num_Al == 0:
        matplotlib.pyplot.ylabel("Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Electronic", size=s)
    else:
        matplotlib.pyplot.tick_params(axis='y', labelleft=False)
    g.set(yscale="log")
    #seaborn.despine()
    matplotlib.pyplot.ylim([10**(-5), 10**4])
    
for num_Al, Al_thickness in enumerate(Al_shielding_gcm2_dict_subset):
    this_A = All_equivalence_iso[All_equivalence_iso.Al_Shielding_Thickness_gcm2 == Al_shielding_gcm2_dict[Al_thickness]]
    ax = plt.subplot(gs1[num_Al+3])

    g.set(yscale="log")
    df_subset = df_subset_energies[df_subset_energies.Al_Shielding_Thickness_gcm2 == Al_shielding_gcm2_dict_subset[Al_thickness]]
    g = seaborn.lineplot(x="Sample_thickness_um", y='Slope_Year_1E11p_cm2_n', hue="Orbit", data=df_subset,  ci="sd" ,legend=False)
    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.7)
    
    ax.set_ylabel("")
    matplotlib.pyplot.xlabel(r'Sample Thickness [$\mu$m]', size=s)
    
    if num_Al == 0:
        matplotlib.pyplot.ylabel("Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence - Nuclear", size=s+2)
    else:
        matplotlib.pyplot.tick_params(axis='y', labelleft=False)
        
    if num_Al == 2:
        ax.legend(orbits_legend.values(), fontsize=(s-1), loc = 'lower center')
    g.set(yscale="log")
    #seaborn.despine()
    matplotlib.pyplot.ylim([10**(-5), 10**4])

matplotlib.pyplot.show()




# In[19]:


All_equivalence_iso_fluence_slope['n_e_Slope_Ratio'] = All_equivalence_iso_fluence_slope['Slope_Year_1E11p_cm2_n']/All_equivalence_iso_fluence_slope['Slope_Year_1E11p_cm2_e']
All_equivalence_iso_fluence['n_e_Ratio'] = All_equivalence_iso_fluence['Nuclear_Equivalence_s']/ All_equivalence_iso_fluence['Electronic_Equivalence_s']

Al_shielding_gcm2_dict_subset = {'0.0': 0.0, '0p1': 0.1,'1p0': 1.0, '5p0': 5.0}
seaborn.set_style("ticks")


s = 16
params = {'legend.fontsize': s-1,
         'axes.labelsize': s+4,
         'axes.titlesize':  s-8,
         'xtick.labelsize': s,
         'ytick.labelsize': s,
         "lines.linewidth": 3.0}

pylab.rcParams.update(params)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(12,10))
gs1 = gridspec.GridSpec(2,2)
gs1.update(wspace=0.03, hspace=0.12) # set the spacing between axes. 


df_subset_energies = All_equivalence_iso_fluence[(All_equivalence_iso_fluence.Test_Energy_MeV == 45) | (All_equivalence_iso_fluence.Test_Energy_MeV == 63) | (All_equivalence_iso_fluence.Test_Energy_MeV == 100) | (All_equivalence_iso_fluence.Test_Energy_MeV == 200) | (All_equivalence_iso_fluence.Test_Energy_MeV == 230)]

#df_subset_energies = All_equivalence_iso_fluence_slope[(All_equivalence_iso_fluence_slope.Test_Energy_MeV == 45) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 63) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 100) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 200) | (All_equivalence_iso_fluence_slope.Test_Energy_MeV == 230)]


for num_Al, Al_thickness in enumerate(Al_shielding_gcm2_dict_subset):
    ax = plt.subplot(gs1[num_Al])
    
    df_subset = df_subset_energies[df_subset_energies.Al_Shielding_Thickness_gcm2 == Al_shielding_gcm2_dict_subset[Al_thickness]]
    g = seaborn.lineplot(x="Sample_thickness_um", y='n_e_Ratio', hue="Orbit", data=df_subset, ci="sd" , legend=False)
    #g = seaborn.lineplot(x="Sample_thickness_um", y='n_e_Slope_Ratio', hue="Orbit", data=df_subset, ci="sd" , legend=False)
    
    
    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.7)    
    matplotlib.pyplot.title('Al thickness ' + str(Al_shielding_gcm2_dict[Al_thickness]) + r' g/cm$^2$', size=s)
    
    
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    if (num_Al == 0) or (num_Al == 2):
        matplotlib.pyplot.ylabel("Ratio of Nuclear to Electronic \n Orbit Time Equivalence", size=s)
    else:
        g.tick_params(axis='y', labelleft=False)

        
    if (num_Al == 2) or (num_Al == 3):   
        matplotlib.pyplot.xlabel(r'Sample Thickness [$\mu$m]', size=s)
    else:
        g.tick_params(axis='x', labelbottom=False)

        
    if num_Al == 3:
        ax.legend(orbits_legend.values(), fontsize=(s-3), loc = 'upper center')
        
    ax.set_ylim([0.1, 500])


    g.set(yscale="log")
    #seaborn.despine()
    
# NUMBER THAT ARE LESS THAN 1

# LEO:
# ratio decreases with Al shielding
# ratio decreases with material mat thickness
# ratio increases with test E

# MEO:
# ratio incereases 0-0.1 then decreases with Al shielding
# ratio decreases with material mat thickness
# ratio increases with test E

# HEO:
# ratio incereases 0-0.5 then decreases with Al shielding
# ratio decreases with material mat thickness
# ratio increases with test E


# In[27]:


Al_thicks_long = Al_thicks + list(Al_thicks)
mats_long = mats + list(mats)
orbits_long = orbits + list(orbits) 
testEs_long = testEs + list(testEs)
max_distances_long = max_distances + list(max_distances)

n_label = ['n'] * len(Al_thicks)
e_label = ['e'] * len(Al_thicks)

# years per 10**11 p/cm2
slope_long = slope_n + slope_e
type_long = n_label + e_label

All_equivalence_data_iso_fluence_slope_long = {'Sample_thickness_um':numpy.array(max_distances_long), 'Al_Shielding_Thickness_gcm2':numpy.array(Al_thicks_long), 'Material':numpy.array(mats_long), 'Orbit':numpy.array(orbits_long), 'Test_Energy_MeV':numpy.array(testEs_long), 'Type': numpy.array(type_long), 'Slope_Year_1E11p_cm2': numpy.array(slope_long)}
All_equivalence_iso_fluence_slope_long = pd.DataFrame(All_equivalence_data_iso_fluence_slope_long)             

s = 18
import matplotlib.pylab as pylab
params = {'legend.fontsize': s-1,
          'figure.figsize': (15, 5),
         'axes.labelsize': s,
         'axes.titlesize':  s,
         'xtick.labelsize': s-2,
         'ytick.labelsize': s-2}
pylab.rcParams.update(params)


df_subset = All_equivalence_iso_fluence_slope_long[All_equivalence_iso_fluence_slope_long.Sample_thickness_um == 100]
matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y="Slope_Year_1E11p_cm2", hue="Type", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()

df_subset = All_equivalence_iso_fluence_slope_long[All_equivalence_iso_fluence_slope_long.Al_Shielding_Thickness_gcm2 == 0.0]
matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y="Slope_Year_1E11p_cm2", hue="Type", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()

df_subset = All_equivalence_iso_fluence_slope_long[(All_equivalence_iso_fluence_slope_long.Al_Shielding_Thickness_gcm2 == 1.0) | (All_equivalence_iso_fluence_slope_long.Al_Shielding_Thickness_gcm2 == 0.0)]
matplotlib.pyplot.figure()
g = seaborn.factorplot(x="Orbit", y="Slope_Year_1E11p_cm2", hue="Type", data=df_subset, saturation=5, aspect=3, kind="box", legend_out=True)
g.set(yscale="log")
g.set(ylabel="Orbit Equivalence Years per \n10$^{11}$ p/cm$^2$ Test Fluence")
g.set_xticklabels(orbits_labels)
matplotlib.pyplot.show()



# In[29]:


df_subset = All_equivalence_iso_fluence_slope_long
matplotlib.pyplot.figure()
g = seaborn.lineplot(x="Sample_thickness_um", y="Slope_Year_1E11p_cm2", hue="Orbit", ci="sd" , data=df_subset[df_subset.Type == 'n'])
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
g = seaborn.lineplot(x="Al_Shielding_Thickness_gcm2", y="Slope_Year_1E11p_cm2", hue="Orbit" , ci="sd" , data=df_subset[df_subset.Type == 'n'])
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
g = seaborn.lineplot(x="Test_Energy_MeV", y="Slope_Year_1E11p_cm2", hue="Orbit" ,ci="sd", data=df_subset[df_subset.Type == 'n'])
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
g = seaborn.lineplot(x="Test_Energy_MeV", y="Slope_Year_1E11p_cm2",ci="sd",  hue="Type" , data=df_subset)
matplotlib.pyplot.show()

