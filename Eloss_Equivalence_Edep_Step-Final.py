#!/usr/bin/env python
# coding: utf-8

# In[14]:


#import sys
#!{sys.executable} -m pip install --upgrade uncertainties
#!{sys.executable} -m pip install --upgrade periodictable
#!{sys.executable} -m pip install --upgrade brewer2mpl
#!{sys.executable} -m pip install PAScual

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
material_densities_keys =     {"CdTe": "CdTe", "ZnS": "ZnS", "ZnSe": "ZnSe", "ZnTe": "ZnTe", "InSb": "InSb", "In5Ga5As": r"In$_{0.5}$Ga$_{0.5}$As", "In5Ga5Sb":  "In$_{0.5}$Ga$_{0.5}$Sb", "Hg5Cd5Te": "Hg$_{0.5}$Cd$_{0.5}$Te", "GaN":"GaN", "ZnO":"ZnO", "MgO":"MgO",  "GaP":"GaP", "InAs":"InAs", "InP":"InP", "Si":"Si"}


# In[15]:


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


filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded/WITH_MAGNETOSPHERE_SHIELDING/'
for Al_thickness in Al_shielding_gcm2_names:
    for orbit_keyvalue, filename in filenames.items():
        fn = filepath + Al_thickness + '/' + filename
        keyvalue = orbit_keyvalue + '_' + Al_thickness
        
        print(keyvalue)

        proton_spectrum = pd.read_csv(fn, sep=',', skiprows=13, skipfooter=121, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (proton_spectrum['Energy_keV_high'] - proton_spectrum['Energy_keV_low'])/1000.0
        proton_spectrum['DFlux_MeVcm2s'] = (proton_spectrum.Flux_cm2bin/E_width)/mission_time_s
        

        electron_spectrum = pd.read_csv(fn, sep=',', skiprows=73, skipfooter=62, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (electron_spectrum['Energy_keV_high'] - electron_spectrum['Energy_keV_low'])/1000.0
        electron_spectrum['DFlux_MeVcm2s'] = (electron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_electrons_yesMagnetosphere[keyvalue] = electron_spectrum
        
        neutron_spectrum = pd.read_csv(fn, sep=',', skiprows=133, skipfooter=1, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (neutron_spectrum['Energy_keV_high'] - neutron_spectrum['Energy_keV_low'])/1000.0
        neutron_spectrum['DFlux_MeVcm2s'] = (neutron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_neutrons_yesMagnetosphere[keyvalue] = neutron_spectrum
        
        fn = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded_GEANT4_SolarHe/' + orbit_keyvalue + '_SolarHeSpectrum_' + Al_thickness + '_Al.txt' 
        solarHe_spectrum = pd.read_csv(fn, sep = ' ', names=['E_MeV', 'DFlux_MeVcm2s'])
        Alshielded_solarHe_yesMagnetosphere[keyvalue] = solarHe_spectrum 

        
        fn = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded_GEANT4_GCRH/' + orbit_keyvalue + '_GCRHSpectrum_' + Al_thickness + '_Al.txt' 
        GCRH_spectrum = pd.read_csv(fn, sep = ' ', names=['E_MeV', 'DFlux_MeVcm2s'])
        proton_spectrum['DFlux_MeVcm2s'] = proton_spectrum['DFlux_MeVcm2s'] + GCRH_spectrum['DFlux_MeVcm2s']
        Alshielded_GCRH_yesMagnetosphere[keyvalue] = GCRH_spectrum
        
        Alshielded_protons_yesMagnetosphere[keyvalue] = proton_spectrum

Alshielded_protons_noMagnetosphere = {}
Alshielded_electrons_noMagnetosphere = {}
Alshielded_neutrons_noMagnetosphere = {}

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
        
# this contains all of the proton energies to propogate through for all orbits and shieldings, MeV
protons_Al_shielded_bin_energies_all_yesMagnetosphere = (0.5/1000.0)*(Alshielded_protons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_low']+Alshielded_protons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_high'])

# this contains all of the proton energies to propogate through for all orbits and shieldings, MeV
electrons_Al_shielded_bin_energies_all_yesMagnetosphere = (0.5/1000.0)*(Alshielded_electrons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_low']+Alshielded_electrons_yesMagnetosphere['LEO_600_0p1']['Energy_keV_high'])


# In[16]:


# sum up for without Al shielding:

# this contains all of the proton energies to propogate through for all orbits and with no shielding
protons_unshielded_bin_energies_all_yesMagnetosphere = numpy.append(numpy.array(unshielded_solar_protons_yesMagnetosphere['LEO_600']['Energy_MeV']), [5.5e+02, 6.0e+02, 6.5e+02, 7.0e+02, 7.5e+02, 8.0e+02, 8.5e+02, 9.0e+02, 9.5e+02, 1.0e+03])
protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s = {}

# for each orbit
for keyvalue, filename in filenames.items():
    this_sum = numpy.zeros(len(protons_unshielded_bin_energies_all_yesMagnetosphere))
    # for each energy in the orbit
    for num, E_current in enumerate(protons_unshielded_bin_energies_all_yesMagnetosphere):
        this_E_sum = 0       
        orbit_function = interp1d(unshielded_trapped_protons[keyvalue].Energy_MeV, unshielded_trapped_protons[keyvalue].DFlux_MeVcm2s, kind='quadratic', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + orbit_function(E_current)
        orbit_function = interp1d(unshielded_solar_protons_yesMagnetosphere[keyvalue].Energy_MeV, unshielded_solar_protons_yesMagnetosphere[keyvalue].DFlux_MeVcm2s*unshielded_solar_protons_yesMagnetosphere[keyvalue].AttenFactor, kind='quadratic', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + orbit_function(E_current)
        orbit_function = interp1d(unshielded_H_galactic_yesMagnetosphere[keyvalue].Energy_MeV, (1.0/10000.0)*math.pi*unshielded_H_galactic_yesMagnetosphere[keyvalue].H_DFlux_m2sSrMeV, kind='quadratic', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + orbit_function(E_current)
        
        this_sum[num] = this_E_sum
        
    protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue] = this_sum

    
electrons_unshielded_bin_energies_all_yesMagnetosphere = numpy.array(unshielded_trapped_electrons['LEO_600']['Energy_MeV'])
electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s = {}
for keyvalue, filename in filenames.items():
    electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue] = numpy.array(unshielded_trapped_electrons[keyvalue].DFlux_MeVcm2s)



# In[17]:


# Read in the nuclear energy loss for protons

fn = 'NIEL_SRNIEL/p/spenvis_niel_Si.txt'
dE_dx_nuclear_Si = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_MgO.txt'
dE_dx_nuclear_MgO = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_ZnO.txt'
dE_dx_nuclear_ZnO = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_InAs.txt'
dE_dx_nuclear_InAs = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_InP.txt'
dE_dx_nuclear_InP = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_GaP.txt'
dE_dx_nuclear_GaP = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_GaN.txt'
dE_dx_nuclear_GaN = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')


fn = 'NIEL_SRNIEL/p/spenvis_niel_Hg5Cd5Te.txt'
dE_dx_nuclear_Hg5Cd5Te = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_In5Ga5Sb.txt'
dE_dx_nuclear_In5Ga5Sb = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_In5Ga5As.txt'
dE_dx_nuclear_In5Ga5As = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_ZnTe.txt'
dE_dx_nuclear_ZnTe = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_ZnSe.txt'
dE_dx_nuclear_ZnSe = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_ZnS.txt'
dE_dx_nuclear_ZnS = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_CdTe.txt'
dE_dx_nuclear_CdTe = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/p/spenvis_niel_InSb.txt'
dE_dx_nuclear_InSb = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')


dE_dx_nuclear = {
  "Si": dE_dx_nuclear_Si,
  "MgO": dE_dx_nuclear_MgO,
  "ZnO": dE_dx_nuclear_ZnO,
  "InAs": dE_dx_nuclear_InAs,
  "InP": dE_dx_nuclear_InP,
  "GaP": dE_dx_nuclear_GaP,
  "GaN": dE_dx_nuclear_GaN,
    "Hg5Cd5Te": dE_dx_nuclear_Hg5Cd5Te,
    "In5Ga5Sb": dE_dx_nuclear_In5Ga5Sb ,
    "In5Ga5As": dE_dx_nuclear_In5Ga5As ,
    "ZnTe": dE_dx_nuclear_ZnTe ,
    "ZnSe": dE_dx_nuclear_ZnSe ,
    "ZnS": dE_dx_nuclear_ZnS,
    "CdTe": dE_dx_nuclear_CdTe, 
    "InSb": dE_dx_nuclear_InSb 

}



# Read in electronic energy loss for protons

# Energy_MeV', 'Si_EstoppingPower_MeVcm2g', 'Si_IonizingDose_MeVg', 'Si_IonizingDose_MeVg.1', 'GaN_EstoppingPower_MeVcm2g','GaN_IonizingDose_MeVg', 'GaN_IonizingDose_MeVg.1','GaP_EstoppingPower_MeVcm2g', 'GaP_IonizingDose_MeVg','GaP_IonizingDose_MeVg.1', 'InP_EstoppingPower_MeVcm2g', 'InP_IonizingDose_MeVg', 'InP_IonizingDose_MeVg.1','InAs_EstoppingPower_MeVcm2g', 'InAs_IonizingDose_MeVg','InAs_IonizingDose_MeVg.1', 'MgO_EstoppingPower_MeVcm2g','MgO_IonizingDose_MeVg', 'MgO_IonizingDose_MeVg.1','ZnO_EstoppingPower_MeVcm2g', 'ZnO_IonizingDose_MeVg','ZnO_IonizingDose_MeVg.1'
fn = "ELECTRONIC_ELOSS/p/SR_NIEL_StoppingPowers_Electronic.csv"
dE_dx_electronic = pd.read_csv(fn)
dE_dx_electronic.dropna(how='all', axis='columns', inplace=True)
dE_dx_electronic.dropna(inplace=True)



# In[18]:


# Read in the nuclear energy loss for electrons

fn = 'NIEL_SRNIEL/e/spenvis_niel_Si.txt'
dE_dx_nuclear_Si = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_MgO.txt'
dE_dx_nuclear_MgO = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_ZnO.txt'
dE_dx_nuclear_ZnO = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_InAs.txt'
dE_dx_nuclear_InAs = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_InP.txt'
dE_dx_nuclear_InP = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_GaP.txt'
dE_dx_nuclear_GaP = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_GaN.txt'
dE_dx_nuclear_GaN = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')


fn = 'NIEL_SRNIEL/e/spenvis_niel_Hg5Cd5Te.txt'
dE_dx_nuclear_Hg5Cd5Te = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_In5Ga5Sb.txt'
dE_dx_nuclear_In5Ga5Sb = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_In5Ga5As.txt'
dE_dx_nuclear_In5Ga5As = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_ZnTe.txt'
dE_dx_nuclear_ZnTe = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_ZnSe.txt'
dE_dx_nuclear_ZnSe = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_ZnS.txt'
dE_dx_nuclear_ZnS = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_CdTe.txt'
dE_dx_nuclear_CdTe = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')
fn = 'NIEL_SRNIEL/e/spenvis_niel_InSb.txt'
dE_dx_nuclear_InSb = pd.read_csv(fn, sep=',', skiprows=15, skipfooter=2, names=['Energy_MeV', 'NIEL_MeVcm2_g'], engine='python')


dE_dx_nuclear_electrons = {
  "Si": dE_dx_nuclear_Si,
  "MgO": dE_dx_nuclear_MgO,
  "ZnO": dE_dx_nuclear_ZnO,
  "InAs": dE_dx_nuclear_InAs,
  "InP": dE_dx_nuclear_InP,
  "GaP": dE_dx_nuclear_GaP,
  "GaN": dE_dx_nuclear_GaN,
    "Hg5Cd5Te": dE_dx_nuclear_Hg5Cd5Te,
    "In5Ga5Sb": dE_dx_nuclear_In5Ga5Sb ,
    "In5Ga5As": dE_dx_nuclear_In5Ga5As ,
    "ZnTe": dE_dx_nuclear_ZnTe ,
    "ZnSe": dE_dx_nuclear_ZnSe ,
    "ZnS": dE_dx_nuclear_ZnS,
    "CdTe": dE_dx_nuclear_CdTe, 
    "InSb": dE_dx_nuclear_InSb 

}



# Read in electronic energy loss for electrons

# Energy_MeV', 'Si_EstoppingPower_MeVcm2g', 'Si_IonizingDose_MeVg', 'Si_IonizingDose_MeVg.1', 'GaN_EstoppingPower_MeVcm2g','GaN_IonizingDose_MeVg', 'GaN_IonizingDose_MeVg.1','GaP_EstoppingPower_MeVcm2g', 'GaP_IonizingDose_MeVg','GaP_IonizingDose_MeVg.1', 'InP_EstoppingPower_MeVcm2g', 'InP_IonizingDose_MeVg', 'InP_IonizingDose_MeVg.1','InAs_EstoppingPower_MeVcm2g', 'InAs_IonizingDose_MeVg','InAs_IonizingDose_MeVg.1', 'MgO_EstoppingPower_MeVcm2g','MgO_IonizingDose_MeVg', 'MgO_IonizingDose_MeVg.1','ZnO_EstoppingPower_MeVcm2g', 'ZnO_IonizingDose_MeVg','ZnO_IonizingDose_MeVg.1'
fn = "ELECTRONIC_ELOSS/e/ESTAR_StoppingPower.csv"
dE_dx_electronic_electrons = pd.read_csv(fn)
dE_dx_electronic_electrons.dropna(how='all', axis='columns', inplace=True)
dE_dx_electronic_electrons.dropna(inplace=True)


# In[19]:


# Read in the nuclear energy loss for ions

ions = ['He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']

dE_dx_nuclear_ions = {}
matplotlib.pyplot.figure(dpi=200)
for ion in ions:
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_Si.txt'
    dE_dx_nuclear_Si = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_MgO.txt'
    dE_dx_nuclear_MgO = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_ZnO.txt'
    dE_dx_nuclear_ZnO = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_InAs.txt'
    dE_dx_nuclear_InAs = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_InP.txt'
    dE_dx_nuclear_InP = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_GaP.txt'
    dE_dx_nuclear_GaP = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_GaN.txt'
    dE_dx_nuclear_GaN = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')


    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_Hg5Cd5Te.txt'
    dE_dx_nuclear_Hg5Cd5Te = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_In5Ga5Sb.txt'
    dE_dx_nuclear_In5Ga5Sb = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_In5Ga5As.txt'
    dE_dx_nuclear_In5Ga5As = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_ZnTe.txt'
    dE_dx_nuclear_ZnTe = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_ZnSe.txt'
    dE_dx_nuclear_ZnSe = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_ZnS.txt'
    dE_dx_nuclear_ZnS = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_CdTe.txt'
    dE_dx_nuclear_CdTe = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'NIEL_SRNIEL/'+ion+'/spenvis_niel_InSb.txt'
    dE_dx_nuclear_InSb = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'NIEL_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')


    dE_dx_nuclear_ions[ion] = {
      "Si": dE_dx_nuclear_Si,
      "MgO": dE_dx_nuclear_MgO,
      "ZnO": dE_dx_nuclear_ZnO,
      "InAs": dE_dx_nuclear_InAs,
      "InP": dE_dx_nuclear_InP,
      "GaP": dE_dx_nuclear_GaP,
      "GaN": dE_dx_nuclear_GaN,
        "Hg5Cd5Te": dE_dx_nuclear_Hg5Cd5Te,
        "In5Ga5Sb": dE_dx_nuclear_In5Ga5Sb ,
        "In5Ga5As": dE_dx_nuclear_In5Ga5As ,
        "ZnTe": dE_dx_nuclear_ZnTe ,
        "ZnSe": dE_dx_nuclear_ZnSe ,
        "ZnS": dE_dx_nuclear_ZnS,
        "CdTe": dE_dx_nuclear_CdTe, 
        "InSb": dE_dx_nuclear_InSb }
    
    matplotlib.pyplot.loglog(dE_dx_nuclear_ions[ion]['Si']['Energy_MeV'],   dE_dx_nuclear_ions[ion]['Si'][ 'NIEL_MeVcm2_g'])

matplotlib.pyplot.legend(ions)
matplotlib.pyplot.show()


# Read in the electronic energy loss of the ions

dE_dx_electronic_ions = {}
matplotlib.pyplot.figure(dpi=200)
for ion in ions:
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_Si.txt'
    dE_dx_electronic_Si = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_MgO.txt'
    dE_dx_electronic_MgO = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_ZnO.txt'
    dE_dx_electronic_ZnO = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_InAs.txt'
    dE_dx_electronic_InAs = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_InP.txt'
    dE_dx_electronic_InP = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_GaP.txt'
    dE_dx_electronic_GaP = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_GaN.txt'
    dE_dx_electronic_GaN = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')


    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_Hg5Cd5Te.txt'
    dE_dx_electronic_Hg5Cd5Te = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_In5Ga5Sb.txt'
    dE_dx_electronic_In5Ga5Sb = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_In5Ga5As.txt'
    dE_dx_electronic_In5Ga5As = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_ZnTe.txt'
    dE_dx_electronic_ZnTe = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_ZnSe.txt'
    dE_dx_electronic_ZnSe = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_ZnS.txt'
    dE_dx_electronic_ZnS = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_CdTe.txt'
    dE_dx_electronic_CdTe = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')
    fn = 'ELECTRONIC_ELOSS/'+ion+'/spenvis_niel_InSb.txt'
    dE_dx_electronic_InSb = pd.read_csv(fn, sep='\t', skiprows=5, error_bad_lines=False, names=['Energy_MeV', 'ELECT_MeVcm2_g', 'Dose_MeV_g', 'Dose_Gy'], comment='*', engine='python')


    dE_dx_electronic_ions[ion] = {
      "Si": dE_dx_electronic_Si,
      "MgO": dE_dx_electronic_MgO,
      "ZnO": dE_dx_electronic_ZnO,
      "InAs": dE_dx_electronic_InAs,
      "InP": dE_dx_electronic_InP,
      "GaP": dE_dx_electronic_GaP,
      "GaN": dE_dx_electronic_GaN,
        "Hg5Cd5Te": dE_dx_electronic_Hg5Cd5Te,
        "In5Ga5Sb": dE_dx_electronic_In5Ga5Sb ,
        "In5Ga5As": dE_dx_electronic_In5Ga5As ,
        "ZnTe": dE_dx_electronic_ZnTe ,
        "ZnSe": dE_dx_electronic_ZnSe ,
        "ZnS": dE_dx_electronic_ZnS,
        "CdTe": dE_dx_electronic_CdTe, 
        "InSb": dE_dx_electronic_InSb 

    }


    matplotlib.pyplot.loglog(dE_dx_electronic_ions[ion]['Si']['Energy_MeV'],   dE_dx_electronic_ions[ion]['Si'][ 'ELECT_MeVcm2_g'])

matplotlib.pyplot.legend(ions)
matplotlib.pyplot.show()




# In[63]:


# plot ratio of nuclear to electronic energy loss

matplotlib.pyplot.figure(dpi=200)
for num, sm in enumerate(material_densities):  
    if True:
    
        density = material_densities[sm]

        # nuclear MeV/cm
        dE_dx_nuclear_current = dE_dx_nuclear[sm]['NIEL_MeVcm2_g']*density
        E_MeV_nuclear_current = dE_dx_nuclear[sm]['Energy_MeV']

        # electronic MeV/cm
        name_e = sm +'_EstoppingPower_MeVcm2g'
        dE_dx_electronic_current = dE_dx_electronic[name_e]*density
        E_MeV_electronic_current = dE_dx_electronic['Energy_MeV']  

        # build the functions        
        nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
        electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')
        
        testE = 63
        E1 = 100
        E2 = 0.001
        print((nuclear_stopping_function(testE)/nuclear_stopping_function(E1))/(nuclear_stopping_function(testE)/nuclear_stopping_function(E2)))
        print((electronic_stopping_function(testE)/electronic_stopping_function(E1))/(electronic_stopping_function(testE)/electronic_stopping_function(E2)))
        print('....')
        print((nuclear_stopping_function(testE)/nuclear_stopping_function(E1))/(electronic_stopping_function(testE)/electronic_stopping_function(E1)))
        print((nuclear_stopping_function(testE)/nuclear_stopping_function(E2))/(electronic_stopping_function(testE)/electronic_stopping_function(E2)))

        
        E = numpy.geomspace(10**(-3), 1000, 100)
        nuc_elect_ratio_p = nuclear_stopping_function(E)/electronic_stopping_function(E)
        matplotlib.pyplot.loglog(E,nuc_elect_ratio_p)
        matplotlib.pyplot.xlabel('Energy [MeV]')
        matplotlib.pyplot.ylabel('proton [dE/dx]$_n$/[dE/dx]$_e$')

        
        
#matplotlib.pyplot.axvline(0.22, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(22, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(44, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(50, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(50, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(50, 10**(-4), 10**(2), color='gray')

#matplotlib.pyplot.axvline(0.16, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(6, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(15, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(22, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(28, 10**(-4), 10**(2), color='gray')
#matplotlib.pyplot.axvline(37, 10**(-4), 10**(2), color='gray')



#matplotlib.pyplot.axvline(63, 10**(-4), 10**(2), color='black')
#matplotlib.pyplot.axvline(100, 10**(-4), 10**(2), color='black')
#matplotlib.pyplot.axvline(200, 10**(-4), 10**(2), color='black')
matplotlib.pyplot.legend(material_densities_keys.values(), fontsize=8,ncol=3)     
matplotlib.pyplot.xlim([10**(-3), 10**3])
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.show()


# In[7]:


# THESE STORE EVERYTHING!!

# for each material, for each energy bin, it gives these values as a function of distance
nuclear_energy_loss_allMat_Al_shielded = {}
electronic_energy_loss_allMat_Al_shielded = {}
currentE_allMat_Al_shielded = {}

# for each material, for each energy bin, it gives these values as a function of distance
nuclear_energy_loss_allMat_unshielded = {}
electronic_energy_loss_allMat_unshielded = {}
currentE_allMat_unshielded = {}

# for each material, for the energy value, it gives these values as a functino of distance
nuclear_energy_loss_allMat_test = {}
electronic_energy_loss_allMat_test = {}
currentE_allMat_test = {}

#  = 0.1 um (i.e. tiny!)
distance_step_cm = 0.00001
distance_array_cm = numpy.arange(0,0.2,distance_step_cm)

# because we stored every 10 in the initial spectrum which is 1 micron and want to store every 10 so increment by 10 each time
# this gives you in micron units (so 10 is 10 microns)
bins_10micron = numpy.arange(0,int(len(distance_array_cm)/10),10)


# In[8]:


# For just energies of test

test_energies = {'63':63.0, '70':70.0, '100':100.0, '200':200.0, '230':230.0}
                   
# For test energies

# Store in nuclear_energy_loss_allMat_test, electronic_energy_loss_allMat_test, currentE_allMat_test:
# For each material, for each energy in the spectrum, the 1 um energy, or losses

# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear[sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear[sm]['Energy_MeV']
                      
    # electronic MeV/cm
    name_e = sm +'_EstoppingPower_MeVcm2g'
    dE_dx_electronic_current = dE_dx_electronic[name_e]*density
    E_MeV_electronic_current = dE_dx_electronic['Energy_MeV']  
        
    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')
    
    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    
    # incident energies from all of the orbits without shielding
    energies_MeV = test_energies.values()
            
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current)
                
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 
                    

                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1e-06:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            # means you have gone through 1 micron
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss total
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
        # these are each as a function of distance in 1 um bins          
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
    # these are all dictionaries by incident energy   
    nuclear_energy_loss_allMat_test[sm] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_test[sm] = electronic_energy_loss_thisMat
    currentE_allMat_test[sm] = currentE_thisMat


# In[180]:


energies_MeV = test_energies.values()


w = csv.writer(open("electronic_E_loss_vsEin_vsDistance_test.csv", "a"))    
# for each material
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, sum(electronic_energy_loss_allMat_test[sm][E_in][i:(i+10)])])

w = csv.writer(open("nuclear_E_loss_vsEin_vsDistance_test.csv", "a"))    
# for each material
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, sum(nuclear_energy_loss_allMat_test[sm][E_in][i:(i+10)])])
   
w = csv.writer(open("E_vsEin_vsDistance_test.csv", "a"))    
# for each material
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, (1.0/10.0)*sum(currentE_allMat_test[sm][E_in][i:(i+10)])])


# In[181]:


# Without aluminum shielding - FOR THE PROTONS

# NOTE: for all with aluminum shielding, the energy bins are the same
# as a result you only need to store for each material for the given energy bins

# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear[sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear[sm]['Energy_MeV']
                      
    # electronic MeV/cm
    name_e = sm +'_EstoppingPower_MeVcm2g'
    dE_dx_electronic_current = dE_dx_electronic[name_e]*density
    E_MeV_electronic_current = dE_dx_electronic['Energy_MeV']  
        
    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')

    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    # incident energies from all of the orbits without shielding
    energies_MeV = protons_unshielded_bin_energies_all_yesMagnetosphere
            
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
         
        # for each distance, calculate the energy loss
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current)   
                
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 


                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1e-06:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                        
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
                    
        
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
        
    nuclear_energy_loss_allMat_unshielded[sm] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_unshielded[sm] = electronic_energy_loss_thisMat
    currentE_allMat_unshielded[sm] = currentE_thisMat


# In[11]:


# Without aluminum shielding - FOR THE ELECTRONS

# NOTE: for all with aluminum shielding, the energy bins are the same
# as a result you only need to store for each material for the given energy bins

# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear_electrons[sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear_electrons[sm]['Energy_MeV']
                      
    # electronic MeV/cm
    name_e = sm +'_Tot_MeVcm2_g'
    dE_dx_electronic_current = dE_dx_electronic_electrons[name_e]*density
    E_MeV_electronic_current = dE_dx_electronic_electrons['Energy_MeV'] 
    
        
    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')

    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    # incident energies from all of the orbits without shielding
    energies_MeV = electrons_unshielded_bin_energies_all_yesMagnetosphere
            
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
 
        # for each distance, calculate the energy loss
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current) 
                
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 


                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1.00E-03:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                        
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
                    
        
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
        
    nuclear_energy_loss_allMat_unshielded[sm+'_electrons'] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_unshielded[sm+'_electrons'] = electronic_energy_loss_thisMat
    currentE_allMat_unshielded[sm+'_electrons'] = currentE_thisMat


# In[182]:



energies_MeV_p = protons_unshielded_bin_energies_all_yesMagnetosphere
energies_MeV_e = electrons_unshielded_bin_energies_all_yesMagnetosphere
 

w = csv.writer(open("electronic_E_loss_vsEin_vsDistance_unshielded.csv", "a"))    
# for each material - protons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_p:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, sum(electronic_energy_loss_allMat_unshielded[sm][E_in][i:(i+10)])])
            
# for each material - electrons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_e:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'e', E_in, i, sum(electronic_energy_loss_allMat_unshielded[sm+'_electrons'][E_in][i:(i+10)])])


w = csv.writer(open("nuclear_E_loss_vsEin_vsDistance_unshielded.csv", "a"))    
# for each material - protons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_p:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, sum(nuclear_energy_loss_allMat_unshielded[sm][E_in][i:(i+10)])])

            
# for each material - electrons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_e:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'e', E_in, i, sum(nuclear_energy_loss_allMat_unshielded[sm+'_electrons'][E_in][i:(i+10)])])

   
w = csv.writer(open("E_vsEin_vsDistance_unshielded.csv", "a"))    
# for each material - protons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_p:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, (0.1)*sum(currentE_allMat_unshielded[sm][E_in][i:(i+10)])])
            
# for each material - electrons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_e:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'e', E_in, i, (0.1)*sum(currentE_allMat_unshielded[sm+'_electrons'][E_in][i:(i+10)])])


# In[183]:


# With aluminum shielding - protons

# Store: per aluminum thickness, per material, per orbit, per energy bin in the spectrum, the nuclear and electronic energy loss as a function of distance into the 2 mm thick material

# NOTE: for all with aluminum shielding, the energy bins are the same
# as a result you only need to store for each material for the given energy bins

# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear[sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear[sm]['Energy_MeV']
                      
    # electronic MeV/cm
    name_e = sm +'_EstoppingPower_MeVcm2g'
    dE_dx_electronic_current = dE_dx_electronic[name_e]*density
    E_MeV_electronic_current = dE_dx_electronic['Energy_MeV']  
        
    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')

    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    # incident energies from all of the orbits without shielding
    energies_MeV = protons_Al_shielded_bin_energies_all_yesMagnetosphere
            
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
 
        # for each distance, calculate the energy loss
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current)  
                
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 


                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1e-06:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                        
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
                           
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
        
    nuclear_energy_loss_allMat_Al_shielded[sm] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_Al_shielded[sm] = electronic_energy_loss_thisMat
    currentE_allMat_Al_shielded[sm] = currentE_thisMat


# In[14]:


# With aluminum shielding - electrons

# Store: per aluminum thickness, per material, per orbit, per energy bin in the spectrum, the nuclear and electronic energy loss as a function of distance into the 2 mm thick material

# NOTE: for all with aluminum shielding, the energy bins are the same
# as a result you only need to store for each material for the given energy bins

# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear_electrons[sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear_electrons[sm]['Energy_MeV']
                      
    # electronic MeV/cm
    name_e = sm +'_Tot_MeVcm2_g'
    dE_dx_electronic_current = dE_dx_electronic_electrons[name_e]*density
    E_MeV_electronic_current = dE_dx_electronic_electrons['Energy_MeV']  
        
    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')

    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    # incident energies from all of the orbits without shielding
    energies_MeV = electrons_Al_shielded_bin_energies_all_yesMagnetosphere
            
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
 

        # for each distance, calculate the energy loss
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current)   
            
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 


                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1.00E-03:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                        
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
                           
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
        
    nuclear_energy_loss_allMat_Al_shielded[sm+'_electrons'] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_Al_shielded[sm+'_electrons'] = electronic_energy_loss_thisMat
    currentE_allMat_Al_shielded[sm+'_electrons'] = currentE_thisMat


# In[41]:


# Without aluminum shielding - FOR GCR Ions



# for each material
for num, sm in enumerate(material_densities):
    density = material_densities[sm]
    
    for ion in ions:
    
                                        
        # nuclear MeV/cm
        dE_dx_nuclear_current = dE_dx_nuclear_ions[ion][sm]['NIEL_MeVcm2_g']*density
        E_MeV_nuclear_current = dE_dx_nuclear_ions[ion][sm]['Energy_MeV']

        # electronic MeV/cm
        dE_dx_electronic_current = dE_dx_electronic_ions[ion][sm]['ELECT_MeVcm2_g']*density
        E_MeV_electronic_current = dE_dx_electronic_ions[ion][sm]['Energy_MeV']


        # build the functions        
        nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic', bounds_error=False, fill_value="extrapolate")

        nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear', bounds_error=False, fill_value="extrapolate")
        electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear', bounds_error=False, fill_value="extrapolate")

        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV = unshielded_ion_galactic_yesMagnetosphere[ion]['LEO_600']['Energy_MeV']

        # for all of the energies, store these values as a function of distance into the material
        nuclear_energy_loss_thisMat = {}
        electronic_energy_loss_thisMat = {}
        currentE_thisMat = {}

        # for each energy in the orbit spectrum, propogate it through
        for E_in in energies_MeV:
            print(E_in, sm)
            # for each energy, we propogate it through and as a function of distance record these things
            E_current = E_in
            nuclear_energy_loss = []
            electronic_energy_loss = []
            current_E = []

            # for each distance, calculate the energy loss, only record every 
            bin_num = 0
            nuclear_1um_accum = 0
            electronic_1um_accum = 0
            currentE_1um_accum = 0

            # for each distance, calculate the energy loss
            for d_cm in distance_array_cm:
                # if you still have energy
                if (E_current > 0):
                    # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                    dE_dx_n = nuclear_stopping_function(E_current)           
                    dE_dx_e = electronic_stopping_function(E_current)   

                    if (dE_dx_n < 0):
                        dE_dx_n = nuclear_stopping_function_lin(E_current)
                        if (dE_dx_n < 0):
                            dE_dx_n = numpy.array(dE_dx_nuclear_current)[-1]
                    if (dE_dx_e < 0):
                        dE_dx_e = electronic_stopping_function_lin(E_current) 
                        if (dE_dx_e < 0):
                            dE_dx_e = numpy.array(dE_dx_electronic_current)[-1]


                    # MeV/cm
                    dE_dx_total = dE_dx_n + dE_dx_e
                    # MeV because distance step is in microns
                    E_current = E_current - dE_dx_total*distance_step_cm

                    dE_n = dE_dx_n*distance_step_cm
                    dE_e = dE_dx_e*distance_step_cm

                    # probably doesn't matter, but if below the range, just add it in                                    
                    if E_current < 1.00E-03:

                        dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                        dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                        E_current = 0

                # if the energy is 0, just don't do anything and put in 0
                else:
                    dE_n = 0
                    dE_e = 0
                    dE_dx_total = 0


                nuclear_1um_accum += dE_n
                electronic_1um_accum += dE_e
                currentE_1um_accum += E_current
                bin_num = bin_num + 1
                if bin_num == 10:
                    bin_num = 0
                    # append the average energy in the 1um bin and the nuclear and electronic energy loss
                    
                    if (nuclear_1um_accum < 0):
                        print(d_cm,  E_current )
                    if (electronic_1um_accum < 0):
                        print(d_cm,  E_current )
                        
                    current_E.append(currentE_1um_accum/10.0)                      
                    nuclear_energy_loss.append(nuclear_1um_accum)
                    electronic_energy_loss.append(electronic_1um_accum)

                    nuclear_1um_accum = 0.0
                    electronic_1um_accum = 0.0
                    currentE_1um_accum = 0.0


            nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
            electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
            currentE_thisMat[E_in] = current_E

        nuclear_energy_loss_allMat_unshielded[sm+'_'+ion] = nuclear_energy_loss_thisMat
        electronic_energy_loss_allMat_unshielded[sm+'_'+ion] = electronic_energy_loss_thisMat
        currentE_allMat_unshielded[sm+'_'+ion] = currentE_thisMat
        
        


# In[36]:


# Without Al shielding for solar He

ion = 'He'
# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear_ions[ion][sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear_ions[ion][sm]['Energy_MeV']

    # electronic MeV/cm
    dE_dx_electronic_current = dE_dx_electronic_ions[ion][sm]['ELECT_MeVcm2_g']*density
    E_MeV_electronic_current = dE_dx_electronic_ions[ion][sm]['Energy_MeV']


    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')

    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    # same for all orbits, this one is arbitrary
    energies_MeV = unshielded_solar_He_yesMagnetosphere['LEO_600']['Energy_MeV']
            
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
 

        # for each distance, calculate the energy loss
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current)  
                
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 


                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1.00E-03:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                        
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
                           
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
        
    nuclear_energy_loss_allMat_unshielded[sm+'_He_solar'] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_unshielded[sm+'_He_solar'] = electronic_energy_loss_thisMat
    currentE_allMat_unshielded[sm+'_He_solar'] = currentE_thisMat
    


# In[37]:


# With Al shielding for solar He

ion = 'He'
# for each material
for num, sm in enumerate(material_densities):
    
    density = material_densities[sm]
                                    
    # nuclear MeV/cm
    dE_dx_nuclear_current = dE_dx_nuclear_ions[ion][sm]['NIEL_MeVcm2_g']*density
    E_MeV_nuclear_current = dE_dx_nuclear_ions[ion][sm]['Energy_MeV']

    # electronic MeV/cm
    dE_dx_electronic_current = dE_dx_electronic_ions[ion][sm]['ELECT_MeVcm2_g']*density
    E_MeV_electronic_current = dE_dx_electronic_ions[ion][sm]['Energy_MeV']


    # build the functions        
    nuclear_stopping_function = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='quadratic')
    electronic_stopping_function = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='quadratic')

    nuclear_stopping_function_lin = interp1d(E_MeV_nuclear_current, dE_dx_nuclear_current, kind='linear')
    electronic_stopping_function_lin = interp1d(E_MeV_electronic_current, dE_dx_electronic_current, kind='linear')

    # same for all orbits, this one is arbitrary
    energies_MeV = Alshielded_solarHe_yesMagnetosphere['LEO_600_0p1']['E_MeV']
    
               
    # for all of the energies, store these values as a function of distance into the material
    nuclear_energy_loss_thisMat = {}
    electronic_energy_loss_thisMat = {}
    currentE_thisMat = {}

    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV:
        print(E_in, sm)
        # for each energy, we propogate it through and as a function of distance record these things
        E_current = E_in
        nuclear_energy_loss = []
        electronic_energy_loss = []
        current_E = []

        # for each distance, calculate the energy loss, only record every 
        bin_num = 0
        nuclear_1um_accum = 0
        electronic_1um_accum = 0
        currentE_1um_accum = 0
 

        # for each distance, calculate the energy loss
        for d_cm in distance_array_cm:
            # if you still have energy
            if (E_current != 0):
                # get the nuclear and electronic stopping powers for this energy (i.e. at this distance)
                dE_dx_n = nuclear_stopping_function(E_current)           
                dE_dx_e = electronic_stopping_function(E_current)  
                
                if (dE_dx_n < 0):
                    dE_dx_n = nuclear_stopping_function_lin(E_current) 
                if (dE_dx_e < 0):
                    dE_dx_e = electronic_stopping_function_lin(E_current) 


                # MeV/cm
                dE_dx_total = dE_dx_n + dE_dx_e
                # MeV because distance step is in microns
                E_current = E_current - dE_dx_total*distance_step_cm
                            
                dE_n = dE_dx_n*distance_step_cm
                dE_e = dE_dx_e*distance_step_cm
                        
                # probably doesn't matter, but if below the range, just add it in                                    
                if E_current < 1.00E-03:
                           
                    dE_n = dE_n + E_current*(dE_dx_n/(dE_dx_n + dE_dx_e))
                    dE_e = dE_e + E_current*(dE_dx_e/(dE_dx_n + dE_dx_e))                           
                    E_current = 0

            # if the energy is 0, just don't do anything and put in 0
            else:
                dE_n = 0
                dE_e = 0
                dE_dx_total = 0
                        
                                    
            nuclear_1um_accum += dE_n
            electronic_1um_accum += dE_e
            currentE_1um_accum += E_current
            bin_num = bin_num + 1
            if bin_num == 10:
                bin_num = 0
                # append the average energy in the 1um bin and the nuclear and electronic energy loss
                current_E.append(currentE_1um_accum/10.0)                      
                nuclear_energy_loss.append(nuclear_1um_accum)
                electronic_energy_loss.append(electronic_1um_accum)
                
                nuclear_1um_accum = 0.0
                electronic_1um_accum = 0.0
                currentE_1um_accum = 0.0
                           
        nuclear_energy_loss_thisMat[E_in] = nuclear_energy_loss
        electronic_energy_loss_thisMat[E_in] = electronic_energy_loss
        currentE_thisMat[E_in] = current_E
        
    nuclear_energy_loss_allMat_Al_shielded[sm+'_He_solar'] = nuclear_energy_loss_thisMat
    electronic_energy_loss_allMat_Al_shielded[sm+'_He_solar'] = electronic_energy_loss_thisMat
    currentE_allMat_Al_shielded[sm+'_He_solar'] = currentE_thisMat
    


# In[184]:



    
# Write unshielded electronic energy loss for Z > 1 ions

w = csv.writer(open("electronic_E_loss_vsEin_vsDistance_unshielded.csv", "a"))    
# Write GCR data as a function of distance - for each material
for num, sm in enumerate(material_densities):
    for ion in ions:
        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV = unshielded_ion_galactic_yesMagnetosphere[ion]['LEO_600']['Energy_MeV']
        for E_in in energies_MeV:
            for i in bins_10micron:
                # for each distance, sum calculate the energy loss
                w.writerow([sm, ion, E_in, i, sum(electronic_energy_loss_allMat_unshielded[sm+'_'+ion][E_in][i:(i+10)])])

# Write solar He data - for each material
for num, sm in enumerate(material_densities):
    for ion in ['He']:
        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV =  unshielded_solar_He_yesMagnetosphere['LEO_600']['Energy_MeV']
        for E_in in energies_MeV:
            for i in bins_10micron:
                # for each distance, sum calculate the energy loss
                w.writerow([sm, ion, E_in, i, sum(electronic_energy_loss_allMat_unshielded[sm+'_He_solar'][E_in][i:(i+10)])])

# Write unshielded nuclear energy loss for Z > 1 ions

w = csv.writer(open("nuclear_E_loss_vsEin_vsDistance_unshielded.csv", "a"))    
# Write GCR data as a function of distance - for each material
for num, sm in enumerate(material_densities):
    for ion in ions:
        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV = unshielded_ion_galactic_yesMagnetosphere[ion]['LEO_600']['Energy_MeV']
        for E_in in energies_MeV:
            for i in bins_10micron:
                # for each distance, sum calculate the energy loss
                w.writerow([sm, ion, E_in, i, sum(nuclear_energy_loss_allMat_unshielded[sm+'_'+ion][E_in][i:(i+10)])])

# Write solar He data - for each material
for num, sm in enumerate(material_densities):
    for ion in ['He']:
        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV =  unshielded_solar_He_yesMagnetosphere['LEO_600']['Energy_MeV']
        for E_in in energies_MeV:
            for i in bins_10micron:
                # for each distance, sum calculate the energy loss
                w.writerow([sm, ion, E_in, i, sum(nuclear_energy_loss_allMat_unshielded[sm+'_He_solar'][E_in][i:(i+10)])])

# Write unshielded energy as a function of distance for Z > 1 ions

w = csv.writer(open("E_vsEin_vsDistance_unshielded.csv", "a"))    
# Write GCR data as a function of distance - for each material
for num, sm in enumerate(material_densities):
    for ion in ions:
        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV = unshielded_ion_galactic_yesMagnetosphere[ion]['LEO_600']['Energy_MeV']
        for E_in in energies_MeV:
            for i in bins_10micron:
                # for each distance, sum calculate the energy loss
                w.writerow([sm, ion, E_in, i, (0.1)*sum(currentE_allMat_unshielded[sm+'_'+ion][E_in][i:(i+10)])])

# Write solar He data - for each material
for num, sm in enumerate(material_densities):
    for ion in ['He']:
        # incident energies from all of the GCR orbits, orbit is arbitrary
        energies_MeV =  unshielded_solar_He_yesMagnetosphere['LEO_600']['Energy_MeV']
        for E_in in energies_MeV:
            for i in bins_10micron:
                # for each distance, sum calculate the energy loss
                w.writerow([sm, ion, E_in, i, (0.1)*sum(currentE_allMat_unshielded[sm+'_He_solar'][E_in][i:(i+10)])])


# In[187]:



energies_MeV_p  = protons_Al_shielded_bin_energies_all_yesMagnetosphere
energies_MeV_e  = electrons_Al_shielded_bin_energies_all_yesMagnetosphere
energies_MeV_solarHe  = Alshielded_solarHe_yesMagnetosphere['LEO_600_0p1']['E_MeV']

    

# Write shielded electronic energy loss for all - NOTE just bins matter so don't need to specify shielding
# all used the same energy bins

w = csv.writer(open("electronic_E_loss_vsEin_vsDistance_Al_shielded.csv", "a"))    
# for each material - protons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_p:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, sum(electronic_energy_loss_allMat_Al_shielded[sm][E_in][i:(i+10)])])
            
# for each material - electrons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_e:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'e', E_in, i, sum(electronic_energy_loss_allMat_Al_shielded[sm+'_electrons'][E_in][i:(i+10)])])

# for each material - solar He
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_solarHe:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'He', E_in, i, sum(electronic_energy_loss_allMat_Al_shielded[sm+'_He_solar'][E_in][i:(i+10)])])



w = csv.writer(open("nuclear_E_loss_vsEin_vsDistance_Al_shielded.csv", "a"))    
# for each material - protons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_p:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, sum(nuclear_energy_loss_allMat_Al_shielded[sm][E_in][i:(i+10)])])

            
# for each material - electrons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_e:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'e', E_in, i, sum(nuclear_energy_loss_allMat_Al_shielded[sm+'_electrons'][E_in][i:(i+10)])])
            
 # for each material - solar He
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_solarHe:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'He', E_in, i, sum(nuclear_energy_loss_allMat_Al_shielded[sm+'_He_solar'][E_in][i:(i+10)])])
           
   
w = csv.writer(open("E_vsEin_vsDistance_Al_shielded.csv", "a"))    
# for each material - protons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_p:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'p', E_in, i, (0.1)*sum(currentE_allMat_Al_shielded[sm][E_in][i:(i+10)])])
            
# for each material - electrons
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_e:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'e', E_in, i, (0.1)*sum(currentE_allMat_Al_shielded[sm+'_electrons'][E_in][i:(i+10)])])
            
            
# for each material - solar He
for num, sm in enumerate(material_densities):    
    # for each energy in the orbit spectrum, propogate it through
    for E_in in energies_MeV_solarHe:
        for i in bins_10micron:
            # for each distance, sum calculate the energy loss
            w.writerow([sm, 'He', E_in, i, (0.1)*sum(currentE_allMat_Al_shielded[sm+'_He_solar'][E_in][i:(i+10)])])


            
            


# In[45]:


# plot bin-wise Edep for GaN

bins_1micron = numpy.arange(0,2000,1)

nuclear_energy_loss_allMat_unshielded 
electronic_energy_loss_allMat_unshielded 

energies_MeV_p = protons_unshielded_bin_energies_all_yesMagnetosphere
energies_MeV_e = electrons_unshielded_bin_energies_all_yesMagnetosphere

energies_MeV = electrons_Al_shielded_bin_energies_all_yesMagnetosphere

energies_MeV = energies_MeV_p[0::6]

# for each material
for num, sm in enumerate(material_densities): 
    if sm == 'GaN':
        matplotlib.pyplot.figure(dpi=200)
        # for each energy in the orbit spectrum, propogate it through
        for num, E_in in enumerate(energies_MeV):
            if num < 10:
                matplotlib.pyplot.semilogy(bins_1micron, electronic_energy_loss_allMat_unshielded[sm][E_in][0:2000])
            else:
                matplotlib.pyplot.semilogy(bins_1micron, electronic_energy_loss_allMat_unshielded[sm][E_in][0:2000], linestyle=styles[1])
        matplotlib.pyplot.title(sm)
        matplotlib.pyplot.xlabel(r'distance [$\mu m$]')
        matplotlib.pyplot.ylabel(r'energy deposited [MeV/$\mu m$]$_e$')
        matplotlib.pyplot.legend(energies_MeV, title='Orbit E Bin (subset) [MeV]', ncol=2, loc='upper center')
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)

        matplotlib.pyplot.show()

# for each material
for num, sm in enumerate(material_densities): 
    if sm == 'GaN':
        matplotlib.pyplot.figure(dpi=200)
        # for each energy in the orbit spectrum, propogate it through
        for num, E_in in enumerate(energies_MeV):
            if num < 10:
                matplotlib.pyplot.semilogy(bins_1micron, nuclear_energy_loss_allMat_unshielded[sm][E_in][0:2000])
            else:
                matplotlib.pyplot.semilogy(bins_1micron, nuclear_energy_loss_allMat_unshielded[sm][E_in][0:2000], linestyle=styles[1])
        matplotlib.pyplot.title(sm)
        matplotlib.pyplot.xlabel(r'distance [$\mu m$]')
        matplotlib.pyplot.ylabel(r'energy deposited [MeV/$\mu m$]$_n$')
        matplotlib.pyplot.legend(energies_MeV, title='Orbit E Bin (subset) [MeV]', ncol=2, loc='upper center')
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)

        matplotlib.pyplot.show()
    


# In[ ]:


#TO DO:
    
#Simulate this in GEANT4 with variable Al shielding - just 1 simulation and put in the given thicknesses as layers


# for each orbit
for orbit_keyvalue, filename in filenames.items():
    # for each energy in the orbit
    file = open(orbit_keyvalue+"_GCRHSpectrum_Unshielded.txt", "w")
    file.write("0.0 0.0\n")
    Energy_MeV = protons_Al_shielded_bin_energies_all_yesMagnetosphere
    Energy_width = numpy.diff(Energy_MeV, prepend=0)
    for num, E_current in enumerate(Energy_MeV):
        orbit_function = interp1d(unshielded_H_galactic_yesMagnetosphere[orbit_keyvalue].Energy_MeV, (1.0/10000.0)*math.pi*unshielded_H_galactic_yesMagnetosphere[orbit_keyvalue].H_DFlux_m2sSrMeV, kind='quadratic', bounds_error = False, fill_value = 0)
        file.write(str(E_current) + " " + str(orbit_function(E_current)*Energy_width[num]) + "\n")
    file.close()

        


# In[ ]:


#TO DO:
    
#Simulate the maximum neutron energy deposition 
#Occurs for: 5 g/cm2 Al LEO polar sun-sync 800km for maximum density material 2 mm thick
#Use max density material, ZnTe: 6.34 g/cm2
file = open("LEO_800_5p0g_cm2_MaxNeutronSpectrum.txt", "w")
keyvalue =  'LEO_800'
keyvalue = keyvalue + '_' + '5p0'
neutron_spectrum = Alshielded_neutrons_yesMagnetosphere[keyvalue]         
for i in numpy.arange(0,len(neutron_spectrum.DFlux_MeVcm2s),1):
    min_energy = (1.0/1000.0)*numpy.array(neutron_spectrum.Energy_keV_low)[i]
    energy_bin_width = (1.0/1000.0)*numpy.array(neutron_spectrum.Energy_keV_high - neutron_spectrum.Energy_keV_low)[i]
    weight = energy_bin_width*(neutron_spectrum.DFlux_MeVcm2s)[i]
    print(min_energy, energy_bin_width, weight)
    file.write(str(min_energy) + " " + str(weight) + "\n")
file.close()
# This is what we compare and see if it is negligble in contribution

