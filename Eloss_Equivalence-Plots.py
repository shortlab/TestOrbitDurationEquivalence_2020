#!/usr/bin/env python
# coding: utf-8

# In[5]:


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



# In[6]:


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
      
    
    unshielded_H_galactic_noMagnetosphere[keyvalue] = H_spectrum
    unshielded_He_galactic_noMagnetosphere[keyvalue] = He_spectrum
    unshielded_C_galactic_noMagnetosphere[keyvalue] = C_spectrum
    unshielded_N_galactic_noMagnetosphere[keyvalue] = N_spectrum
    unshielded_O_galactic_noMagnetosphere[keyvalue] = O_spectrum

    unshielded_Ne_galactic_noMagnetosphere[keyvalue] = Ne_spectrum
    unshielded_Mg_galactic_noMagnetosphere[keyvalue] = Mg_spectrum
    unshielded_Si_galactic_noMagnetosphere[keyvalue] = Si_spectrum
    unshielded_Fe_galactic_noMagnetosphere[keyvalue] = Fe_spectrum

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

filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded/WITH_MAGNETOSPHERE_SHIELDING/'
for Al_thickness in Al_shielding_gcm2_names:
    for keyvalue, filename in filenames.items():
        fn = filepath + Al_thickness + '/' + filename
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = pd.read_csv(fn, sep=',', skiprows=13, skipfooter=121, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (proton_spectrum['Energy_keV_high'] - proton_spectrum['Energy_keV_low'])/1000.0
        proton_spectrum['DFlux_MeVcm2s'] = (proton_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_protons_yesMagnetosphere[keyvalue] = proton_spectrum

        electron_spectrum = pd.read_csv(fn, sep=',', skiprows=73, skipfooter=62, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (electron_spectrum['Energy_keV_high'] - electron_spectrum['Energy_keV_low'])/1000.0
        electron_spectrum['DFlux_MeVcm2s'] = (electron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_electrons_yesMagnetosphere[keyvalue] = electron_spectrum
        
        neutron_spectrum = pd.read_csv(fn, sep=',', skiprows=133, skipfooter=1, names=['Energy_keV_low', 'Energy_keV_high', 'Energy_keV_mean', 'Flux_cm2bin', 'Error_Flux_cm2bin'], engine='python')
        E_width = (neutron_spectrum['Energy_keV_high'] - neutron_spectrum['Energy_keV_low'])/1000.0
        neutron_spectrum['DFlux_MeVcm2s'] = (neutron_spectrum.Flux_cm2bin/E_width)/mission_time_s
        Alshielded_neutrons_yesMagnetosphere[keyvalue] = neutron_spectrum
 

Alshielded_protons_noMagnetosphere = {}
Alshielded_electrons_noMagnetosphere = {}
Alshielded_neutrons_noMagnetosphere = {}

filepath = '/Users/jvl2xv/anaconda/AFRL_RV/Test_Orbit_Spectra_Equivalence/Orbital_Spectra/Shielded/WITHOUT_MAGNETOSPHERE_SHIELDING/'
for Al_thickness in Al_shielding_gcm2_names:
    for keyvalue, filename in filenames.items():
        fn = filepath + Al_thickness + '/' + filename
        keyvalue = keyvalue + '_' + Al_thickness

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
        


# In[7]:


# sum up for without Al shielding:

protons_unshielded_bin_energies_all_yesMagnetosphere = numpy.append(numpy.array(unshielded_solar_protons_yesMagnetosphere['LEO_600']['Energy_MeV']), [5.5e+02, 6.0e+02, 6.5e+02, 7.0e+02, 7.5e+02, 8.0e+02, 8.5e+02, 9.0e+02, 9.5e+02, 1.0e+03])
protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s = {}

# for each orbit
for keyvalue, filename in filenames.items():
    this_sum = numpy.zeros(len(protons_unshielded_bin_energies_all_yesMagnetosphere))
    # for each energy in the orbit
    for num, E_current in enumerate(protons_unshielded_bin_energies_all_yesMagnetosphere):
        this_E_sum = 0       
        orbit_function = interp1d(unshielded_trapped_protons[keyvalue].Energy_MeV, unshielded_trapped_protons[keyvalue].DFlux_MeVcm2s, kind='linear', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + orbit_function(E_current)
        orbit_function = interp1d(unshielded_solar_protons_yesMagnetosphere[keyvalue].Energy_MeV, unshielded_solar_protons_yesMagnetosphere[keyvalue].DFlux_MeVcm2s*unshielded_solar_protons_yesMagnetosphere[keyvalue].AttenFactor, kind='linear', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + orbit_function(E_current)
        orbit_function = interp1d(unshielded_H_galactic_yesMagnetosphere[keyvalue].Energy_MeV, (1.0/10000.0)*math.pi*unshielded_H_galactic_yesMagnetosphere[keyvalue].H_DFlux_m2sSrMeV, kind='linear', bounds_error = False, fill_value = 0)
        this_E_sum = this_E_sum + orbit_function(E_current)
        
        this_sum[num] = this_E_sum
        
    protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue] = this_sum

    
electrons_unshielded_bin_energies_all_yesMagnetosphere = numpy.array(unshielded_trapped_electrons['LEO_600']['Energy_MeV'])
electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s = {}
for keyvalue, filename in filenames.items():
    electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue] = numpy.array(unshielded_trapped_electrons[keyvalue].DFlux_MeVcm2s)


# In[8]:


fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,5))
legend_strings = []
ax = fig.add_subplot(1,2,1)
for keyvalue, legvalue in orbits_legend.items():        
    legend_strings.append(legvalue)
    
    proton_spectrum = protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    proton_energies = protons_unshielded_bin_energies_all_yesMagnetosphere
    electron_spectrum = electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    electron_energies = electrons_unshielded_bin_energies_all_yesMagnetosphere
    
    
    matplotlib.pyplot.loglog(proton_energies, proton_spectrum, label=legvalue)
    matplotlib.pyplot.xlabel('Energy [MeV]')
    matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')

    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
    matplotlib.pyplot.legend(legend_strings, fontsize=8)
    matplotlib.pyplot.title(r'Proton Flux [yes magneto-shield, Al [g/cm$^2$] = 0]', size=10)    

ax = fig.add_subplot(1,2,2)
for keyvalue, legvalue in orbits_legend.items():        

    proton_spectrum = protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    proton_energies = protons_unshielded_bin_energies_all_yesMagnetosphere
    electron_spectrum = electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    electron_energies = electrons_unshielded_bin_energies_all_yesMagnetosphere
    
    
    matplotlib.pyplot.loglog(electron_energies, electron_spectrum, label=legvalue)
    matplotlib.pyplot.xlabel('Energy [MeV]')
    matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')

    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
    matplotlib.pyplot.legend(legend_strings, fontsize=8)
    matplotlib.pyplot.title(r'Electron Flux [yes magneto-shield, Al [g/cm$^2$] = 0]', size=10)    

matplotlib.pyplot.show()


# In[22]:


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    proton_spectrum = unshielded_trapped_protons[keyvalue]
    matplotlib.pyplot.loglog(proton_spectrum.Energy_MeV, proton_spectrum.DFlux_MeVcm2s)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Trapped Proton Flux')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    electron_spectrum = unshielded_trapped_electrons[keyvalue]
    matplotlib.pyplot.loglog(electron_spectrum.Energy_MeV, electron_spectrum.DFlux_MeVcm2s)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [e/MeVcm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Trapped Electron Flux')
matplotlib.pyplot.show()

matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    proton_spectrum = unshielded_solar_protons_noMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(proton_spectrum.Energy_MeV, proton_spectrum.DFlux_MeVcm2s)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Solar Proton Flux [no magnetosphere shielding]')
matplotlib.pyplot.show()

matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    proton_spectrum = unshielded_solar_protons_yesMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(proton_spectrum.Energy_MeV, proton_spectrum.DFlux_MeVcm2s)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Solar Proton Flux [yes magnetosphere shielding]')
matplotlib.pyplot.show()

matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    H_spectrum = unshielded_H_galactic_yesMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(H_spectrum.Energy_MeV, H_spectrum.H_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [p/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic H Flux [yes magnetosphere shielding]')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    H_spectrum = unshielded_H_galactic_noMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(H_spectrum.Energy_MeV, H_spectrum.H_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [p/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic H Flux [no magnetosphere shielding]')
matplotlib.pyplot.show()

matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    He_spectrum = unshielded_He_galactic_yesMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(He_spectrum.Energy_MeV, He_spectrum.He_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [atoms/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic He Flux [yes magnetosphere shielding]')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    H_spectrum = unshielded_He_galactic_noMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(He_spectrum.Energy_MeV, He_spectrum.He_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [atoms/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic He Flux [no magnetosphere shielding]')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    C_spectrum = unshielded_C_galactic_yesMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(C_spectrum.Energy_MeV, C_spectrum.C_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [atoms/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic C Flux [yes magnetosphere shielding]')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    C_spectrum = unshielded_C_galactic_noMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(C_spectrum.Energy_MeV, C_spectrum.C_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [atoms/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic C Flux [no magnetosphere shielding]')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    O_spectrum = unshielded_O_galactic_yesMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(O_spectrum.Energy_MeV, O_spectrum.O_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [atoms/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic O Flux [yes magnetosphere shielding]')
matplotlib.pyplot.show()


matplotlib.pyplot.figure(dpi=150)
legend_strings = []
for keyvalue, legvalue in orbits_legend.items():
    O_spectrum = unshielded_O_galactic_noMagnetosphere[keyvalue]
    matplotlib.pyplot.loglog(O_spectrum.Energy_MeV, O_spectrum.O_DFlux_m2sSrMeV)
    legend_strings.append(legvalue)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [atoms/(MeV/z)Srm$^2$s]')
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(legend_strings)
matplotlib.pyplot.title('Galactic O Flux [no magnetosphere shielding]')
matplotlib.pyplot.show()


# Total Solar + Trapped with differing thickness of Al shielding

# nWith magnetosphere shielding too

handles, labels = (0,0)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))

ax = fig.add_subplot(3,2,1)
for keyvalue, legvalue in orbits_legend.items():        
    legend_strings.append(legvalue)
    
    proton_spectrum = protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    proton_energies = protons_unshielded_bin_energies_all_yesMagnetosphere
    electron_spectrum = electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    electron_energies = electrons_unshielded_bin_energies_all_yesMagnetosphere
    
    
    matplotlib.pyplot.loglog(proton_energies, proton_spectrum, label=legvalue)
    matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')

    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
    #matplotlib.pyplot.legend(legend_strings, fontsize=8)
    matplotlib.pyplot.title(r'Proton Flux [yes magneto-shield, Al [g/cm$^2$] = 0]', size=10)  
    matplotlib.pyplot.ylim([10**(-5), 2*10**(8)])
    plt.tick_params(axis='x', labelbottom=False)
    
    
 
  
    
for num, Al_thickness in enumerate(Al_shielding_gcm2_names):
    ax = fig.add_subplot(3,2,num+2)
    print(Al_thickness)
    for keyvalue, legvalue in orbits_legend.items():
        
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = Alshielded_protons_yesMagnetosphere[keyvalue]
        electron_spectrum = Alshielded_electrons_yesMagnetosphere[keyvalue]
        neutron_spectrum = Alshielded_neutrons_yesMagnetosphere[keyvalue]         
        
        matplotlib.pyplot.loglog((0.5/1000.0)*(proton_spectrum.Energy_keV_low + proton_spectrum.Energy_keV_high), proton_spectrum.DFlux_MeVcm2s, label=legvalue)
        
        if (num == 3) or (num ==4):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if num % 2 != 0:
            matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
        else:
            plt.tick_params(axis='y', labelleft=False)
        matplotlib.pyplot.ylim([10**(-5), 2*10**(8)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        if (num + 2) == 6:
            matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(r'Proton Flux [yes magneto-shield, Al [g/cm$^2$] = ' + str(Al_shielding_gcm2[num]) + ']', size=10)    

        proton_energies = (0.5/1000.0)*(proton_spectrum.Energy_keV_low + proton_spectrum.Energy_keV_high)
        proton_flux = proton_spectrum.DFlux_MeVcm2s
        
        print(keyvalue)
        print(sum(proton_flux*proton_energies)/sum(proton_flux))
        
        
#handles, labels = ax.get_legend_handles_labels()
#ax = fig.add_subplot(3,2,num+2)
#ax.set_axis_off()
#legend = ax.legend(handles, labels, loc='center')

matplotlib.pyplot.show()

handles, labels = (0,0)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))

ax = fig.add_subplot(3,2,1)
for keyvalue, legvalue in orbits_legend.items():        
    legend_strings.append(legvalue)
    
    proton_spectrum = protons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    proton_energies = protons_unshielded_bin_energies_all_yesMagnetosphere
    electron_spectrum = electrons_unshielded_all_yesMagnetosphere_DFlux_MeVcm2s[keyvalue]
    electron_energies = electrons_unshielded_bin_energies_all_yesMagnetosphere
    
    
    matplotlib.pyplot.loglog(electron_energies, electron_spectrum, label=legvalue)
    matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')

    matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
    #matplotlib.pyplot.legend(legend_strings, fontsize=8)
    matplotlib.pyplot.title(r'Electron Flux [yes magneto-shield, Al [g/cm$^2$] = 0]', size=10)  
    matplotlib.pyplot.ylim([10**(-6), 5*10**(8)])
    plt.tick_params(axis='x', labelbottom=False)
     
    
for num, Al_thickness in enumerate(Al_shielding_gcm2_names):
    ax = fig.add_subplot(3,2,num+2)
    for keyvalue, legvalue in orbits_legend.items():
        
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = Alshielded_protons_yesMagnetosphere[keyvalue]
        electron_spectrum = Alshielded_electrons_yesMagnetosphere[keyvalue]
        neutron_spectrum = Alshielded_neutrons_yesMagnetosphere[keyvalue]         
        matplotlib.pyplot.loglog((0.5/1000.0)*(electron_spectrum.Energy_keV_low + electron_spectrum.Energy_keV_high), electron_spectrum.DFlux_MeVcm2s, label=legvalue)
        
        if (num == 3) or (num ==4):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if num % 2 != 0:
            matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
        else:
            plt.tick_params(axis='y', labelleft=False)
        matplotlib.pyplot.ylim([10**(-6), 5*10**(8)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        if (num + 2) == 6:
            matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(r'Electron Flux [yes magneto-shield, Al [g/cm$^2$] = ' + str(Al_shielding_gcm2[num]) + ']', size=10)    

#handles, labels = ax.get_legend_handles_labels()
#ax = fig.add_subplot(3,2,num+2)
#ax.set_axis_off()
#legend = ax.legend(handles, labels, loc='center')

matplotlib.pyplot.show()


handles, labels = (0,0)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))
for num, Al_thickness in enumerate(Al_shielding_gcm2_names):
    ax = fig.add_subplot(3,2,num+1)
    for keyvalue, legvalue in orbits_legend.items():
        
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = Alshielded_protons_yesMagnetosphere[keyvalue]
        electron_spectrum = Alshielded_electrons_yesMagnetosphere[keyvalue]
        neutron_spectrum = Alshielded_neutrons_yesMagnetosphere[keyvalue]         
        matplotlib.pyplot.loglog((0.5/1000.0)*(neutron_spectrum.Energy_keV_low + neutron_spectrum.Energy_keV_high), neutron_spectrum.DFlux_MeVcm2s, label=legvalue)
        
        if (num == 3) or (num ==4):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if num % 2 == 0:
            matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
        else:
            plt.tick_params(axis='y', labelleft=False)
        matplotlib.pyplot.ylim([10**(-5), 5*10**(-1)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        #matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(r'Neutron Flux [yes magneto-shield, Al [g/cm$^2$] = ' + str(Al_shielding_gcm2[num]) + ']', size=10)    

handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(3,2,num+2)
ax.set_axis_off()
legend = ax.legend(handles, labels, loc='center')

matplotlib.pyplot.show()




# Total Solar + Trapped with differing thickness of Al shielding

# Without magnetosphere shielding too

handles, labels = (0,0)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))
for num, Al_thickness in enumerate(Al_shielding_gcm2_names):
    ax = fig.add_subplot(3,2,num+1)
    for keyvalue, legvalue in orbits_legend.items():
        
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = Alshielded_protons_noMagnetosphere[keyvalue]
        electron_spectrum = Alshielded_electrons_noMagnetosphere[keyvalue]
        neutron_spectrum = Alshielded_neutrons_noMagnetosphere[keyvalue]         
        matplotlib.pyplot.loglog((0.5/1000.0)*(proton_spectrum.Energy_keV_low + proton_spectrum.Energy_keV_high), proton_spectrum.DFlux_MeVcm2s, label=legvalue)
        
         
        if (num == 3) or (num ==4):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if num % 2 == 0:
            matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
        else:
            plt.tick_params(axis='y', labelleft=False)
        matplotlib.pyplot.ylim([10**(-5), 2*10**(0)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        #matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(r'Proton Flux [no magneto-shield, Al [g/cm$^2$] = ' + str(Al_shielding_gcm2[num]) + ']', size=10)    

handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(3,2,num+2)
ax.set_axis_off()
legend = ax.legend(handles, labels, loc='center')

matplotlib.pyplot.show()


handles, labels = (0,0)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))
for num, Al_thickness in enumerate(Al_shielding_gcm2_names):
    ax = fig.add_subplot(3,2,num+1)
    for keyvalue, legvalue in orbits_legend.items():
        
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = Alshielded_protons_noMagnetosphere[keyvalue]
        electron_spectrum = Alshielded_electrons_noMagnetosphere[keyvalue]
        neutron_spectrum = Alshielded_neutrons_noMagnetosphere[keyvalue]         
        matplotlib.pyplot.loglog((0.5/1000.0)*(electron_spectrum.Energy_keV_low + electron_spectrum.Energy_keV_high), electron_spectrum.DFlux_MeVcm2s, label=legvalue)
        
        if (num == 3) or (num ==4):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if num % 2 == 0:
            matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
        else:
            plt.tick_params(axis='y', labelleft=False)
        matplotlib.pyplot.ylim([10**(-5), 5*10**(2)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        #matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(r'Electron Flux [no magneto-shield, Al [g/cm$^2$] = ' + str(Al_shielding_gcm2[num]) + ']', size=10)    

handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(3,2,num+2)
ax.set_axis_off()
legend = ax.legend(handles, labels, loc='center')

matplotlib.pyplot.show()


handles, labels = (0,0)

fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,10))
for num, Al_thickness in enumerate(Al_shielding_gcm2_names):
    ax = fig.add_subplot(3,2,num+1)
    for keyvalue, legvalue in orbits_legend.items():
        
        keyvalue = keyvalue + '_' + Al_thickness

        proton_spectrum = Alshielded_protons_noMagnetosphere[keyvalue]
        electron_spectrum = Alshielded_electrons_noMagnetosphere[keyvalue]
        neutron_spectrum = Alshielded_neutrons_noMagnetosphere[keyvalue]         
        matplotlib.pyplot.loglog((0.5/1000.0)*(neutron_spectrum.Energy_keV_low + neutron_spectrum.Energy_keV_high), neutron_spectrum.DFlux_MeVcm2s, label=legvalue)
        
        if (num == 3) or (num ==4):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if num % 2 == 0:
            matplotlib.pyplot.ylabel(r'Flux [p/MeVcm$^2$s]')
        else:
            plt.tick_params(axis='y', labelleft=False)
        matplotlib.pyplot.ylim([10**(-5), 2*10**(-1)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        #matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(r'Neutron Flux [no magneto-shield, Al [g/cm$^2$] = ' + str(Al_shielding_gcm2[num]) + ']', size=10)    

handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(3,2,num+2)
ax.set_axis_off()
legend = ax.legend(handles, labels, loc='center')

matplotlib.pyplot.show()


# In[46]:


ions = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
    
    
fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,4.5))
num = 0
for keyvalue, legvalue in orbits_legend.items():
    num = num + 1
    ax = fig.add_subplot(2,4,num)  
    
    print(keyvalue)
        
    matplotlib.pyplot.loglog(unshielded_trapped_protons[keyvalue]['Energy_MeV'], unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s'], label='trapped H')
    matplotlib.pyplot.loglog(unshielded_trapped_electrons[keyvalue]['Energy_MeV'], unshielded_trapped_electrons[keyvalue]['DFlux_MeVcm2s'], label='trapped e')

    matplotlib.pyplot.loglog(unshielded_solar_protons_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar H')
    matplotlib.pyplot.loglog(unshielded_solar_He_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_He_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar He')
    
    if(sum(unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s']) > 0):
        trapped_p_avg =  sum(unshielded_trapped_protons[keyvalue]['Energy_MeV']*unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s'])/sum(unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s'])
        solar_p_avg =  sum(unshielded_solar_protons_yesMagnetosphere[keyvalue]['Energy_MeV']*unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'])/sum(unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'])
        total_p_avg = (trapped_p_avg*sum(unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s']) + solar_p_avg*sum(unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s']))/(sum(unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s']) + sum(unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s']))
        print(keyvalue, total_p_avg)
    for ion in ions:
        matplotlib.pyplot.loglog(unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue]['Energy_MeV'], unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue][ion+'_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi, label='GCR '+ion)
        
        
        if (num == 5) or (num == 6) or (num == 7):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if (num == 1) or (num == 5):
            matplotlib.pyplot.ylabel(r'Flux [MeVcm$^2$s]$^{-1}$')
        else:
            plt.tick_params(axis='y', labelleft=False)
        #matplotlib.pyplot.ylim([10**(-5), 2*10**(-1)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        #matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(legvalue , size=8)  
        
    matplotlib.pyplot.ylim([10**(-13), 10**(9)])


handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(2,4,num+1)
ax.set_axis_off()
legend = ax.legend(handles, labels, loc='center', fontsize=8,ncol = 2)
matplotlib.pyplot.show()


# In[26]:


# Read in the nuclear energy loss

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



# Read in electronic energy loss

# Energy_MeV', 'Si_EstoppingPower_MeVcm2g', 'Si_IonizingDose_MeVg', 'Si_IonizingDose_MeVg.1', 'GaN_EstoppingPower_MeVcm2g','GaN_IonizingDose_MeVg', 'GaN_IonizingDose_MeVg.1','GaP_EstoppingPower_MeVcm2g', 'GaP_IonizingDose_MeVg','GaP_IonizingDose_MeVg.1', 'InP_EstoppingPower_MeVcm2g', 'InP_IonizingDose_MeVg', 'InP_IonizingDose_MeVg.1','InAs_EstoppingPower_MeVcm2g', 'InAs_IonizingDose_MeVg','InAs_IonizingDose_MeVg.1', 'MgO_EstoppingPower_MeVcm2g','MgO_IonizingDose_MeVg', 'MgO_IonizingDose_MeVg.1','ZnO_EstoppingPower_MeVcm2g', 'ZnO_IonizingDose_MeVg','ZnO_IonizingDose_MeVg.1'
fn = "ELECTRONIC_ELOSS/p/SR_NIEL_StoppingPowers_Electronic.csv"
dE_dx_electronic = pd.read_csv(fn)
dE_dx_electronic.dropna(how='all', axis='columns', inplace=True)
dE_dx_electronic.dropna(inplace=True)



# In[33]:


for i in dE_dx_nuclear.keys():
    print(i+', ')


# In[17]:


ions = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
    
    
fig = matplotlib.pyplot.figure(dpi=200, figsize=(10,6))
num = 0
for keyvalue, legvalue in orbits_legend.items():
    num = num + 1
    ax = fig.add_subplot(2,4,num)  
    
    matplotlib.pyplot.loglog(unshielded_solar_protons_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar H')
    matplotlib.pyplot.loglog(unshielded_solar_He_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_He_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar He')
        
    matplotlib.pyplot.loglog(unshielded_solar_protons_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar H')
    matplotlib.pyplot.loglog(unshielded_solar_He_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_He_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar He')

    for ion in ions:
        matplotlib.pyplot.loglog(unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue]['Energy_MeV'], unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue][ion+'_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi, label='GCR '+ion)
        
        
        if (num == 5) or (num == 6) or (num == 7):
            matplotlib.pyplot.xlabel('Energy [MeV]')
        else:
            plt.tick_params(axis='x', labelbottom=False)
        if (num == 1) or (num == 5):
            matplotlib.pyplot.ylabel(r'Flux [MeVcm$^2$s]$^{-1}$')
        else:
            plt.tick_params(axis='y', labelleft=False)
        #matplotlib.pyplot.ylim([10**(-5), 2*10**(-1)])
        matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
        #matplotlib.pyplot.legend(legend_strings, fontsize=8)
        matplotlib.pyplot.title(legvalue , size=8)    

handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(2,4,num+1)
ax.set_axis_off()
legend = ax.legend(handles, labels, loc='center', fontsize=8)

matplotlib.pyplot.show()


# In[34]:


orbits_legend


# In[43]:


ions = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']

leg = ['Trapped H',
       'Trapped e',
    'Solar H',
  'Solar He',
  'GCR H',
  'GCR He',
  'GCR C',
  'GCR N',
  'GCR O',
  'GCR Ne',
  'GCR Mg',
  'GCR Si',
  'GCR Fe']
    
    
matplotlib.pyplot.figure(dpi=200)
num = 0
for keyvalue, legvalue in orbits_legend.items():
    st = '-'
    
    if (keyvalue == 'LEO_ISS') or (keyvalue == 'HEO_IBEX'):
        st = ':'
    if (keyvalue == 'LEO_800') or (keyvalue == 'MEO_GPS'):
        st = '--'
        
    matplotlib.pyplot.loglog(unshielded_trapped_protons[keyvalue]['Energy_MeV'], unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s'], label='trapped H',  color=colors2[0], linestyle=st)
    matplotlib.pyplot.loglog(unshielded_trapped_electrons[keyvalue]['Energy_MeV'], unshielded_trapped_electrons[keyvalue]['DFlux_MeVcm2s'], label='trapped e',  color=colors2[1], linestyle=st)
    
    matplotlib.pyplot.loglog(unshielded_solar_protons_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar H', color=colors2[4], linestyle=st)
    matplotlib.pyplot.loglog(unshielded_solar_He_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_He_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar He', color=colors2[5], linestyle=st)

    for num_ion, ion in enumerate(ions):
        matplotlib.pyplot.loglog(unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue]['Energy_MeV'], unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue][ion+'_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi, label='GCR '+ion, color=colors2[num_ion + 6], linestyle=st)
   
    num = num + 1
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(leg, ncol = 2)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [MeVcm$^2$s]$^{-1}$')
matplotlib.pyplot.show()




leg = ['Trapped H',
       'Trapped e',
    'Solar H',
  'Solar He',
  'GCR H']
    
    
matplotlib.pyplot.figure(dpi=200)
num = 0
for keyvalue, legvalue in orbits_legend.items():
    st = '-'
    
    if (keyvalue == 'LEO_ISS') or (keyvalue == 'HEO_IBEX'):
        st = ':'
    if (keyvalue == 'LEO_800') or (keyvalue == 'MEO_GPS'):
        st = '--'
        
    matplotlib.pyplot.loglog(unshielded_trapped_protons[keyvalue]['Energy_MeV'], unshielded_trapped_protons[keyvalue]['DFlux_MeVcm2s'], label='trapped H',  color=colors2[0], linestyle=st)
    matplotlib.pyplot.loglog(unshielded_trapped_electrons[keyvalue]['Energy_MeV'], unshielded_trapped_electrons[keyvalue]['DFlux_MeVcm2s'], label='trapped e',  color=colors2[1], linestyle=st)
    
    matplotlib.pyplot.loglog(unshielded_solar_protons_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_protons_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar H', color=colors2[4], linestyle=st)
    matplotlib.pyplot.loglog(unshielded_solar_He_yesMagnetosphere[keyvalue]['Energy_MeV'], unshielded_solar_He_yesMagnetosphere[keyvalue]['DFlux_MeVcm2s'], label='Solar He', color=colors2[5], linestyle=st)

    for num_ion, ion in enumerate(ions):
        if ion == 'H':
            matplotlib.pyplot.loglog(unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue]['Energy_MeV'], unshielded_ion_galactic_yesMagnetosphere[ion][keyvalue][ion+'_DFlux_m2sSrMeV']*(1.0/10000.0)*math.pi, label='GCR '+ion, color=colors2[num_ion + 6], linestyle=st)
   
    num = num + 1
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.legend(leg, ncol = 2)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Flux [MeVcm$^2$s]$^{-1}$')
matplotlib.pyplot.show()


# In[41]:


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




# In[42]:


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


# In[43]:


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


# PLOT ALL THE STOPPING POWERS

# In[75]:


particles = numpy.concatenate((['p', 'e'], ions))

fig = matplotlib.pyplot.figure(dpi=200, figsize=(4,7))

ax = fig.add_subplot(2,1,1)
# nuclear
for particle in particles:
    if particle == 'p':
        matplotlib.pyplot.loglog(dE_dx_nuclear['GaN']['Energy_MeV'], dE_dx_nuclear['GaN']['NIEL_MeVcm2_g'])
    elif particle == 'e':
        matplotlib.pyplot.loglog(dE_dx_nuclear_electrons['GaN']['Energy_MeV'], dE_dx_nuclear_electrons['GaN']['NIEL_MeVcm2_g'])
    else:   
        matplotlib.pyplot.loglog(dE_dx_nuclear_ions[particle]['GaN']['Energy_MeV'],   dE_dx_nuclear_ions[particle]['GaN']['NIEL_MeVcm2_g'])
matplotlib.pyplot.legend(particles, fontsize=8, ncol=2)    
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
#matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Stopping Power [MeVcm$^2$/g]$_n$')
matplotlib.pyplot.xlim([10**(-5), 10**5])

ax = fig.add_subplot(2,1,2)
# electronic
for particle in particles:
    if particle == 'p':
        matplotlib.pyplot.loglog(dE_dx_electronic['Energy_MeV'], dE_dx_electronic['GaN_EstoppingPower_MeVcm2g'])
    elif particle == 'e':
        matplotlib.pyplot.loglog(dE_dx_electronic_electrons['Energy_MeV'], dE_dx_electronic_electrons['GaN_Tot_MeVcm2_g'])
    else:   
        matplotlib.pyplot.loglog(dE_dx_electronic_ions[particle]['GaN']['Energy_MeV'],   dE_dx_electronic_ions[particle]['GaN'][ 'ELECT_MeVcm2_g'])
matplotlib.pyplot.legend(particles, fontsize=8, ncol=2)     
matplotlib.pyplot.grid(b=None, which='major', axis='both', color='grey', linestyle='-', linewidth=.2)
matplotlib.pyplot.xlabel('Energy [MeV]')
matplotlib.pyplot.ylabel(r'Stopping Power [MeVcm$^2$/g]$_e$')
matplotlib.pyplot.xlim([10**(-5), 10**5])
matplotlib.pyplot.show()
    


# In[ ]:




