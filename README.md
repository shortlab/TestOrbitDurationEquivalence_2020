# TestOrbitDurationEquivalence_2020

Description of Files:

1. Eloss_Equivalence_Edep_Step-Final.py - Used to propogate the all orbit and test particles through all materials and to record the nuclear and electronic energy deposited as a function of distance into each material. Output files contain this information and are used as input to Eloss_Equivalence_Minimal-ReadIn_NoLoop.py.
2. Eloss_Equivalence_Minimal-ReadIn_NoLoop.py - Reads in the nuclear and electronic energy deposition as a function of distance and computed the flux weighted sums to determine the nuclear and electronic orbit equivalent duration of tests.
3. Eloss_Equivalence-Plots.py - Produced graphs.

Description of Folders:

1. Orbit_Spectra - Contains the SPENVIS and GEANT4 generated raw and Al shielded spectra from all sources in all orbits.
2. EdepVsDistance - Contains the output of Eloss_Equivalence_Edep_Step-Final.py and input to Eloss_Equivalence_Minimal-ReadIn_NoLoop.py, the nuclear and electronic energy desposited as a function of distance from all particles in all materials.
