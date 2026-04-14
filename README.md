"# Rotor-Surrogate" 
/help => Includes the functions for post processig and running simulations + time averaging:
  - Forces_PostProc.jl => Fuctions for post processing and calculationg forces on rotors.
  - MeanFlow_PostProc.jl => Functions for time averaging + rinning/continuing simulations.
  - ThreeD_Plots.jl => Functions for generating 3D visualizations from saved simulation files
  - TwoD_Plots.jl => Functions for generating 2D visualizations from saved simulation files
    
/ml => Includes the functions for POD and ML + scripts to run & save:
  - Bayesian_Regression.py => Core functions to run BR from saved POD files.
  - ML.jl => Core functions for training machine learning from saved POD files.
  - POD.jl => Core functions for carrying out POD analysis from saved simulation files.
  - Run_PostProc_ML.jl => Main script to run POD and post-process results.
  - Run_PostProc_POD.jl => Main script to train/test ML models and evaluate performance.
  - rec_error.jl => Script to calculate flow field errors between recovered and ground truth values.
  - recover_forces.jl => Script to reconstruct forces from predicted flow fields.
  - recover_pressure.jl => Reconstructs pressure from velocity fields (Note: Direct training on pressure was found to be more efficient/accurate).
    
/setup => Simulation setups for WaterLily.jl
  - 2D.jl => 2D setups
  - 3D.jl => 3D setups
    
/sims => Scripts for running, time-averaging and post processing all tandem and single rotor simulations.
  - ThreeD_Rotor.jl => Main script for tandem rotor simulations.
  - ThreeD_Rotor_Single.jl => Main script for single rotor simulations (incl. validation cases)
    
/test => Quick tests for new sim setups
  - VTK.jl => Functions for exporting initial setups to VTK format for inspection in ParaView.
  - forces_pressure.jl => Functions for checking force and pressure values and pressure solver at the start of the simulation.
  - sdf.jl => Functions for checking the SDF field.
  - tests.jl => Main script for quick checks.
