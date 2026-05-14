# Tasks

## Active
- Use Claude Cowork to validate data
## Waiting On

## Someday

## Done
- Add measured values to plot of Ca2+ concentration (Fig. 3) and plot Phreeqc (not GWB) SI values. Note: Phreeqc values already placed in input file 
- State the assumptions made in the model
- Create a new Python-PHREEQC script to calculate saturation indices and make a time series plot of all series
- Add calcite kinetics to diel model? Model fits calcite SI values except for summer. Allowing calcite precipitation during the day and dissolution at night would not improve the fit. At night measured SI "C:\Users\ayersj\Python\DielLakeCarbon\output\Stephens_Lake_Summer_diel_SI.csv" << model SI "C:\Users\ayersj\Python\DielLakeCarbon\output\summer\output_v9\diel_v9_results.csv"and calcite is undersaturated, so allowing calcite to dissolve in model would increase SI when it needs to decrease. During the day model values are much closer to measured. So why would measured Sic be different from model only on summer nights (actually in morning before sun rises)? 
- Write Discussion section
- Have Claude add missing helper functions for Sc_O2, Sc_CO2 and henry_CO2
- add code to plot sunlight intensity over time.  
- Add calcite dissolution and precipitation to the model, and have it plot the saturation index of calcite overtime
- Set LATITUDE = 35.939311, LONGITUDE = -87.015833
- Have it fit the GPP and ER data also? Yes, it is scaling them, when I want to work with the real values.
- Add Na and Cl for charge balance to input files. 
- Run one sample in Spec8 to make sure there is adequate information
- Try running the Phreeqc script to make sure it works.
- Rerun seasonal models and paste output into Excel summary file
- Check output against validation values
- Have Claude make a list defining each model variable and its units and specifying whether it is input or model.
- V9: Set the model objective to minimize errors (model - measured) in ER in addition to d13C-DIC and GPP.
- Have Claude Code create an R notebook to calculate Descriptive statistics and make time series plots. 
- Have Claude Chat rewrite the Results and Discussion sections by integrating all four seasons; supply csv file Diel_v9_results_allseasons.csv 
- Have Claude create a schematic diagram in svg format that illustrates the natural processes in the model and how they affect CO2 concentrations