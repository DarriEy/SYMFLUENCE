"""
Configuration management utilities for HYPE model.

Handles generation of info.txt, filedir.txt, and par.txt files, including
support for parameter substitution during calibration.
"""

# Standard library imports
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party imports
import pandas as pd
import numpy as np


class HYPEConfigManager:
    """
    Manager for HYPE configuration and control files.

    Handles:
    - Writing info.txt with simulation period and model options
    - Writing filedir.txt for path references
    - Writing par.txt with optional parameter overrides
    - Substitution logic for calibration parameters
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        output_path: Path
    ):
        """
        Initialize the HYPE configuration manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            output_path: Path to output HYPE settings directory
        """
        self.config = config
        self.logger = logger
        self.output_path = Path(output_path)

    def write_info_filedir(self, spinup_days: int, results_dir: str) -> None:
        """Write info.txt and filedir.txt."""
        # 1. filedir.txt
        with open(self.output_path / 'filedir.txt', 'w') as f:
            f.write('./')

        # 2. Extract period from Pobs.txt (must be generated first)
        pobs_path = self.output_path / 'Pobs.txt'
        if not pobs_path.exists():
            self.logger.warning(f"Pobs.txt not found at {pobs_path}, using config defaults for period")
            start_date = self.config.get('EXPERIMENT_TIME_START', '2000-01-01').split(' ')[0]
            end_date = self.config.get('EXPERIMENT_TIME_END', '2000-12-31').split(' ')[0]
        else:
            pobs = pd.read_csv(pobs_path, sep='\t', parse_dates=['time'])
            start_date = pobs['time'].iloc[0].date()
            end_date = pobs['time'].iloc[-1].date()

        spinup_date = pd.to_datetime(start_date) + pd.Timedelta(days=spinup_days)
        spinup_date = spinup_date.date()

        # Build info.txt content
        s1 = """!! ----------------------------------------------------------------------------							
!!							
!! HYPE - Model Agnostic Framework
!!							
!! -----------------------------------------------------------------------------							
!! Check Indata during first runs (deactivate after first runs) 
indatacheckonoff 	2						
indatachecklevel	2		
!! -----------------------------------------------------------------------------							
!!
!! -----------------------------------------------------------------------------							
!!						
!! Simulation settings:							
!!							
!! -----------------	 """
        
        df2 = pd.DataFrame([start_date, spinup_date, end_date, results_dir, 'n', 'y'],
                          index=['bdate', 'cdate', 'edate', 'resultdir', 'instate', 'warning'])

        s3 = """readdaily 	y						
submodel 	n						
calibration	n						
readobsid   n							
soilstretch	n						
steplength	1d							
!! -----------------------------------------------------------------------------							
readsfobs	n	!! For observed snowfall fractions in SFobs.txt							
readswobs	n	!! For observed shortwave radiation in SWobs.txt
readuobs	n	!! For observed wind speeds in Uobs.txt
readrhobs	n	!! For observed relative humidity in RHobs.txt					
readtminobs	y	!! For observed min air temperature in TMINobs.txt				
readtmaxobs	y	!! For observed max air temperature in TMAXobs.txt
soiliniwet	n	
usestop84	n					
!! -----------------------------------------------------------------------------							
modeloption snowfallmodel	0						
modeloption snowdensity	0
modeloption snowfalldist	2
modeloption snowheat	0
modeloption snowmeltmodel	0	
modeloption	snowevapmodel	1				
modeloption snowevaporation	1					
modeloption lakeriverice	0									
modeloption deepground	0 	
modeloption glacierini	1
modeloption floodmodel 0
modeloption frozensoil 2
modeloption infiltration 3
modeloption surfacerunoff 0
modeloption petmodel	1
modeloption wetlandmodel 2		
modeloption connectivity 0					
!! ------------------------------------------------------------------------------------							
timeoutput variable cout	evap	snow
timeoutput meanperiod	1
timeoutput decimals	3					
!! ------------------------------------------------------------------------------------							
!! crit 1 criterion	MKG
!! crit 1 cvariable	cout
!! crit 1 rvariable	rout
!! crit 1 weight	1"""

        with open(self.output_path / 'info.txt', 'w') as f:
            f.write(s1 + '\n')
            df2.to_csv(f, sep='\t', index=True, header=False)
            f.write(s3 + '\n')

    def write_par_file(self, params: Optional[Dict[str, Any]] = None, template_file: Optional[Path] = None) -> None:
        """Write par.txt with optional parameter substitution."""
        if template_file and template_file.exists():
            with open(template_file, 'r') as f:
                content = f.read()
        else:
            content = self._get_default_par_content()

        if params:
            for key, value in params.items():
                val_str = "  ".join(map(str, value)) if isinstance(value, (list, np.ndarray)) else str(value)
                content = re.sub(fr'^({key}\s+)[^\!]*', fr'\g<1>{val_str}  ', content, flags=re.MULTILINE)

        with open(self.output_path / 'par.txt', 'w') as f:
            f.write(content)

    def _get_default_par_content(self) -> str:
        """Return the embedded default HYPE parameter file content."""
        return """!!	=======================================================================================================									
!! Parameter file for:										
!! HYPE -- Generated by the Model Agnostic Framework (hypeflow)									
!!	=======================================================================================================									
!!										
!!	------------------------									
!!										
!!	=======================================================================================================									
!!	"SNOW - MELT, ACCUMULATION, AND DISTRIBUTION; sublimation is sorted under Evapotranspiration"									
!!	-----									
!!	"General snow accumulation and melt related parameters (baseline values from SHYPE, unless noted otherwise)"									
ttpi	1.7083	!! width of the temperature interval with mixed precipitation								
sdnsnew	0.13	!! density of fresh snow (kg/dm3)								
snowdensdt	0.0016	!! snow densification parameter								
fsceff	1	!! efficiency of fractional snow cover to reduce melt and evap								
cmrefr	0.2	"!! snow refreeze capacity (fraction of degreeday melt factor) - baseline value from HBV (pers comm Barbro Johansson, but also in publications)"								
!!	-----									
!!	Landuse dependent snow melt parameters									
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
ttmp	 -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    !! Snowmelt threshold temperature (deg), baseline zero for all landuses"				
cmlt	   9.6497	   9.2928	   9.8897	   5.5393	   2.5333   9.6497	   9.2928	   9.8897	   5.5393	   2.5333   9.6497	   9.2928	   9.8897	   5.5393	   2.5333   9.6497	   9.2928	   9.8897	   5.5393	   2.5333	!! Snowmelt degree day coef (mm/deg/timestep)							
!!	-----									
!!	=======================================================================================================									
!!	EVAPOTRANSPIRATION PARAMETERS									
!!	-----									
!!	General evapotranspiration parameters									
lp	    0.6613	!! Threshold for water content reduction of transpiration (fraction of field capacity) - baseline value from SHYPE because its more realistic with a value slightly below field capacity								
epotdist	   4.7088	!! Coefficient in exponential function for potential evapotranspiration's depth dependency - baseline from EHYPE and/or SHYPE (very similar)																					
!!	-----									
!!										
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
cevp	  0.4689	  0.7925	  0.6317	  0.1699	  0.4506    0.4689	  0.7925	  0.6317	  0.1699	  0.4506    0.4689	  0.7925	  0.6317	  0.1699	  0.4506    0.4689	  0.7925	  0.6317	  0.1699	  0.4506
ttrig	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	!! Soil temperature threshold to allow transpiration - disabled if treda is set to zero				
treda	0.84	0.84	0.84	0.84	0.95	0.84	0.84	0.84	0.84	0.95	0.84	0.84	0.84	0.84	0.95	0.84	0.84	0.84	0.84	0.95	"!! Coefficient in soil temperature response function for root water uptake, default value from gren et al, set to zero to disable the function"				
tredb	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	"!! Coefficient in soil temperature response fuction for root water uptake, default value from gren et al"				
fepotsnow	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	!! Fraction of potential evapotranspiration used for snow sublimation				
!!										
!! Frozen soil infiltration parameters										
!! SOIL:	S1	S2								
bfroznsoil  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838								
logsatmp	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15								
bcosby	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669								
!!	=======================================================================================================									
!!	"SOIL/LAND HYDRAULIC RESPONSE PARAMETERS - recession coef., water retention, infiltration, macropore, surface runoff; etc."									
!!	-----									
!!	Soil-class parameters									
!!	S1	S2								
rrcs1   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985	!! recession coefficients uppermost layer (fraction of water content above field capacity/timestep)							
rrcs2   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853	!! recession coefficients bottom layer (fraction of water content above field capacity/timestep)							
rrcs3	    0.0939	!! Recession coefficient (upper layer) slope dependance (fraction/deg)								
sfrost  1   1  1   1  1   1  1   1  1   1  1   1  1   1  1   1  1   1  1   1	!! frost depth parameter (cm/degree Celsius) soil-type dependent							
wcwp    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280	!! Soil water content at wilting point (volume fraction)											
wcfc    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009	!! Field capacity, layerOne (additional to wilting point) (volume fraction)"										
wcep    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165	!! Effective porosity, layerOne (additional to wp and fc) (volume fraction)"							
!!	-----									
!!	Landuse-class parameters	parameters								
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
srrcs   0.0673  0.1012  0.1984  0.0202  0.0202   0.0673  0.1012  0.1984  0.0202  0.0202   0.0673  0.1012  0.1984  0.0202  0.0202   0.0673  0.1012  0.1984  0.0202  0.0202	!! Runoff coefficient for surface runoff from saturated overland flow of uppermost soil layer (fraction/timestep)				
!!	-----									
!!	Regional groundwater outflow									
rcgrw	0	!! recession coefficient for regional groundwater outflow from soil layers								
!!	=======================================================================================================									
!!	SOIL TEMPERATURE AND SOIL FROST DEPT									
!!	-----									
!!	General									
deepmem	1000	!! temperature memory of deep soil (days)								!! temperature memory of deep soil (days)							
!!-----										
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
surfmem 17.8	17.8	17.8	17.8	5.15 17.8	17.8	17.8	17.8	5.15 17.8	17.8	17.8	17.8	5.15 17.8	17.8	17.8	17.8	5.15	!! upper soil layer soil temperature memory (days)				
depthrel	1.1152	1.1152	1.1152	1.1152	2.47    1.1152	1.1152	1.1152	1.1152	2.47	1.1152	1.1152	1.1152	1.1152	2.47	1.1152	1.1152	1.1152	1.1152	2.47	!! depth relation for soil temperature memory (/m)				
frost	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	!! frost depth parameter (cm/degree Celsius) soil-type dependent				
!!	-----									
!!	=======================================================================================================									
!!	LAKE DISCHARGE									
!!	-----									
!!	-----									
!!	"ILAKE and OLAKE REGIONAL PARAMETERS (1 ilakeregions , defined in geodata)"									
!!	ILAKE parameters																	
!! ilRegion	PPR 1									
ilratk  149.9593						
ilratp  4.9537						
illdepth    0.33					
ilicatch    1.0								
!!										
!!	=======================================================================================================									
!!	RIVER ROUTING									
!!	-----									
damp	   0.2719	!! fraction of delay in the watercourse which also causes damping								
rivvel	     9.7605	!! celerity of flood in watercourse (rivvel>0)								
qmean 	200	!! initial value for calculation of mean flow (mm/yr) - can also be given in LakeData								"""
