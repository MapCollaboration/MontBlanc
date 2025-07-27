import sys
import pandas as pd
import yaml
import numpy as np

# TASSO14 - inclusive
def filter_TASSO14_HA():
    nameexp = 'TASSO_14_HA_PLUS_MINUS'
    infile  = 'TASSO/HEPData-ins294755-v1-Table_2.csv'
    ndata   = 20
    cme     = 14.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=12,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        skipfooter=76,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO14 $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ',units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# TASSO22 - inclusive
def filter_TASSO22_HA():
    nameexp = 'TASSO_22_HA_PLUS_MINUS'
    infile  = 'TASSO/HEPData-ins294755-v1-Table_2.csv'
    ndata   = 20
    cme     = 22.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=37,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        skipfooter=51,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO22 $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ',units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# TASSO35 - inclusive
def filter_TASSO35_HA():
    nameexp = 'TASSO_35_HA_PLUS_MINUS'
    infile  = 'TASSO/HEPData-ins294755-v1-Table_2.csv'
    ndata   = 20
    cme     = 35.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=62,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        skipfooter=26,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO35 $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ',units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# TASSO44 - inclusive
def filter_TASSO44_HA():
    nameexp = 'TASSO_44_HA_PLUS_MINUS'
    infile  = 'TASSO/HEPData-ins294755-v1-Table_2.csv'
    ndata   = 20
    cme     = 44.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=87,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        skipfooter=1,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO44 $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ',units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)
            
# TPC - inclusive
def filter_TPC_HA():
    nameexp = 'TPC_HA_PLUS_MINUS'
    infile  = 'TPC/HEPData-ins262143-v1-Table_1.csv'
    ndata   = 34
    cme     = 29.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=102,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TPC $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
             
# ALEPH - inclusive
def filter_ALEPH_HA():
    nameexp = 'ALEPH_HA_PLUS_MINUS'
    infile  = 'ALEPH/HEPData-ins398195-v1-Table_1.csv'
    ndata   = 35
    cme     = 91.20 #GeV
    zmin    = 0.01
    zmax    = 0.95

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "ALEPH $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)

# ALEPH - inclusive longitudinal
def filter_ALEPH_HA_L():
    nameexp = 'ALEPH_HA_L_PLUS_MINUS'
    infile  = 'ALEPH/HEPData-ins398195-v1-Table_3.csv'
    ndata   = 21
    cme     = 91.20 #GeV
    zmin    = 0.01
    zmax    = 0.90

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "ALEPH longitudinal $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: observable, value: FL}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)

# DELPHI - inclusive
def filter_DELPHI_HA():
    nameexp = 'DELPHI_HA_PLUS_MINUS'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_16.csv'
    ndata   = 27
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=13,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "DELPHI $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - inclusive longitudinal
def filter_DELPHI_HA_L():
    nameexp = 'DELPHI_HA_L_PLUS_MINUS'
    infile  = 'DELPHI/HEPData-ins448370-v1-Table_2.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.005
    zmax    = 0.900

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=13,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "DELPHI longitudinal $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: observable, value: FL}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# DELPHI - uds tagged
def filter_DELPHI_HA_UDS():
    nameexp = 'DELPHI_HA_PLUS_MINUS_UDS'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_32.csv'
    ndata   = 27
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=13,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "DELPHI $h^\\\\pm$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - uds tagged longitudinal
def filter_DELPHI_HA_L_UDS():
    nameexp = 'DELPHI_HA_L_PLUS_MINUS_UDS'
    infile  = 'DELPHI/HEPData-ins448370-v1-Table_8.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.005
    zmax    = 0.900

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=40,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "DELPHI longitudinal $h^\\\\pm$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: observable, value: FL}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)
       
# DELPHI - b tagged
def filter_DELPHI_HA_B():
    nameexp = 'DELPHI_HA_PLUS_MINUS_B'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_24.csv'
    ndata   = 27
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=13,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "DELPHI $h^\\\\pm$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [b]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - b tagged
def filter_DELPHI_HA_L_B():
    nameexp = 'DELPHI_HA_L_PLUS_MINUS_B'
    infile  = 'DELPHI/HEPData-ins448370-v1-Table_7.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.005
    zmax    = 0.900

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=40,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "DELPHI longitudinal $h^\\\\pm$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: observable, value: FL}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [b]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)
            
# OPAL - inclusive
def filter_OPAL_HA():
    nameexp = 'OPAL_HA_PLUS_MINUS'
    infile  = 'OPAL/HEPData-ins472637-v1-Table_4.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=12,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "OPAL $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# OPAL - inclusive longitudinal
def filter_OPAL_HA_L():
    nameexp = 'OPAL_HA_L_PLUS_MINUS'
    infile  = 'OPAL/HEPData-ins395450-v1-Table_2.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.005
    zmax    = 0.900

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=12,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "OPAL longitudinal $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: observable, value: FL}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# OPAL - uds tagged
def filter_OPAL_HA_UDS():
    nameexp = 'OPAL_HA_PLUS_MINUS_UDS'
    infile  = 'OPAL/HEPData-ins472637-v1-Table_1.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=12,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "OPAL $h^\\\\pm$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# OPAL - c tagged
def filter_OPAL_HA_C():
    nameexp = 'OPAL_HA_PLUS_MINUS_C'
    infile  = 'OPAL/HEPData-ins472637-v1-Table_2.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=12,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "OPAL $h^\\\\pm$ Multiplicity c-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [c]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# OPAL - b tagged
def filter_OPAL_HA_B():
    nameexp = 'OPAL_HA_PLUS_MINUS_B'
    infile  = 'OPAL/HEPData-ins472637-v1-Table_3.csv'
    ndata   = 22
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=12,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "OPAL $h^\\\\pm$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [b]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], factor), file=f)

# SLD - inclusive
def filter_SLD_HA():
    nameexp = 'SLD_HA_PLUS_MINUS'
    infile  = 'SLD/HEPData-ins630327-v1-Table_1.csv'
    ndata   = 40
    cme     = 91.28 #GeV
    zmin    = 0.01715
    zmax    = 1.0

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=13,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8,9],
        engine='python')

    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $h^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.+data.iloc[i,8]**2.)), file=f)
            print('    value: ', data.iloc[i,3], file=f)

        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = 1.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# SLD - uds tag
def filter_SLD_HA_UDS():
    nameexp = 'SLD_HA_PLUS_MINUS_UDS'
    infile  = 'SLD/HEPData-ins630327-v1-Table_8.csv'
    ndata   = 40
    cme     = 91.28 #GeV
    zmin    = 0.01715
    zmax    = 1.0

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=15,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=94,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $h^\\\\pm$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = 1
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)

# SLD - c tag
def filter_SLD_HA_C():
    nameexp = 'SLD_HA_PLUS_MINUS_C'
    infile  = 'SLD/HEPData-ins630327-v1-Table_8.csv'
    ndata   = 40
    cme     = 91.28 #GeV
    zmin    = 0.01715
    zmax    = 1.0

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=62,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=47,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $h^\\\\pm$ Multiplicity c-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [c]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = 1
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)

# SLD - b tag
def filter_SLD_HA_B():
    nameexp = 'SLD_HA_PLUS_MINUS_B'
    infile  = 'SLD/HEPData-ins630327-v1-Table_8.csv'
    ndata   = 40
    cme     = 91.28 #GeV
    zmin    = 0.01715
    zmax    = 1.0

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=109,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $h^\\\\pm$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [b]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = 1
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)           

# COMPASS - SIDIS, h+
def filter_COMPASS_HA_PLUS():
    nameexp = 'COMPASS_HA_PLUS_DECORR'
    infile  = 'COMPASS/HEPData-ins1444985-v1-Table_3.csv'
    ndata   = 311
    cme     = 17.34 #GeV

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=14,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "COMPASS $h^+$ Multiplicities"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIDIS}', file=f)
        print('  - {name: observable, value: dsigma/dxdydz}', file=f)
        print('  - {name: target_isoscalarity, value: 0.5}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: Q, low: 1, high: 7.745966692414834, integrate: true}', file=f)
        print('  - {name: x, low: 0.004, high: 0.4, integrate: true}', file=f)
        print('  - {name: z, low: 0.2, high: 0.85, integrate: true}', file=f)
        print('  - {name: PS_reduction, W: 5, ymin: 0.1, ymax: 0.7}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 1}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,11]**2+(0.6*data.iloc[i,13])**2)), file=f)
            print('    - {label: add, value: %7.5f}' % (0.8*data.iloc[i,13]/data.iloc[i,10]), file=f)
            print('    value: ', data.iloc[i,10], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,9], data.iloc[i,8], data.iloc[i,7]), file=f)
        print('- header: {name: "x"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
        print('- header: {name: "y"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,5], data.iloc[i,4], data.iloc[i,3]) , file=f)
        print('- header: {name: "Q2"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,6], data.iloc[i,6], data.iloc[i,6]) , file=f)

# COMPASS - SIDIS, h-
def filter_COMPASS_HA_MINUS():
    nameexp = 'COMPASS_HA_MINUS_DECORR'
    infile  = 'COMPASS/HEPData-ins1444985-v1-Table_4.csv'
    ndata   = 311
    cme     = 17.34 #GeV

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=14,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "COMPASS $K^-$ Multiplicities"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIDIS}', file=f)
        print('  - {name: observable, value: dsigma/dxdydz}', file=f)
        print('  - {name: target_isoscalarity, value: 0.5}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: Q, low: 1, high: 7.745966692414834, integrate: true}', file=f)
        print('  - {name: x, low: 0.004, high: 0.4, integrate: true}', file=f)
        print('  - {name: z, low: 0.2, high: 0.85, integrate: true}', file=f)
        print('  - {name: PS_reduction, W: 5, ymin: 0.1, ymax: 0.7}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: -1}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,11]**2+(0.6*data.iloc[i,13])**2)), file=f)
            print('    - {label: add, value: %7.5f}' % (0.8*data.iloc[i,13]/data.iloc[i,10]), file=f)
            print('    value: ', data.iloc[i,10], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,9], data.iloc[i,8], data.iloc[i,7]), file=f)
        print('- header: {name: "x"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
        print('- header: {name: "y"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,5], data.iloc[i,4], data.iloc[i,3]) , file=f)
        print('- header: {name: "Q2"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,6], data.iloc[i,6], data.iloc[i,6]) , file=f)

# COMPASS - SIDIS, h+ 2024
def filter_COMPASS_HA_PLUS_2024():
    nameexp = 'COMPASS_HA_PLUS_2024'
    infile  = 'COMPASS_2024/Table_HA.csv'
    ndata   = 302
    cme     = 17.34 #GeV

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=1,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "COMPASS 2024 $h^+$ Multiplicities"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIDIS}', file=f)
        print('  - {name: observable, value: dsigma/dxdydz}', file=f)
        print('  - {name: target_isoscalarity, value: 1.0}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: Q, low: 1, high: 7.745966692414834, integrate: true}', file=f)
        print('  - {name: x, low: 0.004, high: 0.4, integrate: true}', file=f)
        print('  - {name: z, low: 0.2, high: 0.85, integrate: true}', file=f)
        print('  - {name: PS_reduction, W: 5, ymin: 0.1, ymax: 0.7}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: 1}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,7]**2+(0.6*data.iloc[i,8])**2)), file=f)
            print('    - {label: add, value: %7.5f}' % (0.8*data.iloc[i,8]/data.iloc[i,6]), file=f)
            print('    value: ', data.iloc[i,6], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,5], data.iloc[i,4], ((data.iloc[i,5]+data.iloc[i,4])/2.)), file=f)
        print('- header: {name: "x"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,1], data.iloc[i,0], ((data.iloc[i,1]+data.iloc[i,0])/2.)) , file=f)
        print('- header: {name: "y"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,3], data.iloc[i,2], ((data.iloc[i,3]+data.iloc[i,2])/2.)) , file=f)
        print('- header: {name: "Q2"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            Q2av = (data.iloc[i,1]+data.iloc[i,0])/2.*(data.iloc[i,3]+data.iloc[i,2])/2.*cme**2.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (Q2av, Q2av, Q2av) , file=f)

# COMPASS - SIDIS, h- 2024
def filter_COMPASS_HA_MINUS_2024():
    nameexp = 'COMPASS_HA_MINUS_2024'
    infile  = 'COMPASS_2024/Table_HA.csv'
    ndata   = 302
    cme     = 17.34 #GeV

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=1,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "COMPASS 2024 $h^-$ Multiplicities"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIDIS}', file=f)
        print('  - {name: observable, value: dsigma/dxdydz}', file=f)
        print('  - {name: target_isoscalarity, value: 1.0}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: Q, low: 1, high: 7.745966692414834, integrate: true}', file=f)
        print('  - {name: x, low: 0.004, high: 0.4, integrate: true}', file=f)
        print('  - {name: z, low: 0.2, high: 0.85, integrate: true}', file=f)
        print('  - {name: PS_reduction, W: 5, ymin: 0.1, ymax: 0.7}', file=f)
        print('  - {name: hadron, value: HA}', file=f)
        print('  - {name: charge, value: -1}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,10]**2+(0.6*data.iloc[i,11])**2)), file=f)
            print('    - {label: add, value: %7.5f}' % (0.8*data.iloc[i,11]/data.iloc[i,9]), file=f)
            print('    value: ', data.iloc[i,9], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,5], data.iloc[i,4], ((data.iloc[i,5]+data.iloc[i,4])/2.)), file=f)
        print('- header: {name: "x"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,1], data.iloc[i,0], ((data.iloc[i,1]+data.iloc[i,0])/2.)) , file=f)
        print('- header: {name: "y"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,3], data.iloc[i,2], ((data.iloc[i,3]+data.iloc[i,2])/2.)) , file=f)
        print('- header: {name: "Q2"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            Q2av = (data.iloc[i,1]+data.iloc[i,0])/2.*(data.iloc[i,3]+data.iloc[i,2])/2.*cme**2.
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (Q2av, Q2av, Q2av) , file=f)
            
# Filter data
filter_TASSO14_HA()
filter_TASSO22_HA()
filter_TPC_HA()
filter_TASSO35_HA()
filter_TASSO44_HA()
filter_ALEPH_HA()
filter_ALEPH_HA_L()
filter_DELPHI_HA()
filter_DELPHI_HA_L()
filter_DELPHI_HA_UDS()
filter_DELPHI_HA_L_UDS()
filter_DELPHI_HA_B()
filter_DELPHI_HA_L_B()
filter_OPAL_HA()
filter_OPAL_HA_UDS()
filter_OPAL_HA_C()
filter_OPAL_HA_B()
filter_OPAL_HA_L()
filter_SLD_HA()
filter_SLD_HA_UDS()
filter_SLD_HA_C()
filter_SLD_HA_B()
filter_COMPASS_HA_PLUS()
filter_COMPASS_HA_MINUS()
filter_COMPASS_HA_PLUS_2024()
filter_COMPASS_HA_MINUS_2024()
