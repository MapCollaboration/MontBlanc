import sys
import pandas as pd
import yaml
import numpy as np

mPR = 0.9382720813 # GeV

# BABAR - conventional inclusive
def filter_BABAR_PR_CONVENTIONAL():
    nameexp = 'BABAR_PR_PLUS_MINUS_CONVENTIONAL'
    infile  = 'BABAR/HEPData-ins1238276-v1-Table_2.csv'
    ndata   = 45
    cme     = 10.54 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=110,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    breakdown = pd.read_csv(
        'BABAR/babarbdPR.data',
        dtype={"user_ld": float},
        header=None,
        sep=',',
        usecols=[0,1,2,3,4,5,6,7,8,9])
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "BABAR coventional $b+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+(breakdown.iloc[i,9]/100.*data.iloc[i,3])**2.)), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,2]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,3]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,4]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,6]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,7]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,8]/100.), file=f)
            print('    - {label: mult, value: 0.0098}', file=f)
            print('    value: %7.5f' % (data.iloc[i,3]), file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# BABAR - prompt inclusive
def filter_BABAR_PR_PROMPT():
    nameexp = 'BABAR_PR_PLUS_MINUS_PROMPT'
    infile  = 'BABAR/HEPData-ins1238276-v1-Table_1.csv'
    ndata   = 45
    cme     = 10.54 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=110,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    breakdown = pd.read_csv(
        'BABAR/babarbdPR.data',
        dtype={"user_ld": float},
        header=None,
        sep=',',
        usecols=[0,1,2,3,4,5,6,7,8,9])
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "BABAR prmpt $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+(breakdown.iloc[i,9]/100.*data.iloc[i,3])**2.)), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,2]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,3]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,4]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,5]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,7]/100.), file=f)
            print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,8]/100.), file=f)
            print('    - {label: mult, value: 0.0098}', file=f)
            print('    value: %7.5f' % (data.iloc[i,3]), file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TASSO12 - inclusive
def filter_TASSO12_PR():
    nameexp = 'TASSO_12_PR_PLUS_MINUS'
    infile  = 'TASSO12/HEPData-ins153656-v1-Table_4.csv'
    ndata   = 3
    cme     = 12.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2],
        skipfooter=14,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO12 $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    - {label: mult, value: 0.20}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# TASSO14 - inclusive
def filter_TASSO14_PR():
    nameexp = 'TASSO_14_PR_PLUS_MINUS'
    infile  = 'TASSO14/HEPData-ins181470-v1-Table_3.csv'
    ndata   = 9
    cme     = 14.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO14 $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    - {label: mult, value: 0.085}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TASSO22 - inclusive
def filter_TASSO22_PR():
    nameexp = 'TASSO_22_PR_PLUS_MINUS'
    infile  = 'TASSO22/HEPData-ins181470-v1-Table_6.csv'
    ndata   = 9
    cme     = 22.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO22 $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    - {label: mult, value: 0.063}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)






# TPC - inclusive
def filter_TPC_PR():
    nameexp = 'TPC_PR_PLUS_MINUS'
    infile  = 'TPC/HEPData-ins262143-v1-Table_1.csv'
    ndata   = 20
    cme     = 29.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=77,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        skipfooter=39,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TPC $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)


            
# TASSO30 - inclusive
def filter_TASSO30_PR():
    nameexp = 'TASSO_30_PR_PLUS_MINUS'
    infile  = 'TASSO30/HEPData-ins153656-v1-Table_7.csv'
    ndata   = 3
    cme     = 30.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2],
        skipfooter=14,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO30 $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    - {label: mult, value: 0.20}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# TASSO34 - inclusive
def filter_TASSO34_PR():
    nameexp = 'TASSO_34_PR_PLUS_MINUS'
    infile  = 'TASSO34/HEPData-ins267755-v1-Table_9.csv'
    ndata   = 11
    cme     = 34.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO34 $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    - {label: mult, value: 0.06}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TASSO44 - inclusive
def filter_TASSO44_PR():
    nameexp = 'TASSO_44_PR_PLUS_MINUS'
    infile  = 'TASSO44/HEPData-ins267755-v1-Table_12.csv'
    ndata   = 4
    cme     = 44.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO44 $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,4]), file=f)
            print('    - {label: mult, value: 0.06}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
      
# ALEPH - inclusive
def filter_ALEPH_PR():
    nameexp = 'ALEPH_PR_PLUS_MINUS'
    infile  = 'ALEPH/HEPData-ins382179-v1-Table_3.csv'
    ndata   = 26
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=11,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "ALEPH $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            print('    - {label: mult, value: 0.05}', file=f)
            print('    value: ', data.iloc[i,3], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - inclusive
def filter_DELPHI_PR():
    nameexp = 'DELPHI_PR_PLUS_MINUS'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_22.csv'
    ndata   = 23
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
        print('- header: {title: "DELPHI $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
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
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - uds tagged
def filter_DELPHI_PR_UDS():
    nameexp = 'DELPHI_PR_PLUS_MINUS_UDS'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_38.csv'
    ndata   = 23
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
        print('- header: {title: "DELPHI $p+\\\\bar{p}$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
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
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# DELPHI - b tagged
def filter_DELPHI_PR_B():
    nameexp = 'DELPHI_PR_PLUS_MINUS_B'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_30.csv'
    ndata   = 23
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
        print('- header: {title: "DELPHI $p+\\\\bar{p}$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
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
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# OPAL - inclusive
def filter_OPAL_PR():
    nameexp = 'OPAL_PR_PLUS_MINUS'
    infile  = 'OPAL/HEPData-ins372772-v1-Table_3.csv'
    ndata   = 37
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
        print('- header: {title: "OPAL $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
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
            factor = np.sqrt(1. + mPR**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], factor * 2./cme * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# SLD - inclusive
def filter_SLD_PR():
    nameexp = 'SLD_PR_PLUS_MINUS'
    infile  = 'SLD/HEPData-ins630327-v1-Table_4.csv'
    ndata   = 36
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=55,
        sep=',',
        header=None,
        skipfooter=1,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $p+\\\\bar{p}$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# SLD - uds tag
def filter_SLD_PR_UDS():
    nameexp = 'SLD_PR_PLUS_MINUS_UDS'
    infile  = 'SLD/HEPData-ins630327-v1-Table_7.csv'
    ndata   = 36
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=15,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=90,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $p+\\\\bar{p}$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# SLD - c tagged
def filter_SLD_PR_C():
    nameexp = 'SLD_PR_PLUS_MINUS_C'
    infile  = 'SLD/HEPData-ins630327-v1-Table_7.csv'
    ndata   = 36
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=59,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=46,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $p+\\\\bar{p}$ Multiplicity c-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [c]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# SLD - b tagged
def filter_SLD_PR_B():
    nameexp = 'SLD_PR_PLUS_MINUS_B'
    infile  = 'SLD/HEPData-ins630327-v1-Table_7.csv'
    ndata   = 36
    cme     = 91.20 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=103,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=2,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $p+\\\\bar{p}$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: PR}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [b]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (data.iloc[i,2]), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mPR**2./cme**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)
    
# Filter data
filter_BABAR_PR_CONVENTIONAL()
filter_BABAR_PR_PROMPT()
filter_TASSO12_PR()
filter_TASSO14_PR()
filter_TASSO22_PR()
filter_TPC_PR()
filter_TASSO30_PR()
filter_TASSO34_PR()
filter_TASSO44_PR()
filter_ALEPH_PR()
filter_DELPHI_PR()
filter_DELPHI_PR_UDS()
filter_DELPHI_PR_B()
filter_OPAL_PR()
filter_SLD_PR()
filter_SLD_PR_UDS()
filter_SLD_PR_C()
filter_SLD_PR_B()
