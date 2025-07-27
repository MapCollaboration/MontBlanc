import sys
import pandas as pd
import yaml
import numpy as np

mKA = 0.493677 # GeV

# BESIII - inclusive KA+
def filter_BESIII_KAp():
    nameexps = ['BESIII_KA_PLUS_2p0000',
                 'BESIII_KA_PLUS_2p2000',
                 'BESIII_KA_PLUS_2p3960',
                 'BESIII_KA_PLUS_2p6444',
                 'BESIII_KA_PLUS_2p9000',
                 'BESIII_KA_PLUS_3p0500',
                 'BESIII_KA_PLUS_3p5000',
                 'BESIII_KA_PLUS_3p6710']
    infile  = 'BESIII/Table_KAp.csv'
    ndata   = [12, 14, 17, 18, 22, 21, 21, 23]
    cme     = [2.0000, 2.2000, 2.3960, 2.6444, 2.9000, 3.0500, 3.5000, 3.6710] #GeV
    zmin    = 0.011 # to be checked
    zmax    = 0.800 # to be checked 

    l = 0
    i = 2
    j = 3
    k = 4
    
    for nameexp in nameexps:
        
        data = pd.read_csv(
            infile,
            dtype={"user_ld": float},
            skiprows=2,
            sep=',',
            header=None,
            usecols=[0,1,i,j,k],
            engine='python')

        data = data.drop(data[data[i] == 0.].index)
        cval = data[i]
        stat = data[j]
        syst = data[k]
        
        with open(nameexp + '.yaml', 'w') as f:
            print('dependent_variables:', file=f)
            print('- header: {title: "BESIII $K^+$ Multiplicity"}', file=f)
            print('  qualifiers:', file=f)
            print('  - {name: process, value: SIA}', file=f)
            print('  - {name: Vs, value: ', cme[l], ', units: GeV}', file=f)
            print('  - {name: prefactor, value: 1}', file=f)
            print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
            print('  - {name: hadron, value: KA}', file=f)
            print('  - {name: charge, value: +1}', file=f)
            print('  values:', file=f)
            
            for p in range(ndata[l]):
                print('  - errors:', file=f)
                print('    - {label: unc, value: %7.5f}' % (np.sqrt(stat.iloc[p]**2. + syst.iloc[p]**2.)), file=f)
                print('    value: %7.5f' % (cval.iloc[p]), file=f)

            print('independent_variables:', file=f)
            print('- header: {name: "z"}', file=f)
            print('  values:', file=f)
            for p in range(ndata[l]):
                factor = np.sqrt(1. + mKA**2./((data.iloc[p,0] + data.iloc[p,1])/2.)**2.)
                print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                      % (2./cme[l] * factor * data.iloc[p,1], 2./cme[l] * factor * data.iloc[p,0], 2./cme[l] * factor * (data.iloc[p,0] + data.iloc[p,1])/2., 2./cme[l]/factor), file=f)
            print('- header: {name: "ph"}', file=f)
            print('  values:', file=f)
            for p in range(ndata[l]):
                print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                      % (data.iloc[p,1], data.iloc[p,0], (data.iloc[p,0]+data.iloc[p,1])/2.) , file=f)
                        
        l = l + 1
        i = i + 3
        j = j + 3
        k = k + 3

# BESIII - inclusive KA-
def filter_BESIII_KAm():
    nameexps = ['BESIII_KA_MINUS_2p0000',
                 'BESIII_KA_MINUS_2p2000',
                 'BESIII_KA_MINUS_2p3960',
                 'BESIII_KA_MINUS_2p6444',
                 'BESIII_KA_MINUS_2p9000',
                 'BESIII_KA_MINUS_3p0500',
                 'BESIII_KA_MINUS_3p5000',
                 'BESIII_KA_MINUS_3p6710']
    infile  = 'BESIII/Table_KAm.csv'
    ndata   = [12, 14, 17, 18, 22, 21, 21, 23]
    cme     = [2.0000, 2.2000, 2.3960, 2.6444, 2.9000, 3.0500, 3.5000, 3.6710] #GeV
    zmin    = 0.011 # to be checked
    zmax    = 0.800 # to be checked 

    l = 0
    i = 2
    j = 3
    k = 4
    
    for nameexp in nameexps:
        
        data = pd.read_csv(
            infile,
            dtype={"user_ld": float},
            skiprows=2,
            sep=',',
            header=None,
            usecols=[0,1,i,j,k],
            engine='python')

        data = data.drop(data[data[i] == 0.].index)
        cval = data[i]
        stat = data[j]
        syst = data[k]
        
        with open(nameexp + '.yaml', 'w') as f:
            print('dependent_variables:', file=f)
            print('- header: {title: "BESIII $K^-$ Multiplicity"}', file=f)
            print('  qualifiers:', file=f)
            print('  - {name: process, value: SIA}', file=f)
            print('  - {name: Vs, value: ', cme[l], ', units: GeV}', file=f)
            print('  - {name: prefactor, value: 1}', file=f)
            print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
            print('  - {name: hadron, value: KA}', file=f)
            print('  - {name: charge, value: -1}', file=f)
            print('  values:', file=f)
            
            for p in range(ndata[l]):
                print('  - errors:', file=f)
                print('    - {label: unc, value: %7.5f}' % (np.sqrt(stat.iloc[p]**2. + syst.iloc[p]**2.)), file=f)
                print('    value: %7.5f' % (cval.iloc[p]), file=f)

            print('independent_variables:', file=f)
            print('- header: {name: "z"}', file=f)
            print('  values:', file=f)
            for p in range(ndata[l]):
                factor = np.sqrt(1. + mKA**2./((data.iloc[p,0] + data.iloc[p,1])/2.)**2.)
                print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                      % (2./cme[l] * factor * data.iloc[p,1], 2./cme[l] * factor * data.iloc[p,0], 2./cme[l] * factor * (data.iloc[p,0] + data.iloc[p,1])/2., 2./cme[l]/factor), file=f)
            print('- header: {name: "ph"}', file=f)
            print('  values:', file=f)
            for p in range(ndata[l]):
                print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                      % (data.iloc[p,1], data.iloc[p,0], (data.iloc[p,0]+data.iloc[p,1])/2.) , file=f)
                        
        l = l + 1
        i = i + 3
        j = j + 3
        k = k + 3
        
# BELLE - inclusive
def filter_BELLE_KA():
    nameexp = 'BELLE_KA_PLUS_MINUS'
    infile  = 'BELLE/HEPData-ins1216515-v1-Table_1.csv'
    ndata   = 78
    cme     = 10.52 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=94,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')

    delta = (data.loc[:,6]+data.loc[:,7])/2.
    DELTA = (data.loc[:,6]-data.loc[:,7])/2.
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "BELLE coventional $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1.96235e+6}  #0.65 * sigmatot(NLO) [fb]', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s,c]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.1f}' % (np.sqrt(data.iloc[i,4]**2. + DELTA.iloc[i]**2. + 2.*delta.iloc[i]**2.)), file=f)
            print('    - {label: mult, value: 0.014}', file=f)
            print('    value: %7.1f' % (data.iloc[i,3]+delta.iloc[i]), file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# BABAR - conventional inclusive
def filter_BABAR_KA_CONVENTIONAL():
    nameexp = 'BABAR_KA_PLUS_MINUS_CONVENTIONAL'
    infile  = 'BABAR/HEPData-ins1238276-v1-Table_2.csv'
    ndata   = 45
    cme     = 10.54 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=61,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        skipfooter=49,
        engine='python')
    
    breakdown = pd.read_csv(
        'BABAR/babarbdKA.data',
        dtype={"user_ld": float},
        header=None,
        sep=',',
        usecols=[0,1,2,3,4,5,6,7,8,9])
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "BABAR coventional $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s,c]}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# BABAR - conventional inclusive (uncorrelated)
def filter_BABAR_KA_CONVENTIONAL_UNCORR():
    nameexp = 'BABAR_KA_PLUS_MINUS_CONVENTIONAL_UNCORR'
    infile  = 'BABAR/HEPData-ins1238276-v1-Table_2.csv'
    ndata   = 45
    cme     = 10.54 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=61,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        skipfooter=49,
        engine='python')
    
    breakdown = pd.read_csv(
        'BABAR/babarbdKA.data',
        dtype={"user_ld": float},
        header=None,
        sep=',',
        usecols=[0,1,2,3,4,5,6,7,8,9])
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "BABAR coventional $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s,c]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,4]**2.+data.iloc[i,6]**2.)), file=f)
            #print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,2]/100.), file=f)
            #print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,3]/100.), file=f)
            #print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,4]/100.), file=f)
            #print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,6]/100.), file=f)
            #print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,7]/100.), file=f)
            #print('    - {label: add, value: %7.5f}' % (breakdown.iloc[i,8]/100.), file=f)
            print('    - {label: mult, value: 0.0098}', file=f)
            print('    value: %7.5f' % (data.iloc[i,3]), file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
          
# BABAR - prompt inclusive
def filter_BABAR_KA_PROMPT():
    nameexp = 'BABAR_KA_PLUS_MINUS_PROMPT'
    infile  = 'BABAR/HEPData-ins1238276-v1-Table_1.csv'
    ndata   = 45
    cme     = 10.54 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=61,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5,6,7],
        skipfooter=49,
        engine='python')
    
    breakdown = pd.read_csv(
        'BABAR/babarbdKA.data',
        dtype={"user_ld": float},
        header=None,
        sep=',',
        usecols=[0,1,2,3,4,5,6,7,8,9])
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "BABAR prompt $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s,c]}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TASSO12 - inclusive
def filter_TASSO12_KA():
    nameexp = 'TASSO_12_KA_PLUS_MINUS'
    infile  = 'TASSO12/HEPData-ins153656-v1-Table_3.csv'
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
        print('- header: {title: "TASSO12 $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# TASSO14 - inclusive
def filter_TASSO14_KA():
    nameexp = 'TASSO_14_KA_PLUS_MINUS'
    infile  = 'TASSO14/HEPData-ins181470-v1-Table_2.csv'
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
        print('- header: {title: "TASSO14 $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ',units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TASSO22 - inclusive
def filter_TASSO22_KA():
    nameexp = 'TASSO_22_KA_PLUS_MINUS'
    infile  = 'TASSO22/HEPData-ins181470-v1-Table_5.csv'
    ndata   = 10
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
        print('- header: {title: "TASSO22 $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TPC - inclusive
def filter_TPC_KA():
    nameexp = 'TPC_KA_PLUS_MINUS'
    infile  = 'TPC/HEPData-ins262143-v1-Table_1.csv'
    ndata   = 21
    cme     = 29.00 #GeV
    zmin    = 0.011
    zmax    = 0.800

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=51,
        sep=',',
        header=None,
        usecols=[0,1,2,3,4,5],
        skipfooter=64,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TPC $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
 
# TASSO30 - inclusive
def filter_TASSO30_KA():
    nameexp = 'TASSO_30_KA_PLUS_MINUS'
    infile  = 'TASSO30/HEPData-ins153656-v1-Table_6.csv'
    ndata   = 5
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
        skipfooter=18,
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "TASSO30 $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# TASSO34 - inclusive
def filter_TASSO34_KA():
    nameexp = 'TASSO_34_KA_PLUS_MINUS'
    infile  = 'TASSO34/HEPData-ins267755-v1-Table_8.csv'
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
        print('- header: {title: "TASSO34 $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# TASSO44 - inclusive
def filter_TASSO44_KA():
    nameexp = 'TASSO_44_KA_PLUS_MINUS'
    infile  = 'TASSO44/HEPData-ins267755-v1-Table_11.csv'
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
        print('- header: {title: "TASSO44 $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# ALEPH - inclusive
def filter_ALEPH_KA():
    nameexp = 'ALEPH_KA_PLUS_MINUS'
    infile  = 'ALEPH/HEPData-ins382179-v1-Table_2.csv'
    ndata   = 29
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
        print('- header: {title: "ALEPH $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - inclusive
def filter_DELPHI_KA():
    nameexp = 'DELPHI_KA_PLUS_MINUS'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_20.csv'
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
        print('- header: {title: "DELPHI $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# DELPHI - uds tagged
def filter_DELPHI_KA_UDS():
    nameexp = 'DELPHI_KA_PLUS_MINUS_UDS'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_36.csv'
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
        print('- header: {title: "DELPHI $K^\\\\pm$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# DELPHI - b tagged
def filter_DELPHI_KA_B():
    nameexp = 'DELPHI_KA_PLUS_MINUS_B'
    infile  = 'DELPHI/HEPData-ins473409-v1-Table_28.csv'
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
        print('- header: {title: "DELPHI $K^\\\\pm$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# OPAL - inclusive
def filter_OPAL_KA():
    nameexp = 'OPAL_KA_PLUS_MINUS'
    infile  = 'OPAL/HEPData-ins372772-v1-Table_2.csv'
    ndata   = 33
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
        print('- header: {title: "OPAL $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + mKA**2./data.iloc[i,0]**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (2./cme * factor * data.iloc[i,2], 2./cme * factor * data.iloc[i,1], 2./cme * factor * data.iloc[i,0], 2./cme/factor), file=f)
        print('- header: {name: "ph"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)

# SLD - inclusive
def filter_SLD_KA():
    nameexp = 'SLD_KA_PLUS_MINUS'
    infile  = 'SLD/HEPData-ins630327-v1-Table_3.csv'
    ndata   = 36
    cme     = 91.28 #GeV
    zmin    = 0.01715
    zmax    = 1.0

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
        print('- header: {title: "SLD $K^\\\\pm$ Multiplicity"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
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
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,2], factor * data.iloc[i,1], factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {high: %7.5f, low: %7.5f, value: %7.5f}'
                  % (data.iloc[i,2], data.iloc[i,1], data.iloc[i,0]) , file=f)
            
# SLD - uds tag
def filter_SLD_KA_UDS():
    nameexp = 'SLD_KA_PLUS_MINUS_UDS'
    infile  = 'SLD/HEPData-ins630327-v1-Table_6.csv'
    ndata   = 36
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
        skipfooter=90,
        engine='python')

    infile_incl = 'SLD/HEPData-ins630327-v1-Table_3.csv'
    data_incl = pd.read_csv(
        infile_incl,
        dtype={"user_ld": float},
        skiprows=55,
        sep=',',
        header=None,
        skipfooter=1,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $K^\\\\pm$ Multiplicity uds-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [u,d,s]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,2]**2.+(data.iloc[i,1]/data_incl.iloc[i,3]*data_incl.iloc[i,6])**2.)), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# SLD - c tagged
def filter_SLD_KA_C():
    nameexp = 'SLD_KA_PLUS_MINUS_C'
    infile  = 'SLD/HEPData-ins630327-v1-Table_6.csv'
    ndata   = 36
    cme     = 91.28#GeV
    zmin    = 0.01715
    zmax    = 1.0

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=59,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=46,
        engine='python')

    infile_incl = 'SLD/HEPData-ins630327-v1-Table_3.csv'
    data_incl = pd.read_csv(
        infile_incl,
        dtype={"user_ld": float},
        skiprows=55,
        sep=',',
        header=None,
        skipfooter=1,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $K^\\\\pm$ Multiplicity c-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [c]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,2]**2.+(data.iloc[i,1]/data_incl.iloc[i,3]*data_incl.iloc[i,6])**2.)), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# SLD - b tagged
def filter_SLD_KA_B():
    nameexp = 'SLD_KA_PLUS_MINUS_B'
    infile  = 'SLD/HEPData-ins630327-v1-Table_6.csv'
    ndata   = 36
    cme     = 91.28 #GeV
    zmin    = 0.01715
    zmax    = 1.0

    data = pd.read_csv(
        infile,
        dtype={"user_ld": float},
        skiprows=103,
        sep=',',
        header=None,
        usecols=[0,1,2,3],
        skipfooter=2,
        engine='python')

    infile_incl = 'SLD/HEPData-ins630327-v1-Table_3.csv'
    data_incl = pd.read_csv(
        infile_incl,
        dtype={"user_ld": float},
        skiprows=55,
        sep=',',
        header=None,
        skipfooter=1,
        usecols=[0,1,2,3,4,5,6,7],
        engine='python')
    
    with open(nameexp + '.yaml', 'w') as f:
        print('dependent_variables:', file=f)
        print('- header: {title: "SLD $K^\\\\pm$ Multiplicity b-tag"}', file=f)
        print('  qualifiers:', file=f)
        print('  - {name: process, value: SIA}', file=f)
        print('  - {name: Vs, value: ', cme, ', units: GeV}', file=f)
        print('  - {name: prefactor, value: 1}', file=f)
        print('  - {name: z, low: ', zmin, ', high: ',zmax, ', integrate: false}', file=f)
        print('  - {name: hadron, value: KA}', file=f)
        print('  - {name: charge, value: 0}', file=f)
        print('  - {name: tagging, value: [b]}', file=f)
        print('  values:', file=f)
        
        for i in range(ndata):
            print('  - errors:', file=f)
            print('    - {label: unc, value: %7.5f}' % (np.sqrt(data.iloc[i,2]**2.+(data.iloc[i,1]/data_incl.iloc[i,3]*data_incl.iloc[i,6])**2.)), file=f)
            print('    - {label: mult, value: 0.01}', file=f)
            print('    value: ', data.iloc[i,1], file=f)
            
        print('independent_variables:', file=f)
        print('- header: {name: "z"}', file=f)
        print('  values:', file=f)

        for i in range(ndata):
            factor = np.sqrt(1. + 4./data.iloc[i,0]**2. * mKA**2./cme**2.)
            print('  - {value: %7.5f, factor: %7.5f}'
                  % (factor * data.iloc[i,0], 1./factor), file=f)
        print('- header: {name: "xp"}', file=f)
        print('  values:', file=f)
        for i in range(ndata):
            print('  - {value: %7.5f}'
                  % (data.iloc[i,0]) , file=f)

# COMPASS - SIDIS, K+
def filter_COMPASS_KA_PLUS():
    nameexp = 'COMPASS_KA_PLUS_DECORR'
    infile  = 'COMPASS/HEPData-ins1483098-v1-Table_1.csv'
    ndata   = 309
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
        print('- header: {title: "COMPASS $K^+$ Multiplicities"}', file=f)
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
        print('  - {name: hadron, value: KA}', file=f)
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

# COMPASS - SIDIS, K-
def filter_COMPASS_KA_MINUS():
    nameexp = 'COMPASS_KA_MINUS_DECORR'
    infile  = 'COMPASS/HEPData-ins1483098-v1-Table_2.csv'
    ndata   = 309
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
        print('  - {name: hadron, value: KA}', file=f)
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

# COMPASS - SIDIS, KA+ 2024
def filter_COMPASS_KA_PLUS_2024():
    nameexp = 'COMPASS_KA_PLUS_2024'
    infile  = 'COMPASS_2024/Table_KA.csv'
    ndata   = 298
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
        print('- header: {title: "COMPASS 2024 $K^+$ Multiplicities"}', file=f)
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
        print('  - {name: hadron, value: KA}', file=f)
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

# COMPASS - SIDIS, K- 2024
def filter_COMPASS_KA_MINUS_2024():
    nameexp = 'COMPASS_KA_MINUS_2024'
    infile  = 'COMPASS_2024/Table_KA.csv'
    ndata   = 298
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
        print('- header: {title: "COMPASS 2024 $K^-$ Multiplicities"}', file=f)
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
        print('  - {name: hadron, value: KA}', file=f)
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
filter_BESIII_KAp()
filter_BESIII_KAm()
filter_BELLE_KA()
filter_BABAR_KA_CONVENTIONAL()
filter_BABAR_KA_CONVENTIONAL_UNCORR()
filter_BABAR_KA_PROMPT()
filter_TASSO12_KA()
filter_TASSO14_KA()
filter_TASSO22_KA()
filter_TPC_KA()
filter_TASSO30_KA()
filter_TASSO34_KA()
filter_TASSO44_KA()
filter_ALEPH_KA()
filter_DELPHI_KA()
filter_DELPHI_KA_UDS()
filter_DELPHI_KA_B()
filter_OPAL_KA()
filter_SLD_KA()
filter_SLD_KA_UDS()
filter_SLD_KA_C()
filter_SLD_KA_B()
filter_COMPASS_KA_PLUS()
filter_COMPASS_KA_MINUS()
filter_COMPASS_KA_PLUS_2024()
filter_COMPASS_KA_MINUS_2024()
