# Data

This folder contains the data used in this project. It consists of SIA and SIDIS
multiplicities for various targets and hadrons measured by the BABAR, BELLE,
COMPASS and  HERMES experiments. The details are as follows.

## BABAR

### Charged pions, kaons and protons/antiprotons

- References: [Phys.Rev.D 88 (2013) 032011](https://doi.org/10.1103/PhysRevD.88.032011); [arXiv:1306.2895](https://arxiv.org/abs/1306.2895); [HepData](https://www.hepdata.net/record/ins1238276).

- Description: Measurement of the inclusive production of PI+-, K+- and PBAR/P
in e+e- annihilations at a centre-of-mass energy of 10.54 GeV, below the threshold for b-bbar pair production.

## BELLE

## Charged pions and kaons

- References: [Phys.Rev.Lett. 111 (2013) 062002](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.062002); [arXiv:1301.6183](https://arxiv.org/abs/1301.6183); [HepData](https://www.hepdata.net/record/ins1216515).

- Description: Precise measurements of inclusive differential cross sections of charged pions and kaons produced  in e+e- collisions at a centre-of-mass energy of 10.52 GeV. The analysis is based on a data sample of  integrated luminosity of 68.0 fb-1 collected from the asymmmetric KEKB collider with beam energies  3.5 on 8 GeV operating 60 MeV below the upsilon(4s) resonance, resulting in 113M q-qbar events. The   cross sections are tabulated as a function of the fractional hadron energy Z. As reported in the paper,  all weakly produced pions and kaons are included in the cross section values, and all decayed pions and   kaons are recovered. An initial state radiation restriction is also part of the data results given    here, as also reported in the PRL paper: only events with less than 0.5% * sqrt(s) ISR energy   loss are accepted in the analysis.

## COMPASS
 
### Charged pions 

- References: [PLB 764 (2017) 001](http://dx.doi.org/10.1016/j.physletb.2016.09.042); [arXiv:1604.02695](http://arxiv.org/abs/arXiv:1604.02695); [HepData](https://www.hepdata.net/record/ins1444985).

- Description: Multiplicities of positively and negatively charged pions from 
semi-inclusive deep-inelastic scattering of muons off an isoscalar target, in 
bins of `x`, `y`, and `z`. Data were obtained by the COMPASS Collaboration 
using a 160 GeV muon beam and an isoscalar target LiD. They cover the 
kinematic domain in the photon virtuality Q^{2}>1 (GeV/c)^{2}, 
0.004<x<0.4, 0.2<z<0.85 and 0.1<y<0.7. Also given are the 
diffractive vector meson correction to the pion count and DIS count, as well as 
the radiative correction factors to the pion count  and DIS count. The 
correction factors were applied to the raw multiplicity to arrive at the final 
multiplicity given in the table multiplicatively.

### Charged kaons 
- References: [PLB 767 (2017) 133](http://dx.doi.org/10.1016/j.physletb.2017.01.053); [arXiv:1608.06760](http://arxiv.org/abs/1608.06760); [HepData](https://www.hepdata.net/record/ins1483098).

- Description: Multiplicities of positively and negatively charged kaons from 
semi-inclusive deep-inelastic scattering of muons off an isoscalar target, in 
bins of `x`, `y`, and `z`. Data were obtained by the COMPASS Collaboration 
using a 160 GeV muon beam and an isoscalar target LiD. They cover the 
kinematic domain in the photon virtuality Q^{2}>1 (GeV/c)^{2}, 
0.004<x<0.4, 0.2<z<0.85 and 0.1<y<0.7. Also given are the 
diffractive vector meson correction to the pion count and DIS count, as well as 
the radiative correction factors to the pion count  and DIS count. The 
correction factors were applied to the raw multiplicity to arrive at the final 
multiplicity given in the table multiplicatively.

### Charged unidentified hadrons
- References: [PLB 764 (2017) 001](http://dx.doi.org/10.1016/j.physletb.2016.09.042); [arXiv:1604.02695](http://arxiv.org/abs/arXiv:1604.02695); [HepData](https://www.hepdata.net/record/ins1444985).

- Description: Multiplicities of positively and negatively charged hadrons from 
semi-inclusive deep-inelastic scattering of muons off an isoscalar target, in 
bins of `x`, `y`, and `z`. Data were obtained by the COMPASS Collaboration 
using a 160 GeV muon beam and an isoscalar target LiD. They cover the 
kinematic domain in the photon virtuality Q^{2}>1 (GeV/c)^{2}, 
0.004<x<0.4, 0.2<z<0.85 and 0.1<y<0.7. Also given are the 
diffractive vector meson correction to the pion count and DIS count, as well as 
the radiative correction factors to the pion count  and DIS count. The 
correction factors were applied to the raw multiplicity to arrive at the final 
multiplicity given in the table multiplicatively.

## HERMES

### Charged pions and kaons

- References: [Phys.Rev. D87 (2013) 074029](https://doi.org/10.1103/PhysRevD.87.074029); [arXiv:](https://arxiv.org/abs/1212.5407); [Hepdata](https://www.hepdata.net/record/ins1208547); [HERMES database](http://hermesmults.appspot.com/).

- Description: Measurements in semi-inclusive DIS are presented for each charge 
state of PI+- and K+- from electron/positron  collisions of energy 27.6 GeV 
with hydrogen and deuterium targets.  The results are presented as functions of 
the kinematic quantities x, Q^2, Z and hadron pT, and with and without 
subtraction of data from exclusive vector meson production. 

- Warnings: Only the 2D (x,z) and (Q2,z) measurements are included in the 
folder, according to the inconsistencies otherwise found In De Florian, Sassot 
and Stratmann studies. In general, only the bins where z>0.2 should be used. 
This excludes the lowest z-bin for all configurations. The bins below z<0.2 are 
included only for technical reason (to control for the model dependence of the 
smearing-unfolding procedure). The data points above z>0.8 should be treated 
with caution, as they lie in the region where the fractional contribution 
from exclusive processes becomes sizable. Due to the unfolding procedure, the 
different bins are statistically correlated. To correctly treat this 
correlation, the statistical covariance matrix should be used. Note that these 
covariance matrices only relate to the statistical uncertainty. These covariance
matrices are not implemented for the time being. The systematic uncertainties 
should be treated as point-to-point uncertainties.  