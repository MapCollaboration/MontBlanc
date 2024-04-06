[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6264693.svg)](https://doi.org/10.5281/zenodo.6264693)

![alt text](resources/MontBlanc.jpg "Mont Blanc")

# MontBlanc

`MontBlanc` is a code devoted to the extraction of collinear distributions. So far, it has been used to determine the fragmentation functions (FFs) of the pion from experimental data for single-inclusive annihilation and semi-inclusive deep-inelastic scattering. Details concerning this fit of FFs in particular and the methodology in general can be found in the reference below.

The FF sets in the LHAPDF format for both positive and negative pions as well as for their sum can be found [here](FFSets/).

## Requirements

In order for the code to pe compiled, the following dependencies need to be preinstalled:

- [`NangaParbat`](https://github.com/vbertone/NangaParbat)
- [`apfelxx`](https://github.com/vbertone/apfelxx)
- [`NNAD`](https://github.com/rabah-khalek/NNAD)
- [`ceres-solver`](http://ceres-solver.org)
- [`LHAPDF`](https://lhapdf.hepforge.org)
- [`yaml-cpp`](https://github.com/jbeder/yaml-cpp)
- [`GSL`](https://www.gnu.org/software/gsl/)

## Compilation and installation

The `MontBlanc` library only relies on `cmake` for configuration and installation. This is done by following the standard procedure:
```
mkdir build
cd build
cmake ..
make -j
make install
```
The library can be uninstalled by running:
```
make clean
xargs rm < install_manifest.txt
```

## Usage

The relevant source code to perform a fit and analyse the results can be found in the `run/` folder. However, in the following we assume to be in the `build/run/` folder that will be created after the `cmake` procedure detailed above and that contains the executables. In this folder we need to create a subfolder called `fit/` that will be used to store the results. A short description of each code is as follows:

1. `Optimize`: this code is responsible for performing the fit. An example of the usage of this code is:
    ```
    ./Optimize 1 ../../config/MAPFF10_210301_SIA_SIDIS_7fl_pos_woSLDc.yaml ../../data/ fit/
    ```
    The first argument indicates the Monte Carlo replica index, the second points to the input card containing the main parameters of the fit as well as the data sets to be fitted (see [here](config/MAPFF10_210301_SIA_SIDIS_7fl_pos_woSLDc.yaml) for a commented example and [here](data/README.md) for additional information concerning the data sets), the third points to the folder where the data files are contained, and the last argument is the folder where the results of the fit will be dumped. This code produces in the folder `fit/` a file called `BestParameters.yaml` that contains the best fit parameters of the NN along with some additional information such as the training, validation, and global χ<sup>2</sup>'s. In addition, this code will place in the `fit/` folder two additional subfolders, `log/` and `data/`, containing respectively the log file of the fit and the data files for the fitted experimental sets. If a new fit with a different Monte Carlo replica index is run specifying the `fit/` as a destination for the results, the best fit parameters of this new fit will be appended to the `BestParameters.yaml` file and a new log file will be created in the `fit/log/` subfolder. Notice that Monte Carlo replica indices equal or larger than one correspond to actual random fluctuations of the central values of the experimental data, while the index 0 corresponds to a fit to the central values, i.e. no fluctuations are performed. If the code `Optimize` is run without any arugments it will prompt a short usage description.

2. `LHAPDFGrid`: this code produces an LHAPDF grid for a given fit. In order to produce a grid for the fit in the `fit/` folder, the syntax is:
    ```
    ./LHAPDFGrid fit/
    ```
    The produced grid can be found in the `fit/` folder under the name `LHAPDFSet` and corresponds to positive pion FFs. This set will eventually be used for analysing the results. It possible to customise the output by providing the script with additional options. Specifically, it possible to produce a grid for negative as well as for the sum of positive and negative pion FFs, to change the default name, and to specificy the number of replicas to be produced. The last option is applicable only when more fits have been run in the `fit/` folder and the number of user-provided replicas does not exceed the number of fits. For example, assuming to have performed 120 fits, the following:
    ```
    ./LHAPDFGrid fit/ PIm MySetForPim 100
    ```
    will produce a set named `MySetForPim` for negative pions and with 101 replicas where the zero-replica is the average over the following 100. Similarly, the following:
    ```
    ./LHAPDFGrid fit/ PIp   MySetForPip   100
    ./LHAPDFGrid fit/ PIsum MySetForPisum 100
    ```
    will respectively produce a grid for positive pions (default) and for the sum of positive and negative pions. In addition, the `LHAPDFGrid` code sorts the replicas in the global χ<sup>2</sup> from the smallest to the largest. Therefore, the resulting set will containg the 100 replicas out of 120 with best global χ<sup>2</sup>'s. Also in this case, if the code `LHAPDFGrid` is run without any arugments it will prompt a short usage description.

3. `ComputeChi2s`: as the name says, the code computes the χ<sup>2</sup>'s using the fit results. The syntax is:
    ```
    ./ComputeChi2s fit/
    ```
    This code relies on the presence of an LHAPDF grid in the fit folder named `LHAPDFSet` and will result in the creation of the file `fit/Chi2s.yaml` containing the χ<sup>2</sup> for the single experiments included in the fit. It is also possible to change the name of the FF set to be used to compute the χ<sup>2</sup>'s. For example:
    ```
    ./ComputeChi2s fit/ MySetForPim
    ```
    will compute the χ<sup>2</sup>'s using the `MySetForPim` set that has to be either in the `fit/` folder or in the LHAPDF data directory (that can be retrieved by running the command `lhapdf-config --datadir` from shell).

4. `Predictions`: this code computes the predictions for all the points included in the fit. It is used as:
    ```
    ./Predictions fit/
    ```
    Also this code relies on the presence of an LHAPDF grid in the fit folder named `LHAPDFSet` and will produce the file `fit/Predictions.yaml`. Again, it is possible to use a different name for the FF set to be used to compute the χ<sup>2</sup>'s. For example:
    ```
    ./Predictions fit/ MySetForPim
    ```
    will compute the predictions using the `MySetForPim` set that has to be either in the `fit/` folder or in the LHAPDF data directory.

The results produced by the codes described above can finally be visualised by copying  into the `fit/` folder and running the template `jupyter` notebook [`AnalysePredictions.ipynb`](analysis/AnalysePredictions.ipynb) that is in the `analysis/` folder. This is exactly how the fit of pion FFs documented in the reference below has been obtained and any user should be able to reproduce it by following the steps above. For reference, we have linked the folder of the baseline fit [here](Results/MAPFF10NLOPIp) along with the corresponding `jupyter` [notebook](Results/MAPFF10NLOPIp/AnalysePredictions.ipynb).

## Reference

If you use this code you might want to refer to and cite the following reference:

- Rabah Abdul Khalek, Valerio Bertone, Emanuele R. Nocera, "A determination of unpolarised pion fragmentation functions using semi-inclusive deep-inelastic-scattering data: MAPFF1.0", [arXiv:2105.08725](https://arxiv.org/abs/2105.08725)
- Rabah Abdul Khalek, Valerio Bertone, Alice Khoudli, Emanuele R. Nocera, "Pion and kaon fragmentation functions at next-to-next-to-leading order", [arXiv:2105.08725](https://arxiv.org/abs/2204.10331)

## Contacts

For additional information or questions, contact us using the email adresses below:

- Rabah Abdul Khalek: rabah.khalek@gmail.com
- Valerio Bertone: valerio.bertone@cern.ch
- Emanuele R. Nocera: enocera@ed.ac.uk
