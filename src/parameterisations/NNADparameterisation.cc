//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include <algorithm>
#include "MontBlanc/NNADparameterisation.h"

namespace MontBlanc
{
  //_________________________________________________________________________
  NNADparameterisation::NNADparameterisation(YAML::Node const &config, std::shared_ptr<const apfel::Grid> g):
    NangaParbat::Parameterisation("NNAD", 2, {}, false),
              _NNarchitecture(config["architecture"].as<std::vector<int>>()),
              _NN(new nnad::FeedForwardNN<double>(_NNarchitecture, config["seed"].as<int>(), false)),
              _Nout(_NNarchitecture.back()),
              _Np(_NN->GetParameterNumber()),
              _OutputFunction(config["output function"] ? config["output function"].as<int>() : 1),
              _g(g)
  {
    this->_pars = _NN->GetParameters();

    // Rotation from physical to QCD-evolution basis
    std::vector<double> R(13*13, 0.0);
    for (int i = 0; i < 13; i++)
      for (int j = 0; j < 13; j++)
        R[j + 13*i] = apfel::RotPhysToQCDEvFull[i][j];

    // In principle, NNAD may parameterise more than one hadron
    std::map<std::string, nnad::Matrix<double>> Rotations;
    int TotalSize = 0;
    for (auto &maps : config["flavour maps"])
    {
      std::string hadron = maps["hadron"].as<std::string>();
      std::vector<double> FlavourMap = maps["map"].as<std::vector<double>>();
      _HadronOutputSizeMap[hadron] = (int) FlavourMap.size() / 13;

      // Layer: Rotation into full flavour basis
      nnad::Matrix<double> FlavourMapT{ _HadronOutputSizeMap[hadron], 13, FlavourMap};
      if (config["combine"] ? config["combine"].as<bool>() : false)
        FlavourMapT = FlavourMapT.PseudoInverse_LLR();
      else
        FlavourMapT.Transpose();

      // Layer: Rotation to QCD evolution from full flavour basis
      _Rotations[hadron] = nnad::Matrix<double> {13, 13, R} * FlavourMapT;

      // Prepare the matrix that splits the output of the network
      nnad::Matrix<double> SplitMatrix {_HadronOutputSizeMap[hadron], _Nout, std::vector<double>(_HadronOutputSizeMap[hadron] * _Nout, 0.0)};
      for (int j = 0; j < (int) _HadronOutputSizeMap[hadron]; j++)
            SplitMatrix.SetElement(j, j + TotalSize, 1.0);
      _SplitMatrices[hadron] = SplitMatrix;

      TotalSize += _HadronOutputSizeMap[hadron];
      _NNderivativeSets[hadron] = std::vector<apfel::Set<apfel::Distribution>>(_Np + 1, apfel::Set<apfel::Distribution> {apfel::DiagonalBasis{13}, std::map<int, apfel::Distribution>{}});
    }

    // Check that the size of the maps match the output of the network
    // TODO: Check if this makes sense
    if (_Nout != TotalSize)
        throw std::runtime_error("[NNADparametersiation::Constructor]: FlavourMap doesn't match NNarchitecture.");

    // Fill in grid
    EvaluateOnGrid();
  }

  //_________________________________________________________________________
  void NNADparameterisation::EvaluateOnGrid()
  {
    // Get NN at 1
    const std::vector<double> nn1 = _NN->Evaluate({1});

    // Loop over the hadrons
    for (auto const& RotationMap : _Rotations)
      {
        // Construct the function that will be discretised on the grid
        const std::function<std::vector<double>(double const&)> NormNN = [=] (double const& x) -> std::vector<double>
        {
          // Get NN at x
          std::vector<double> nnx = _NN->Evaluate({x});

          // Subtract the NN at 1 to ensure NN(x = 1) = 0.
          std::transform(nnx.begin(), nnx.end(), nn1.begin(), nnx.begin(), std::minus<double>());

          // If _OutputFunction == 2 square the output vector.
          if (_OutputFunction == 2)
            std::transform(nnx.begin(), nnx.end(), nnx.begin(), nnx.begin(), std::multiplies<double>());

          // Selects the specific hadron output
          nnad::Matrix<double> SplittedOutput = _SplitMatrices.at(RotationMap.first) * nnad::Matrix(nnx.size(), 1, nnx);
          return SplittedOutput.GetVector();
        };

        // Create a apfel::Set of apfe::Distribution which contains all the final hadrons.
        const apfel::Set<apfel::Distribution> outputs{apfel::DiagonalBasis{_HadronOutputSizeMap[RotationMap.first]}, DistributionMap(*_g, NormNN, _HadronOutputSizeMap[RotationMap.first])};

        std::map<int, apfel::Distribution> dists;
        for (int i = 0; i < 13; i++)
          dists.emplace(i, outputs.Combine(RotationMap.second.GetLine(i)));
        _NNderivativeSets.at(RotationMap.first)[0].SetObjects(dists);
      }
      // @note: So far everything runs fine (maybe with the wrong result)
      // Implement sum rules
      // _NNderivativeSets["sum1"][0] = _NNderivativeSets["PI"][0] + _NNderivativeSets["KA"][0];
      // _NNderivativeSets["sum2"][0] = _NNderivativeSets["PI"][0] + _NNderivativeSets["KA"][0];
      // _NNderivativeSets["sum3"][0] = _NNderivativeSets["PI"][0] + _NNderivativeSets["KA"][0];
      // _NNderivativeSets["h+"][0] =  _NNderivativeSets["hres"][0] - _NNderivativeSets["PI"][0] + _NNderivativeSets["KA"][0];
  }

  //_________________________________________________________________________
  void NNADparameterisation::DeriveOnGrid()
  {
    // Get NN at 1
    const std::vector<double> dnn1 = _NN->Derive({1});
    int HadronSizeOffset = 0;

    // Loop over the hadrons
    for (auto const& RotationMap : _Rotations)
      {
        // Construct the function that will be discretised on the grid
        const int HadronOutputSize = _HadronOutputSizeMap[RotationMap.first];
        const std::function<std::vector<double>(double const&)> NormNN = [=] (double const& x) -> std::vector<double>
        {
          // Get NN at x
          std::vector<double> dnnx = _NN->Derive({x});

          // Subtract the NN at 1 to ensure NN(x = 1) = 0.
          std::transform(dnnx.begin(), dnnx.end(), dnn1.begin(), dnnx.begin(), std::minus<double>());

          // If _OutputFunction == 2 square the first _Nout components
          // of dnnx but first multiply the derivative components,
          // i.e. those after the first _Nout, by twice the first _Nout
          // outputs.
          if (_OutputFunction == 2)
            {
              // Multiply derivarives by 2 * NN(x)
              for (int ip = 1; ip <= _Np; ip++)
                std::transform(dnnx.begin() + ip * _Nout, dnnx.begin() + ( ip + 1 ) * _Nout, dnnx.begin(), dnnx.begin() + ip * _Nout,
                [] (double const& a, double const& b) -> double { return 2 * a * b; });

              // Now square the outputs
              std::transform(dnnx.begin(), dnnx.begin() + _Nout, dnnx.begin(), dnnx.begin(), std::multiplies<double>());
            }

          std::vector<double> FilteredOutput (HadronOutputSize * (_Np + 1), 0.0);
          for (int par = 0; par < _Np + 1; par++)
          {
            for (int idx = 0; idx < HadronOutputSize; idx++)       
            {
              FilteredOutput[idx + HadronOutputSize * par] = dnnx[par * _Nout + (idx + HadronSizeOffset)];
            }
          }

          return FilteredOutput;
        };

        std::map<int, apfel::Distribution> dm = DistributionMap(*_g, NormNN, ( _Np + 1 ) * _HadronOutputSizeMap[RotationMap.first]);
        for (int ip = 0; ip < _Np + 1; ip++)
          {
            const apfel::Set<apfel::Distribution> outputs{apfel::DiagonalBasis{_HadronOutputSizeMap[RotationMap.first]}, std::map<int, apfel::Distribution>{std::next(dm.begin(), ip * _HadronOutputSizeMap[RotationMap.first]), std::next(dm.begin(), ( ip + 1 ) * _HadronOutputSizeMap[RotationMap.first])}};
            std::map<int, apfel::Distribution> dists;
            for (int i = 0; i < RotationMap.second.GetLines(); i++)
              dists.emplace(i, outputs.Combine(RotationMap.second.GetLine(i)));
            _NNderivativeSets.at(RotationMap.first)[ip].SetObjects(dists);
          }
        HadronSizeOffset += HadronOutputSize;
      }
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> NNADparameterisation::DistributionFunction(std::string const& hadron) const
  {
    return [=] (double const &) -> apfel::Set<apfel::Distribution> { return _NNderivativeSets.at(hadron)[0]; };
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> NNADparameterisation::DistributionDerivative(int ipar, std::string const& hadron) const
  {
    return [=] (double const &) -> apfel::Set<apfel::Distribution> { return _NNderivativeSets.at(hadron)[ipar+1]; };
  }
}
