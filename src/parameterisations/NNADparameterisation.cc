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
              _g(g),
              _NNderivativeSets(_Np + 1, apfel::Set<apfel::Distribution> {apfel::DiagonalBasis{13}, std::map<int, apfel::Distribution>{}})
  {
    this->_pars = _NN->GetParameters();

    // Rotation from physical to QCD-evolution basis
    std::vector<double> R;
    for (int i = 0; i < 13; i++)
      for (int j = 0; j < 13; j++)
        R.push_back(apfel::RotPhysToQCDEvFull[i][j]);

    std::vector<double> FlavourMap = config["flavour map"].as<std::vector<double>>();
    if (13 * _Nout != (int) FlavourMap.size())
      throw std::runtime_error("[NNADparametersiation::Constructor]: FlavourMap doesn't match NNarchitecture.");

    // Combine rotation matrix with the flavour map
    nnad::Matrix<double> FlavourMapT{_Nout, 13, FlavourMap};
    FlavourMapT.Transpose();
    _Rotation = nnad::Matrix<double> {13, 13, R} * FlavourMapT;

    // Fill in grid
    EvaluateOnGrid();
  }

  //_________________________________________________________________________
  void NNADparameterisation::EvaluateOnGrid()
  {
    const std::vector<double> nn1 = _NN->Evaluate({1});
    const std::function<std::vector<double>(double const&)> NormNN = [=] (double const& x) -> std::vector<double>
    {
      // Get NN at x
      std::vector<double> nnx = _NN->Evaluate({x});

      // Subtract the NN at 1 to ensure NN(x = 1) = 0.
      std::transform(nnx.begin(), nnx.end(), nn1.begin(), nnx.begin(), std::minus<double>());

      // If _OutputFunction == 2 square the output vector.
      if (_OutputFunction == 2)
        std::transform(nnx.begin(), nnx.end(), nnx.begin(), nnx.begin(), std::multiplies<double>());

      return nnx;
    };

    const apfel::Set<apfel::Distribution> outputs{apfel::DiagonalBasis{_Nout}, DistributionMap(*_g, NormNN, _Nout)};
    std::map<int, apfel::Distribution> dists;
    for (int i = 0; i < 13; i++)
      dists.emplace(i, outputs.Combine(_Rotation.GetLine(i)));
    _NNderivativeSets[0].SetObjects(dists);
  }

  //_________________________________________________________________________
  void NNADparameterisation::DeriveOnGrid()
  {
    const std::vector<double> dnn1 = _NN->Derive({1});
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

      return dnnx;
    };
    std::map<int, apfel::Distribution> dm = DistributionMap(*_g, NormNN, ( _Np + 1 ) * _Nout);
    for (int ip = 0; ip < _Np + 1; ip++)
      {
        const apfel::Set<apfel::Distribution> outputs{apfel::DiagonalBasis{_Nout}, std::map<int, apfel::Distribution>{std::next(dm.begin(), ip * _Nout), std::next(dm.begin(), ( ip + 1 ) * _Nout)}};
        std::map<int, apfel::Distribution> dists;
        for (int i = 0; i < _Rotation.GetLines(); i++)
          dists.emplace(i, outputs.Combine(_Rotation.GetLine(i)));
        _NNderivativeSets[ip].SetObjects(dists);
      }
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> NNADparameterisation::DistributionFunction() const
  {
    return [=] (double const &) -> apfel::Set<apfel::Distribution> { return _NNderivativeSets[0]; };
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> NNADparameterisation::DistributionDerivative(int ipar) const
  {
    return [=] (double const &) -> apfel::Set<apfel::Distribution> { return _NNderivativeSets[ipar+1]; };
  }
}
