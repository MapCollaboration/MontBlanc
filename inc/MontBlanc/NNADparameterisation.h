//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#pragma once

#include "MontBlanc/NNADparameterisation.h"

#include <NNAD/FeedForwardNN.h>
#include <yaml-cpp/yaml.h>
#include <NangaParbat/parameterisation.h>

namespace MontBlanc
{
  class NNADparameterisation: public NangaParbat::Parameterisation
  {
  public:
    /**
     * @brief The "NNADparameterisation" constructor
     */
    NNADparameterisation(YAML::Node const &config, std::shared_ptr<const apfel::Grid> g);

    std::vector<double> GetParameters() const { return _pars; }
    int GetParameterNumber()            const { return _Np; }

    void SetParameters(std::vector<double> const &pars)
    {
      this->_pars = pars;
      _NN->SetParameters(pars);
    };

    void EvaluateOnGrid();
    void DeriveOnGrid();

    std::function<apfel::Set<apfel::Distribution>(double const&)> DistributionFunction() const;
    std::function<apfel::Set<apfel::Distribution>(double const&)> DistributionDerivative(int ipar) const;

  private:
    std::vector<int>                             _NNarchitecture;
    nnad::FeedForwardNN<double>                 *_NN;
    int                                          _Nout;
    int                                          _Np;
    int                                          _OutputFunction;
    std::shared_ptr<const apfel::Grid>           _g;
    nnad::Matrix<double>                         _Rotation;
    std::vector<apfel::Set<apfel::Distribution>> _NNderivativeSets;
  };
}
