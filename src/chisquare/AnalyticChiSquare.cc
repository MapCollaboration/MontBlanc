//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/AnalyticChiSquare.h"

namespace MontBlanc
{
  //--- NangaParbat inherited
  //_________________________________________________________________________
  AnalyticChiSquare::AnalyticChiSquare(std::vector<std::pair<NangaParbat::DataHandler *, NangaParbat::ConvolutionTable *>> DSVect,
                                       NangaParbat::Parameterisation * FFs,
                                       bool NUMERIC):
    NangaParbat::ChiSquare({}, FFs)
  {
    for (auto const &ds : DSVect)
      ds.second->SetInputFFs(FFs->DistributionFunction());

    for (auto const &ds : DSVect)
      AddBlock(ds);

    _Np = FFs->GetParameterNumber();

    if (!NUMERIC)
      {
        set_num_residuals(std::accumulate(_ndata.begin(), _ndata.end(), 0));

        // Set sizes of the parameter blocks. There are as many parameter
        // blocks as free parameters and each block has size 1.
        for (int ip = 0; ip < _Np; ip++)
          mutable_parameter_block_sizes()->push_back(1);
      }
  }

  //_________________________________________________________________________
  AnalyticChiSquare::AnalyticChiSquare(NangaParbat::Parameterisation * FFs):
    AnalyticChiSquare::AnalyticChiSquare({}, FFs)
  {
  }

  //_________________________________________________________________________
  void AnalyticChiSquare::SetParameters(std::vector<double> const &pars)
  {
    _NPFunc->SetParameters(pars);
    _NPFunc->EvaluateOnGrid();
    for (int ids = 0; ids < (int) _ndata.size(); ids++)
      _DSVect[ids].second->SetInputFFs(_NPFunc->DistributionFunction());
  }

  //_________________________________________________________________________________
  void AnalyticChiSquare::AddBlock(std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*> DSBlock)
  {
    // Push "DataHandler-ConvolutionTable" back
    DSBlock.second->SetInputFFs(_NPFunc->DistributionFunction());
    _DSVect.push_back(DSBlock);
    _ndata.push_back(DSBlock.first->GetKinematics().ndata);
    const std::valarray<bool> cm = DSBlock.second->GetCutMask();
    _ndatac.push_back(std::count(std::begin(cm), std::end(cm), true));
  }

  //--- ceres inherited
  //_________________________________________________________________________
  bool AnalyticChiSquare::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
  {
    std::vector<double> pars(_Np);

    for (int i = 0; i < _Np; i++)
      pars[i] = parameters[i][0];

    _NPFunc->SetParameters(pars);

    // Residuals and Jacobian
    if (jacobians != NULL)
      {
        _NPFunc->DeriveOnGrid();
        int acc = 0;
        for (int ids = 0; ids < (int) _ndata.size(); ids++)
          {
            if (ids != 0)
              acc += _ndata[ids-1];

            _DSVect[ids].second->SetInputFFs(_NPFunc->DistributionFunction());
            std::vector<double> resids = GetResiduals(ids);

            for (int id = 0; id < _ndata[ids]; id++)
              residuals[acc + id] = resids[id];

            for (int ip = 0; ip < _Np; ip++)
              {
                if (jacobians[ip] == nullptr)
                  continue;

                _DSVect[ids].second->SetInputFFs(_NPFunc->DistributionDerivative(ip));
                std::vector<double> jacobs = GetResidualDerivatives(ids, 0);

                for (int id = 0; id < _ndata[ids]; id++)
                  jacobians[ip][acc + id] = jacobs[id];
              }
          }
      }
    // Only residuals
    else
      {
        _NPFunc->EvaluateOnGrid();
        int acc = 0;
        for (int ids = 0; ids < (int) _ndata.size(); ids++)
          {
            if (ids != 0)
              acc += _ndata[ids - 1];

            _DSVect[ids].second->SetInputFFs(_NPFunc->DistributionFunction());
            std::vector<double> resids = GetResiduals(ids);

            for (int id = 0; id < _ndata[ids]; id++)
              residuals[acc + id] = resids[id];
          }
      }
    return true;
  }

  // NumericCostFunction
  //_________________________________________________________________________
  bool AnalyticChiSquare::operator()(double const *const *parameters, double *residuals) const
  {
    std::vector<double> pars(_Np);

    for (int i = 0; i < _Np; i++)
      pars[i] = parameters[i][0];

    _NPFunc->SetParameters(pars); // Set parameters of the NN

    _NPFunc->EvaluateOnGrid();

    for (int ids = 0; ids < (int) _ndata.size(); ids++)
      {
        _DSVect[ids].second->SetInputFFs(_NPFunc->DistributionFunction());
        std::vector<double> resids = GetResiduals(ids);

        for (int id = 0; id < (int) _ndata[ids]; id++)
          residuals[ids * _ndata.size() + id] = resids[id];
      }
    return true;
  }

} // namespace MontBlanc
