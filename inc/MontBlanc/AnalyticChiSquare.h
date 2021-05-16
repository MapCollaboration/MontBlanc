//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#pragma once

#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/predictionshandler.h"

#include <NangaParbat/parameterisation.h>
#include <NangaParbat/chisquare.h>
#include <NangaParbat/linearsystems.h>

#include <ceres/ceres.h>

namespace MontBlanc
{
  class AnalyticChiSquare : public NangaParbat::ChiSquare, public ceres::CostFunction
  {
  public:
    /**
     * @brief The "AnalyticChiSquare" constructor
     */
    AnalyticChiSquare(std::vector<std::pair<NangaParbat::DataHandler *, NangaParbat::ConvolutionTable *>> DSVect, NangaParbat::Parameterisation *NPFunc, bool Numeric = false);
    AnalyticChiSquare(NangaParbat::Parameterisation *NPFunc);

    // Analytic chi2 in ceres
    virtual bool Evaluate(double const *const *, double *, double **) const;

    void SetParameters(std::vector<double> const &pars);

    // Numeric chi2 in ceres
    bool operator()(double const *const *, double *) const;

    void AddBlock(std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*> DSBlock);

  private:
    int _Np;
  };
}
