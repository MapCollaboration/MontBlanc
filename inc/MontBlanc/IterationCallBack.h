//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>
#include <NangaParbat/chisquare.h>

// Standard libs
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <ctime>
#include <csignal>

namespace MontBlanc
{
  class IterationCallBack : public ceres::IterationCallback
  {
  public:
    IterationCallBack(bool VALIDATION,
                      std::string OutputFolder,
                      int replica,
                      int Nt,
                      std::vector<double *> const &Parameters,
                      NangaParbat::ChiSquare *chi2v);

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary &);

    int GetBestIteration() { return _BestIteration; };
    double GetBestValidationChi2() { return _Bestchi2v; };
    std::vector<double> GetBestParameters() { return _BestParameters; };

  private:
    const bool _VALIDATION;
    const std::string _OutputFolder;
    const int _replica;
    const int _Nt;
    int _BestIteration;
    double _Bestchi2v;
    std::vector<double> _BestParameters;
    std::vector<double *> _Parameters;
    NangaParbat::ChiSquare *_chi2v;

  };
} // namespace MontBlanc
