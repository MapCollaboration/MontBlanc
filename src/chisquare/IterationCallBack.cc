//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/IterationCallBack.h"

#include <NangaParbat/direxists.h>

namespace MontBlanc
{
  volatile sig_atomic_t stop;

  //_________________________________________________________________________
  void inthand(int signum)
  {
    stop = 1;
  }

  //_________________________________________________________________________
  IterationCallBack::IterationCallBack(bool VALIDATION,
                                       std::string OutputFolder,
                                       int replica,
                                       std::vector<double*> const &Parameters,
                                       NangaParbat::ChiSquare *chi2t,
                                       NangaParbat::ChiSquare *chi2v):
    _VALIDATION(VALIDATION),
    _OutputFolder(OutputFolder),
    _replica(replica),
    _BestIteration(0),
    _Bestchi2v(1e10),
    _Parameters(Parameters),
    _chi2t(chi2t),
    _chi2v(chi2v)
  {
    signal(SIGINT, inthand);
    const int Np = _Parameters.size();
    _BestParameters.resize(Np);
    for (int ip = 0; ip < Np; ip++)
      _BestParameters[ip] = _Parameters[ip][0];

    // Create output folder but throw an exception if it does not
    // exist.
    if (!NangaParbat::dir_exists(OutputFolder))
      system(("mkdir " + OutputFolder).c_str());

    // Create log folder
    if (!NangaParbat::dir_exists(OutputFolder + "/log/"))
      system(("mkdir " + OutputFolder + "/log/").c_str());
  }

  //_________________________________________________________________________
  ceres::CallbackReturnType IterationCallBack::operator()(const ceres::IterationSummary &summary)
  {
    // Return if the iteration is not succesful
    if (!summary.step_is_successful)
      return ceres::SOLVER_CONTINUE;

    // Get NN parameters
    int Np = _Parameters.size();
    std::vector<double> vpar(Np);
    for (int ip = 0; ip < Np; ip++)
      vpar[ip] = _Parameters[ip][0];

    double chi2v = 0;
    if (_VALIDATION)
      {
        _chi2v->SetParameters(vpar);
        chi2v = _chi2v->Evaluate();
        if (chi2v <= _Bestchi2v)
          {
            _Bestchi2v = chi2v;
            _BestIteration = summary.iteration;
            _BestParameters = vpar;
          }
      }

    // Training chi2's
    _chi2t->SetParameters(vpar);
    const double chi2t_tot = _chi2t->Evaluate();
    const int Nexp = _chi2t->GetNumberOfExperiments();
    std::vector<double> chi2t_par(Nexp);
    for (int iexp = 0; iexp < Nexp; iexp++)
      chi2t_par[iexp] = _chi2t->Evaluate(iexp);

    // Output parameters into yaml file
    YAML::Emitter emitter;
    emitter << YAML::BeginSeq;
    emitter << YAML::Flow << YAML::BeginMap;
    emitter << YAML::Key << "iteration" << YAML::Value << summary.iteration;
    //emitter << YAML::Key << "training chi2" << YAML::Value << 2 * summary.cost / _chi2t->GetDataPointNumber();
    emitter << YAML::Key << "training chi2" << YAML::Value << chi2t_tot;
    emitter << YAML::Key << "training partial chi2s" << YAML::Value << YAML::Flow << chi2t_par;
    if (_VALIDATION)
      emitter << YAML::Key << "validation chi2" << YAML::Value << chi2v;
    //emitter << YAML::Key << "parameters" << YAML::Value << YAML::Flow << vpar;
    emitter << YAML::EndMap;
    emitter << YAML::EndSeq;
    emitter << YAML::Newline;
    std::ofstream fout(_OutputFolder + "/log/replica_" + std::to_string(_replica) + ".yaml", std::ios::out | std::ios::app);
    fout << emitter.c_str();
    fout.close();

    // Manual stop by the user
    if (stop)
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;

    return ceres::SOLVER_CONTINUE;
  }
} // namespace MontBlanc
