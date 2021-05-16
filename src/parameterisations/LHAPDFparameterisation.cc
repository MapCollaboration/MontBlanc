//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/LHAPDFparameterisation.h"

namespace MontBlanc
{
  //_________________________________________________________________________
  LHAPDFparameterisation::LHAPDFparameterisation(std::string const &name, std::shared_ptr<const apfel::Grid> g, int const& member):
    NangaParbat::Parameterisation("LHAPDFparameterisation::LHAPDFparameterisation", 2, {}, false),
              _FFs(LHAPDF::mkPDF(name, member)),
              _g(g),
              _cmap(apfel::DiagonalBasis{13})
  {
  }

  //_________________________________________________________________________
  LHAPDFparameterisation::LHAPDFparameterisation(LHAPDF::PDF* set, std::shared_ptr<const apfel::Grid> g):
    NangaParbat::Parameterisation("LHAPDFparameterisation::LHAPDFparameterisation", 2, {}, false),
  _FFs(set),
  _g(g),
  _cmap(apfel::DiagonalBasis{13})
  {
  }

  //_________________________________________________________________________
  LHAPDFparameterisation::LHAPDFparameterisation(std::vector<LHAPDF::PDF*> sets, std::shared_ptr<const apfel::Grid> g, int const& member):
    NangaParbat::Parameterisation("LHAPDFparameterisation::LHAPDFparameterisation", 2, {}, false),
  _FFs(sets[member]),
  _g(g),
  _cmap(apfel::DiagonalBasis{13})
  {
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> LHAPDFparameterisation::DistributionFunction() const
  {
    return [=] (double const &Q) -> apfel::Set<apfel::Distribution>
    {
      return apfel::Set<apfel::Distribution> {_cmap, apfel::DistributionMap(*_g, [=] (double const& x, double const& Q) -> std::map<int, double> { return apfel::PhysToQCDEv(_FFs->xfxQ(x, Q)); }, Q)};
    };
  }
}
