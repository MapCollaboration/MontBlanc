//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/LHAPDFparameterisation.h"

namespace MontBlanc
{
  //_________________________________________________________________________
  LHAPDFparameterisation::LHAPDFparameterisation(std::unordered_map<std::string, std::string> const &names, std::shared_ptr<const apfel::Grid> g, std::unordered_map<std::string, int> const& members):
    NangaParbat::Parameterisation("LHAPDFparameterisation::LHAPDFparameterisation", 2, {}, false),
              _g(g),
              _cmap(apfel::DiagonalBasis{13})
  {
    for (auto const& name : names)
        _MapFFs.insert({name.first, LHAPDF::mkPDF(name.second, members.at(name.first))});
  }

  //_________________________________________________________________________
  LHAPDFparameterisation::LHAPDFparameterisation(std::unordered_map<std::string, LHAPDF::PDF*> MapSet, std::shared_ptr<const apfel::Grid> g):
    NangaParbat::Parameterisation("LHAPDFparameterisation::LHAPDFparameterisation", 2, {}, false),
  _MapFFs(MapSet),
  _g(g),
  _cmap(apfel::DiagonalBasis{13})
  {
  }

  //_________________________________________________________________________
  LHAPDFparameterisation::LHAPDFparameterisation(std::unordered_map<std::string, std::vector<LHAPDF::PDF*>> MapSets, std::shared_ptr<const apfel::Grid> g, std::unordered_map<std::string, int> const& MapMember):
    NangaParbat::Parameterisation("LHAPDFparameterisation::LHAPDFparameterisation", 2, {}, false),
  _g(g),
  _cmap(apfel::DiagonalBasis{13})
  {
    for (auto const& Set : MapSets)
      {
        _MapFFs.insert({Set.first, Set.second[MapMember.at(Set.first)]});
      }
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> LHAPDFparameterisation::DistributionFunction(std::string const& hadron) const
  {
    return [=] (double const &Q) -> apfel::Set<apfel::Distribution>
    {
      return apfel::Set<apfel::Distribution> {_cmap, apfel::DistributionMap(*_g, [=] (double const& x, double const& Q) -> std::map<int, double> { return apfel::PhysToQCDEv(_MapFFs.at(hadron)->xfxQ(x, Q)); }, Q)};
    };
  }
}
