//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#pragma once

#include "MontBlanc/LHAPDFparameterisation.h"

#include <LHAPDF/LHAPDF.h>
#include <NangaParbat/parameterisation.h>
#include <apfel/apfelxx.h>

namespace MontBlanc
{
  class LHAPDFparameterisation: public NangaParbat::Parameterisation
  {
  public:

    /**
     * @brief The "LHAPDFparameterisation" constructor
     */
    LHAPDFparameterisation(std::unordered_map<std::string, std::string> const &names, std::shared_ptr<const apfel::Grid> g, std::unordered_map<std::string, int> const& members);

    /**
     * @brief The "LHAPDFparameterisation" constructor
     */
    LHAPDFparameterisation(std::unordered_map<std::string, LHAPDF::PDF*> MapSet, std::shared_ptr<const apfel::Grid> g);

    /**
     * @brief The "LHAPDFparameterisation" constructor
     */
    LHAPDFparameterisation(std::unordered_map<std::string, std::vector<LHAPDF::PDF*>> MapSets, std::shared_ptr<const apfel::Grid> g, std::unordered_map<std::string, int> const& MapMember = {});

    /**
     * @brief The "LHAPDFparameterisation" destructor
     */
    ~LHAPDFparameterisation()
      {
        for (auto const& p : _MapFFs)
          delete p.second;
      }

    /**
     * @brief Function that returns the parametrisation in the form of
     * a std::function.
     */
    std::function<apfel::Set<apfel::Distribution>(double const&)> DistributionFunction(std::string const&) const;


  private:
    std::unordered_map<std::string, LHAPDF::PDF*>        _MapFFs;
    std::shared_ptr<const apfel::Grid>                   _g;
    apfel::ConvolutionMap                                _cmap;
  };

}
