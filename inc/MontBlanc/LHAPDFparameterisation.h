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
    LHAPDFparameterisation(std::string const &name, std::shared_ptr<const apfel::Grid> g, int const& member = 0);

    /**
     * @brief The "LHAPDFparameterisation" constructor
     */
    LHAPDFparameterisation(LHAPDF::PDF* set, std::shared_ptr<const apfel::Grid> g);

    /**
     * @brief The "LHAPDFparameterisation" constructor
     */
    LHAPDFparameterisation(std::vector<LHAPDF::PDF*> sets, std::shared_ptr<const apfel::Grid> g, int const& member = 0);

    /**
     * @brief The "LHAPDFparameterisation" destructor
     */
    ~LHAPDFparameterisation() { delete _FFs; };

    /**
     * @brief Function that returns the parametrisation in the form of
     * a std::function.
     */
    std::function<apfel::Set<apfel::Distribution>(double const&)> DistributionFunction() const;

  private:
    LHAPDF::PDF                       *_FFs;
    std::shared_ptr<const apfel::Grid> _g;
    apfel::ConvolutionMap              _cmap;
  };

}
