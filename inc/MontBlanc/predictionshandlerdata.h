//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#pragma once

#include <NangaParbat/convolutiontable.h>
#include <NangaParbat/datahandler.h>
#include <NangaParbat/cut.h>
#include <yaml-cpp/yaml.h>
#include <apfel/apfelxx.h>

namespace MontBlanc
{
  /**
   * @brief The "PredictionsHandlerData" class provides an interface
   * to the predictions assumed to be equal to experimental central
   * values. This class is useful for testing the MC replica
   * generation.
   */
  class PredictionsHandlerData: public NangaParbat::ConvolutionTable
  {
  public:
    /**
     * @brief The "PredictionsHandlerData" constructor
     */
    PredictionsHandlerData(NangaParbat::DataHandler const& DH, std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts);

    /**
     * @brief Overload function of the NangaParbat::ConvolutionTable
     * class
     */
    std::vector<double> GetPredictions(std::function<double(double const&, double const&, double const&)> const&) const;
    std::vector<double> GetPredictions(std::function<double(double const&, double const&, double const&)> const&,
                                       std::function<double(double const&, double const&, double const&)> const&) const;
    std::vector<double> GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&) const;
    std::vector<double> GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&,
                                       std::function<double(double const&, double const&, double const&, int const&)> const&) const;

  private:
    NangaParbat::DataHandler const& _DH;
  };
}
