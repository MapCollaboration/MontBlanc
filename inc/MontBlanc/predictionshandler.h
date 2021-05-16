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
   * @brief The "PredictionsHandler" class provides an interface to
   * the theorerical predictions.
   */
  class PredictionsHandler: public NangaParbat::ConvolutionTable
  {
  public:
    /**
     * @brief The "PredictionsHandler" constructor
     */
    PredictionsHandler(YAML::Node                                     const& config,
                       NangaParbat::DataHandler                       const& DH,
                       std::shared_ptr<const apfel::Grid>             const& g,
                       std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts = {});

    /**
     * @brief The "PredictionsHandler" copy constructor with possibly
     * additional cuts
     */
    PredictionsHandler(PredictionsHandler                             const& DH,
                       std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts = {});

    /**
     * @brief Function that sets the input set of FFs at the initial scale
     */
    void SetInputFFs(std::function<apfel::Set<apfel::Distribution>(double const&)> const& InDistFunc);

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
    double                                         const _mu0;
    std::vector<double>                            const _Thresholds;
    std::shared_ptr<const apfel::Grid>             const _g;
    std::vector<NangaParbat::DataHandler::Binning> const _bins;
    std::vector<double>                            const _qTfact;
    apfel::ConvolutionMap                          const _cmap;
    std::vector<double>                                  _ChargeMap;
    std::vector<apfel::Set<apfel::Operator>>             _FKt;
    apfel::Set<apfel::Distribution>                      _D;
  };
}
