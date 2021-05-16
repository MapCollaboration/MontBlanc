//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/predictionshandlerdata.h"

namespace MontBlanc
{
  //_________________________________________________________________________
  PredictionsHandlerData::PredictionsHandlerData(NangaParbat::DataHandler const& DH, std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    _DH(DH)
  {
    // Set cuts in the mother class
    this->_cuts = cuts;

    // Compute total cut mask as a product of single masks
    _cutmask.resize(DH.GetBinning().size(), true);
    for (auto const& c : _cuts)
      _cutmask *= c->GetMask();
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandlerData::GetPredictions(std::function<double(double const&, double const&, double const&)> const&) const
  {
    return _DH.GetMeanValues();
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandlerData::GetPredictions(std::function<double(double const&, double const&, double const&)> const&,
                                                             std::function<double(double const&, double const&, double const&)> const&) const
  {
    return PredictionsHandlerData::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandlerData::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandlerData::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandlerData::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&,
                                                             std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandlerData::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }
}
