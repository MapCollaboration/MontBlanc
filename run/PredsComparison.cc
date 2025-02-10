//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//          Hakim Khoudli: hakim.khoudli@polytechnique.org
//

#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/predictionshandler_approx.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"

#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <fstream>
#include <iomanip>

#include <getopt.h>

std::string GetCurrentWorkingDir()
{
  char buff[FILENAME_MAX];
  getcwd(buff, FILENAME_MAX);
  return buff;
}

int main(int argc, char *argv[])
{
  if ((argc - optind) < 1)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder> [<set name> (default: LHAPDFSet)]" << std::endl;
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[optind];

  // Input information
  const std::string InputCardPath = ResultFolder + "/config.yaml";
  const std::string datafolder    = ResultFolder + "/data/";
  std::string LHAPDFSet = "LHAPDFSet";
  if (argc - optind >= 2)
    LHAPDFSet = argv[optind + 1];

  // Timer
  apfel::Timer t;

  // Input card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Set silent mode for APFEL++
  apfel::SetVerbosityLevel(0);

  // Include new search path in LHAPDF
  if (ResultFolder[0]=='/')
    LHAPDF::pathsPrepend(ResultFolder + "/");
  else
    LHAPDF::pathsPrepend(GetCurrentWorkingDir() + "/" + ResultFolder + "/");

  // Set PDF set member to central set
  config["Predictions"]["pdfset"]["member"] = 0;

// APFEL++ x-space and z-space grids
  std::vector<apfel::SubGrid> vsgx, vsgz;
  const auto config_xgrid = config["Predictions"]["xgrid"];
  const auto config_zgrid = (config["Predictions"]["zgrid"] ? config["Predictions"]["zgrid"] : config_xgrid);
  // Grid for x
  for (auto const &sgx : config_xgrid)
    vsgx.push_back({sgx[0].as<int>(), sgx[1].as<double>(), sgx[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> gx(new const apfel::Grid{vsgx});
  // Grid for z
  for (auto const &sgz : config_zgrid)
    vsgz.push_back({sgz[0].as<int>(), sgz[1].as<double>(), sgz[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> gz(new const apfel::Grid{vsgz});

  // Run over the data set and gather pairs of DatHandler and
  // PredictionHandler pairs. Do not impose cuts to compute
  // predictions for all available points.
  std::cout << "Computing tables..." << std::endl;
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect_legacy;
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect_new;
  for (auto const& ds : config["Data"]["sets"])
    {
      std::cout << "#Experiment = " << ds["name"].as<std::string>() << std::endl;
      // Dataset
      NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(datafolder + ds["file"].as<std::string>())};

      // Add block to the chi2
      DSVect_legacy.push_back(std::make_pair(DH, new MontBlanc::PredictionsHandlerApprox{config["Predictions"], *DH, gz}));
      DSVect_new.push_back(std::make_pair(DH, new MontBlanc::PredictionsHandler{config["Predictions"], *DH, gx, gz}));
    }

  // Run over the experiments, compute central values and standard
  // deviations (of the shifted predictions) over the replicas.
  std::cout << "\nComputing predictions..." << std::endl;
  for (int iexp = 0; iexp < (int) DSVect_legacy.size(); iexp++)
    {
      std::cout << "#Experiment = " << DSVect_legacy[iexp].first->GetName() << std::endl;

      // Get experimental central values and uncorrelated ucertainties
      const std::vector<double> mvs = DSVect_legacy[iexp].first->GetMeanValues();
      const std::vector<double> unc = DSVect_legacy[iexp].first->GetUncorrelatedUnc();

      // Get binning
      const std::vector<NangaParbat::DataHandler::Binning> bins = DSVect_legacy[iexp].first->GetBinning();

      // Initialise averages
      std::vector<double> av_legacy(bins.size(), 0);
      std::vector<double> av_new(bins.size(), 0);
      std::vector<double> avor_legacy(bins.size(), 0);
      std::vector<double> avor_new(bins.size(), 0);

      // Get LHAPDF set
      const std::vector<LHAPDF::PDF*> sets = LHAPDF::mkPDFs(LHAPDFSet);

      // Run over replicas
      const int nrep = sets.size() - 1;
      for (int irep = 1; irep <= nrep; irep++)
        {
          // Construct chi2 object with the irep-th replica
          MontBlanc::AnalyticChiSquare chi2_legacy{DSVect_legacy, new MontBlanc::LHAPDFparameterisation{sets, gz, 0}};
          MontBlanc::AnalyticChiSquare chi2_new{DSVect_new, new MontBlanc::LHAPDFparameterisation{sets, gz, 0}};
          const std::vector<double> prds_legacy = DSVect_legacy[iexp].second->GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
          const std::vector<double> prds_new = DSVect_new[iexp].second->GetPredictions([](double const &, double const &, double const &) -> double { return 0; });

          //const std::pair<std::vector<double>, double> shifts_legacy = chi2_legacy.GetSystematicShifts(iexp);
          //const std::pair<std::vector<double>, double> shifts_new = chi2_new.GetSystematicShifts(iexp);
          for (int i = 0; i < (int) bins.size(); i++)
            {
              //av_legacy[i] += ( prds_legacy[i] + shifts_legacy.first[i] ) / nrep;
              avor_legacy[i] += prds_legacy[i] / nrep;
              //av_new[i] += ( prds_new[i] + shifts_new.first[i] ) / nrep;
              avor_new[i] += prds_new[i] / nrep;
            }
        }

      std::cout << std::setw(15) << std::left << "Experimental\n   value"
                << std::setw(20) << std::right << "Legacy"
                << std::setw(20) << std::right << "new"
                << std::setw(30) << std::right << "(legacy - new) / legacy"
                << std::endl;
      std::cout << std::setw(40) << std::setfill('-') << "" << std::setfill(' ') << std::endl;  

      for (int i = 0; i < (int) bins.size(); i++)
        {
          std::cout << std::setw(20) << std::left << mvs[i]
                    << std::setw(10) << std::right << avor_legacy[i]
                    << std::setw(20) << std::right << avor_new[i]
                    << std::setw(20) << std::right << (avor_legacy[i] - avor_new[i]) / avor_legacy[i] 
                    << std::endl;
        }
    }

  t.stop(true);
  return 0;
}
