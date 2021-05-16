//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"

#include <LHAPDF/LHAPDF.h>

#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <unistd.h>
#include <fstream>

std::string GetCurrentWorkingDir()
{
  char buff[FILENAME_MAX];
  getcwd(buff, FILENAME_MAX);
  return buff;
}

int main(int argc, char *argv[])
{
  if (argc != 4)
    {
      std::cerr << "Usage: " << argv[0] << " <LHAPDF set location> <input card> <path to data>" << std::endl;
      exit(-1);
    }

  // Input information
  const std::string LHAPDFSetLoc = argv[1];
  const std::string InputCardPath = argv[2];
  const std::string DataFolder = (std::string) argv[3] + "/";

  // Timer
  apfel::Timer t;

  // Read Input Card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Set silent mode for APFEL++
  apfel::SetVerbosityLevel(0);

  // APFEL++ x-space grid
  std::vector<apfel::SubGrid> vsg;
  for (auto const& sg : config["Predictions"]["xgrid"])
    vsg.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> g(new const apfel::Grid{vsg});

  // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

  // Set PDF set member to central set
  config["Predictions"]["pdfset"]["member"] = 0;

  // Vectors of DataHandler-ConvolutionTable pairs to be fed to the chi2
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect;

  // Run over the data set
  for (auto const& ds : config["Data"]["sets"])
    {
      std::cout << "Initialising " << ds["name"].as<std::string>() << (config["Data"]["closure_test"] ? " for closure test level " + config["Data"]["closure_test"]["level"].as<std::string>() : "" ) + "...\n";

      // Dataset
      NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(DataFolder + ds["file"].as<std::string>()), rng, 0};

      // Accumulate cuts
      std::vector<std::shared_ptr<NangaParbat::Cut>> cuts;
      for (auto const& c :ds["cuts"])
        cuts.push_back(NangaParbat::CutFactory::GetInstance(*DH, c["name"].as<std::string>(), c["min"].as<double>(), c["max"].as<double>()));

      // Predictions
      NangaParbat::ConvolutionTable *PH = new MontBlanc::PredictionsHandler{config["Predictions"], *DH, g, cuts};

      // Push back block
      DSVect.push_back(std::make_pair(DH, PH));
    }

  // Include new search path in LHAPDF
  LHAPDF::setPaths(GetCurrentWorkingDir() + "/" + LHAPDFSetLoc + "/");

  // Load LHAPDF set to get the number of members
  const LHAPDF::PDFSet ffset("LHAPDFSet");
  for (int irep = 1; irep < ffset.get_entry_as<int>("NumMembers"); irep++)
    {
      // Initialiase chi2 object LHAPDF Parameterisation
      NangaParbat::ChiSquare *chi2 = new MontBlanc::AnalyticChiSquare{DSVect, new MontBlanc::LHAPDFparameterisation("LHAPDFSet", g, irep)};

      // Compute weight
      const int    np = chi2->GetDataPointNumber();
      const double c2 = chi2->Evaluate() * np;
      const double lw = ( np - 1 ) * log(c2) / 2 - c2 / 2;

      // Print total chi2
      //std::cout << "Replica " << irep << ", Total chi2 / Npt = " << chi2->Evaluate() << " (Npt = " << chi2->GetDataPointNumberAfterCuts() << ")" << std::endl;
      //std::cout << "Replica " << irep << ", Total chi2 / Npt = " << chi2->Evaluate() << ", log(weight) = " << lw << std::endl;
      std::cout << std::scientific << chi2->Evaluate() << "\t" << lw << std::endl;
    }

  // Delete random-number generator
  gsl_rng_free(rng);

  t.stop(true);
  return 0;
}
