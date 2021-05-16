//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"

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
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder> [<set name> (default: LHAPDFSet)]" << std::endl;
      exit(-1);
    }

  // Timer
  apfel::Timer t;

  // Path to result folder
  const std::string ResultFolder = argv[1];

  // Name of the set
  std::string LHAPDFSet = "LHAPDFSet";
  if (argc >= 3)
    LHAPDFSet = argv[2];

  // Input card
  YAML::Node config = YAML::LoadFile(ResultFolder + "/config.yaml");

  // Set silent mode for APFEL++
  apfel::SetVerbosityLevel(0);

  // APFEL++ x-space grid
  std::vector<apfel::SubGrid> vsg;
  for (auto const& sg : config["Predictions"]["xgrid"])
    vsg.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> g(new const apfel::Grid{vsg});

  // Include new search path in LHAPDF
  if (ResultFolder[0]=='/')
    LHAPDF::pathsAppend(ResultFolder + "/");
  else
    LHAPDF::pathsAppend(GetCurrentWorkingDir() + "/" + ResultFolder + "/");

  // LHAPDF Parameterisation
  NangaParbat::Parameterisation *LHAPDF_FFs = new MontBlanc::LHAPDFparameterisation(LHAPDFSet, g);

  // Initialiase chi2 object
  NangaParbat::ChiSquare *chi2 = new MontBlanc::AnalyticChiSquare{LHAPDF_FFs};

  // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

  // Set PDF set member to central set
  config["Predictions"]["pdfset"]["member"] = 0;

  // Run over the data set
  const std::string DataFolder = ResultFolder + "/data/";
  YAML::Emitter emitter;
  emitter << YAML::BeginSeq;
  for (auto const& ds : config["Data"]["sets"])
    {
      // Dataset
      NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(DataFolder + ds["file"].as<std::string>()), rng, 0};

      // Accumulate cuts
      std::vector<std::shared_ptr<NangaParbat::Cut>> cuts;
      for (auto const& c :ds["cuts"])
        cuts.push_back(NangaParbat::CutFactory::GetInstance(*DH, c["name"].as<std::string>(), c["min"].as<double>(), c["max"].as<double>()));

      // Predictions
      NangaParbat::ConvolutionTable *PH = new MontBlanc::PredictionsHandler{config["Predictions"], *DH, g, cuts};

      // Add block to the chi2
      chi2->AddBlock(std::make_pair(DH, PH));

      // Get datset name and the corresponding chi2 normalised to the
      // number of points that pass the cuts.
      std::cout << "- " << DH->GetName() << ":\n";
      std::cout << "  - Npt = " << chi2->GetDataPointNumbersAfterCuts()[chi2->GetNumberOfExperiments()-1]
                << ", chi2 / Npt = " << chi2->Evaluate(chi2->GetNumberOfExperiments()-1) << std::endl;
      emitter << YAML::BeginMap << YAML::Key << DH->GetName() << YAML::Value;
      emitter << YAML::Flow << YAML::BeginMap;
      emitter << YAML::Key << "chi2" << YAML::Value << chi2->Evaluate(chi2->GetNumberOfExperiments()-1);
      emitter << YAML::Key << "Npt" << YAML::Value << chi2->GetDataPointNumbersAfterCuts()[chi2->GetNumberOfExperiments()-1];
      emitter << YAML::EndMap;
      emitter << YAML::EndMap;
      /*
            // Compute chi2 starting from the penalty
            const std::vector<NangaParbat::DataHandler::Binning> bins = DH->GetBinning();
            const std::vector<double> mvs  = DH->GetFluctutatedData();
            const std::vector<double> unc  = DH->GetUncorrelatedUnc();
            const std::vector<double> prds = PH->GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
            const std::pair<std::vector<double>, double> shifts = chi2->GetSystematicShifts(chi2->GetNumberOfExperiments()-1);
            const std::valarray<bool> CutMask = PH->GetCutMask();

            // Compute chi2 starting from the penalty and print predictions
            double chi2n = shifts.second;
            for (int i = 0; i < (int) bins.size(); i++)
      	{
      	  chi2n += pow( ( ( CutMask[i] ? mvs[i] - prds[i] : 0 ) - shifts.first[i] ) / unc[i], 2);
      	  if (bins[i].Qav > 1.28)
      	    std::cout << i << std::scientific << "\t["
      	      //<< bins[i].xmin << ": " << bins[i].xmax << "]\t["
      	      //<< bins[i].ymin << ": " << bins[i].ymax << "]\t["
      	      //<< bins[i].zmin << ": " << bins[i].zmax << "]\t"
      		      << bins[i].xav << "\t"
      		      << bins[i].zav << "\t"
      		      << pow(bins[i].Qav, 2) << "\t"
      		      << prds[i] << "\t" << shifts.first[i] << "\t" << mvs[i] << " +- " << unc[i] << "\t"
      		      << std::endl;
      	}
            std::cout << "chi2 / Npt (nuis. pars.) = " << chi2n / chi2->GetDataPointNumbers()[chi2->GetNumberOfExperiments()-1] << std::endl;
      */
    }
  std::cout << "\nTotal chi2 / Npt = " << chi2->Evaluate() << "\n" << std::endl;
  emitter << YAML::BeginMap << YAML::Key << "Total" << YAML::Value;
  emitter << YAML::Flow << YAML::BeginMap;
  emitter << YAML::Key << "chi2" << YAML::Value << chi2->Evaluate();
  emitter << YAML::Key << "Npt" << YAML::Value << chi2->GetDataPointNumberAfterCuts();
  emitter << YAML::EndMap;
  emitter << YAML::EndSeq;

  // Print YAML:Emitter to file
  std::ofstream fout(ResultFolder + "/Chi2s.yaml");
  fout << emitter.c_str();
  fout.close();

  // Delete random-number generator
  gsl_rng_free(rng);

  t.stop(true);
  return 0;
}
