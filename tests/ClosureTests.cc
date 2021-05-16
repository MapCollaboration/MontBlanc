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

int main(int argc, char *argv[])
{
  if (argc != 5)
    {
      std::cerr << "Usage: " << argv[0] << " <CT-level> <Nrep> <input card> <path to data>" << std::endl;
      exit(-1);
    }

  // Input information
  const int CTlvl = atoi(argv[1]);
  const int Nrep = atoi(argv[2]);
  const std::string InputCardPath = argv[3];
  const std::string datafolder = (std::string) argv[4] + "/";

  // Read Input Card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  apfel::Timer t;

  // Set silent mode for APFEL++
  apfel::SetVerbosityLevel(0);

  // APFEL++ x-space grid
  std::vector<apfel::SubGrid> vsg;
  for (auto const& sg : config["Predictions"]["xgrid"])
    vsg.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> g(new const apfel::Grid{vsg});

  // LHAPDF Parameterisation
  NangaParbat::Parameterisation *LHAPDF_FFs = new MontBlanc::LHAPDFparameterisation("NNFF10_PIsum_nlo", g);

  // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

  // Run over the data set and accumulate all DH MC replicas
  std::vector<NangaParbat::ConvolutionTable*> CTVect;
  std::vector<std::vector<NangaParbat::DataHandler*>> DHVect(Nrep);
  std::map<std::string, std::vector<double>> chi2s;
  for (auto const& ds : config["Data"]["sets"])
    {
      // Dataset
      for(int replica = 0; replica < Nrep; replica++)
        DHVect[replica].push_back(new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(datafolder + ds["file"].as<std::string>()), rng, replica + 1});

      // Accumulate cuts
      std::vector<std::shared_ptr<NangaParbat::Cut>> cuts;
      for (auto const& c :ds["cuts"])
        cuts.push_back(NangaParbat::CutFactory::GetInstance(*(DHVect[0].back()), c["name"].as<std::string>(), c["min"].as<double>(), c["max"].as<double>()));

      // Predictions
      CTVect.push_back(new MontBlanc::PredictionsHandler{config["Predictions"], *(DHVect[0].back()), g, cuts});
      chi2s.insert({ds["name"].as<std::string>(), {}});
    }

  // Run over the MC replicas and accumulate chi2's
  for(int replica = 0; replica < Nrep; replica++)
    {
      std::cout << "replica = " << replica + 1 << std::endl;
      std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect;
      for (int j = 0; j < (int) CTVect.size(); j++)
        {
          // Pseudo data part
          CTVect[j]->SetInputFFs(LHAPDF_FFs->DistributionFunction());
          const std::vector<double> theories = CTVect[j]->GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
          NangaParbat::DataHandler *PDH = new NangaParbat::DataHandler(*(DHVect[replica][j]));
          PDH->SetMeans(theories);

          switch(CTlvl)
            {
            case 0:
              PDH->FluctuateData(rng, 0);
              break;

            case 1:
              PDH->FluctuateData(rng, config["Data"]["seed"].as<int>());
              break;

            case 2:
              PDH->FluctuateData(rng, config["Data"]["seed"].as<int>());
              PDH->SetMeans(PDH->GetFluctutatedData());
              PDH->FluctuateData(rng, replica + 1);
              break;

            default:
              std::cout << "only CT-level 0, 1 and 2 are defined" << std::endl;
              exit(1);
            }
          DSVect.push_back(std::make_pair(PDH, CTVect[j]));
        }
      MontBlanc::AnalyticChiSquare chi2{DSVect, LHAPDF_FFs};
      for (int j = 0; j < (int) CTVect.size(); j++)
        chi2s[DHVect[replica][j]->GetName()].push_back(chi2.NangaParbat::ChiSquare::Evaluate(j));
    }

  for (auto const& ds : config["Data"]["sets"])
    {
      double avg_chi2 = accumulate(chi2s[ds["name"].as<std::string>()].begin(), chi2s[ds["name"].as<std::string>()].end(), 0.0) / Nrep;
      std::cout << "- " << ds["name"].as<std::string>() << ":\n";
      std::cout << "  - <chi2 / Npt>(Nrep=" << Nrep << ") = " << avg_chi2 << std::endl;
    }
  // Print total chi2
  //std::cout << "\nTotal chi2 / Npt = " << chi2->Evaluate() << "\n" << std::endl;

  // Delete random-number generator
  gsl_rng_free(rng);

  t.stop(true);
  return 0;
}
