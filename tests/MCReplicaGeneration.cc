//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/predictionshandlerdata.h"
#include "MontBlanc/AnalyticChiSquare.h"

#include <NangaParbat/cutfactory.h>

int main(int argc, char *argv[])
{
  if (argc != 4)
    {
      std::cerr << "Usage: " << argv[0] << " <Nrep> <input card> <path to data>" << std::endl;
      exit(-1);
    }

  // Input information
  const int Nrep = atoi(argv[1]);
  const std::string InputCardPath = argv[2];
  const std::string datafolder = (std::string) argv[3] + "/";

  // Read Input Card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Timer
  apfel::Timer t;

  // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

  // Initialise Map of chi2's
  std::map<std::string, std::vector<double>> chi2s;
  for (auto const& ds : config["Data"]["sets"])
    chi2s.insert({ds["name"].as<std::string>(), {}});

  // Run over replicas and accummulate chi2's
  for(int replica = 0; replica < Nrep; replica++)
    {
      // Initialise chi2 object for this particular replica
      MontBlanc::AnalyticChiSquare chi2{new NangaParbat::Parameterisation{"dummy", 2}};

      // Run over the data sets and add blocks
      for (auto const& ds : config["Data"]["sets"])
        {
          // Define dataset object
          NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(datafolder + ds["file"].as<std::string>()), rng, replica + 1};

          // Accumulate cuts
          std::vector<std::shared_ptr<NangaParbat::Cut>> cuts;
          for (auto const& c :ds["cuts"])
            cuts.push_back(NangaParbat::CutFactory::GetInstance(*DH, c["name"].as<std::string>(), c["min"].as<double>(), c["max"].as<double>()));

          chi2.AddBlock(std::make_pair(DH, new MontBlanc::PredictionsHandlerData{*DH, cuts}));
          chi2s[DH->GetName()].push_back(chi2.NangaParbat::ChiSquare::Evaluate(chi2.GetNumberOfExperiments()-1));
        }
    }

  for (auto const& ds : config["Data"]["sets"])
    {
      double avg_chi2 = accumulate(chi2s[ds["name"].as<std::string>()].begin(), chi2s[ds["name"].as<std::string>()].end(), 0.) / Nrep;
      std::cout << "- " << ds["name"].as<std::string>() << ":\n";
      std::cout << "  - <chi2 / Npt>(Nrep=" << Nrep << ") = " << avg_chi2 << std::endl;
    }

  // Delete random-number generator
  gsl_rng_free(rng);

  t.stop(true);
  return 0;
}
