//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include <apfel/apfelxx.h>
#include <LHAPDF/LHAPDF.h>
#include <yaml-cpp/yaml.h>
#include <NNAD/FeedForwardNN.h>

#include <functional>
#include <fstream>
#include <algorithm>

typedef std::pair<std::vector<double>, double> PVD;
typedef std::vector<PVD> VPVD;
typedef std::vector<std::vector<double> > VV;

bool wayToSort(PVD i, PVD j)
{
  return i.second < j.second;  //ascending order
}

int main(int argc, char *argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder> [<hadron> (default: PIp, options: PIp, PIm, PIsum)] [<set name> (default: LHAPDFSet)] [<Nmembers> (default: all)]" << std::endl;
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[1];

  // Name of the set
  std::string hadron = "PIp";
  if (argc >= 3)
    hadron = argv[2];

  std::string OutName = "LHAPDFSet";
  if (argc >= 4)
    OutName = argv[3];

  // Read Input Card
  YAML::Node config = YAML::LoadFile(ResultFolder + "/config.yaml");

  // Retrive best-fit paramaters
  YAML::Node bestfits = YAML::LoadFile(ResultFolder + "/BestParameters.yaml");

  VPVD AllPars;
  VV BestPars;

  // Load all parameters and pair them with the total chi2 value for sorting
  for (auto const &rep : bestfits)
    AllPars.push_back(PVD(rep["parameters"].as<std::vector<double> >(), rep["total chi2"].as<double>()));

  // Sort sets of parameters according to the chi2
  sort(AllPars.begin(), AllPars.end(), wayToSort);

  int Nmembers = 0;
  if (argc >= 5)
    {
      Nmembers = std::stoi(argv[4]);
      std::cout << "Nmembers requested = " << Nmembers << std::endl;
    }
  else
    Nmembers = AllPars.size();

  // Pick the lowest Nmembers of the chi2-sorted replicas
  int i = 0;
  for (auto const &rep : AllPars)
    {
      if (i == Nmembers)
        break;

      BestPars.push_back(rep.first);
      i++;
    }
  if (i < Nmembers)
    {
      std::cerr << "Requested more replicas than available." << std::endl;
      exit(-1);
    }

  // Construct rotation matrix to obtain FFs in the evolution basis
  // that is what is fed to APFEL++ to do the evolution.
  const std::vector<int> Architecture = config["NNAD"]["architecture"].as<std::vector<int>>();

  // Rotation from physical to QCD-evolution basis
  std::vector<double> R;
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 13; j++)
      R.push_back(apfel::RotPhysToQCDEvFull[i][j]);

  // Combine rotation matrix with the flavour map
  nnad::Matrix<double> FlavourMapT{Architecture.back(), 13, config["NNAD"]["flavour map"].as<std::vector<double>>()};

  // Calculate the (Moore-Penrose) pseudo-inverse of the flavour map
  if(config["NNAD"]["combine"])
    {
      if(config["NNAD"]["combine"].as<bool>())
        FlavourMapT = FlavourMapT.PseudoInverse_LLR();
      else
        FlavourMapT.Transpose();
    }
  else
    FlavourMapT.Transpose();
  nnad::Matrix<double> Rotation = nnad::Matrix<double> {13, 13, R} * FlavourMapT;

  // Whether the output is linear or quadratic
  const int OutputFunction = (config["NNAD"]["output function"] ? config["NNAD"]["output function"].as<int>() : 1);

  // Initialise neural network
  nnad::FeedForwardNN<double> NN{Architecture, 0, false};

  // APFEL++ EvolutionSetup object
  apfel::EvolutionSetup es{};

  // Adjust evolution parameters
  es.Virtuality        = apfel::EvolutionSetup::Virtuality::TIME;
  es.Q0                = config["Predictions"]["mu0"].as<double>();
  es.PerturbativeOrder = config["Predictions"]["perturbative order"].as<int>();
  es.QQCDRef           = config["Predictions"]["alphas"]["Qref"].as<double>();
  es.AlphaQCDRef       = config["Predictions"]["alphas"]["aref"].as<double>();
  es.Thresholds        = config["Predictions"]["thresholds"].as<std::vector<double>>();
  es.Masses            = es.Thresholds;
  es.Qmin              = 1;
  es.Qmax              = 1000;
  es.name              = OutName;
  es.GridParameters    = {{100, 1e-2, 3}, {60, 1e-1, 3}, {50, 6e-1, 3}, {50, 8e-1, 3}};
  es.InSet.clear();

  // NN Parameterisation. First compute the average.
  std::vector<std::function<std::map<int, double>(double const&, double const&)>> sets
  {
    [&] (double const& x, double const&) -> std::map<int, double>
    {
      // Initialise map in the QCD evolution basis
      std::map<int, double> EvMap{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}};

      // Number of replicas used for the average
      const int nr = BestPars.size();

      // Now run over all replicas and accumulated
      for (auto p: BestPars)
        {
          // Set NN parameters
          NN.SetParameters(p);

          // Call NN at x and 1
          const std::vector<double> nn1 = NN.Evaluate({1});
          std::vector<double> nnx = NN.Evaluate({x});

          // Subtract the NN at 1 to ensure NN(x = 1) = 0.
          std::transform(nnx.begin(), nnx.end(), nn1.begin(), nnx.begin(), std::minus<double>());

          // If _OutputFunction == 2 square the output vector.
          if (OutputFunction == 2)
            std::transform(nnx.begin(), nnx.end(), nnx.begin(), nnx.begin(), std::multiplies<double>());

          // Rotate into the evolution basis
          const nnad::Matrix<double> nnv = Rotation * nnad::Matrix<double> {Architecture.back(), 1, nnx};

          // Fill in map
          for (int i = 0; i < 13; i++)
            {
              if (!hadron.compare("PIm"))
                EvMap[i] += (!(i % 2) && i != 0 ? -1 : 1) * nnv.GetElement(i, 0) / nr;
              else if (!hadron.compare("PIsum"))
                EvMap[i] += (!(i % 2) && i != 0 ? 0 : 2) * nnv.GetElement(i, 0) / nr;
              else //PIp
                EvMap[i] += nnv.GetElement(i, 0) / nr;
            }
        }
      return EvMap;
    }
  };
  // Now run over replicas
  for (auto p: BestPars)
    {
      sets.push_back([&,p] (double const& x, double const&) -> std::map<int, double>
      {
        // Set NN parameters
        NN.SetParameters(p);

        // Call NN at x and 1
        const std::vector<double> nn1 = NN.Evaluate({1});
        std::vector<double> nnx = NN.Evaluate({x});

        // Subtract the NN at 1 to ensure NN(x = 1) = 0.
        std::transform(nnx.begin(), nnx.end(), nn1.begin(), nnx.begin(), std::minus<double>());

        // If _OutputFunction == 2 square the output vector.
        if (OutputFunction == 2)
          std::transform(nnx.begin(), nnx.end(), nnx.begin(), nnx.begin(), std::multiplies<double>());

        // Rotate into the evolution basis
        const nnad::Matrix<double> nnv = Rotation * nnad::Matrix<double>{Architecture.back(), 1, nnx};

        // Fill in map (ignoring top)
        std::map<int, double> EvMap;
        for (int i = 0; i < 13; i++)
          {
            if (!hadron.compare("PIm"))
              EvMap[i] += (!(i % 2) && i != 0 ? -1 : 1) * nnv.GetElement(i, 0);
            else if (!hadron.compare("PIsum"))
              EvMap[i] += (!(i % 2) && i != 0 ? 0 : 2) * nnv.GetElement(i, 0);
            else //PIp
              EvMap[i] += nnv.GetElement(i, 0);
          }

        return EvMap;
      });
    }
  es.InSet = sets;

  // Feed it to the initialisation class of APFEL++ and create a grid
  apfel::InitialiseEvolution evpdf{es, true};

  // Move set into the result folder if the set does not exist yet
  std::rename(OutName.c_str(), (ResultFolder + "/" + OutName).c_str());

  return 0;
}
