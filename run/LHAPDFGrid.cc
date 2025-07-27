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
typedef std::function<std::vector<double>(double const&, std::vector<double> const&)> NNxFunc;
typedef std::function<std::map<int, double> (double const&, double const&)> EvMapApfel;

apfel::EvolutionSetup initEvSetup(YAML::Node const& config, std::string const& hadron);
void fillEvSet(NNxFunc& NNfunc, std::string const& hadron, nnad::Matrix<double> const& SplitMatrix, nnad::Matrix<double> const& FlavRoatation,  VV const& BestPars, apfel::EvolutionSetup& es);

bool wayToSort(PVD i, PVD j)
{
  return i.second < j.second;  //ascending order
}

int main(int argc, char *argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder> [<set token> (default: LHAPDFSet_)] [<Nmembers> (default: all)]" << std::endl;
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[1];

  // Name of the set
  //TODO backwards compatibility
  // std::string hadron = "PIp";
  // if (argc >= 3)
  //   hadron = argv[2];

  //TODO backwards compatibility
  std::string TokenOutName = "LHAPDFSet_";
  if (argc >= 3)
    TokenOutName = argv[2];

  // Read Input Card
  const YAML::Node config = YAML::LoadFile(ResultFolder + "/config.yaml");

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
  if (argc >= 4)
    {
      Nmembers = std::stoi(argv[3]);
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

  // Whether the output is linear or quadratic
  const int OutputFunction = (config["NNAD"]["output function"] ? config["NNAD"]["output function"].as<int>() : 1);

  // Initialise neural network
  nnad::FeedForwardNN<double> NN{Architecture, 0, false};
  NNxFunc NNx = [&] (double const& x, std::vector<double> const& Pars) -> std::vector<double>
  {
    // Set NN parameters
    NN.SetParameters(Pars);

    // Call NN at x and 1
    const std::vector<double> nn1 = NN.Evaluate({1});
    std::vector<double> nnx = NN.Evaluate({x});

    // Subtract the NN at 1 to ensure NN(x = 1) = 0.
    std::transform(nnx.begin(), nnx.end(), nn1.begin(), nnx.begin(), std::minus<double>());

    // If _OutputFunction == 2 square the output vector.
    if (OutputFunction == 2)
      std::transform(nnx.begin(), nnx.end(), nnx.begin(), nnx.begin(), std::multiplies<double>());

    return nnx;
  };

  // Generate rotation matrices
  int HadronOutputSize;
  nnad::Matrix<double> Rotation;
  int TotalSize = 0;
  for (auto &maps : config["NNAD"]["flavour maps"])
    {
      auto hadron = maps["hadron"].as<std::string>();

      std::vector<double> FlavourMap = maps["map"].as<std::vector<double>>();
      HadronOutputSize = (int) FlavourMap.size() / 13;

      // Layer: Rotation into full flavour basis
      nnad::Matrix<double> FlavourMapT{ HadronOutputSize, 13, FlavourMap};
      if (config["combine"] ? config["combine"].as<bool>() : false)
        FlavourMapT = FlavourMapT.PseudoInverse_LLR();
      else
        FlavourMapT.Transpose();

      // Layer: Rotation to QCD evolution from full flavour basis
      Rotation = nnad::Matrix<double> {13, 13, R} * FlavourMapT;

      // Prepare the matrix that splits the output of the network
      nnad::Matrix<double> SplitMatrix {HadronOutputSize, Architecture.back(), std::vector<double>(HadronOutputSize * Architecture.back(), 0.0)};
      for (int j = 0; j < (int) HadronOutputSize; j++)
        SplitMatrix.SetElement(j, j + TotalSize, 1.0);

      apfel::EvolutionSetup es = initEvSetup(config, TokenOutName + hadron);
      fillEvSet(NNx, hadron, SplitMatrix, Rotation, BestPars, es);

      // Custom LHAPDF-grid header
      std::string GridHeader = "SetDesc: '" + es.name + " ";
      if (hadron == "PIp")
        GridHeader += "- pi^+ ";
      else if (hadron == "PIm")
        GridHeader += "- pi^- ";
      else if (hadron == "PIsum")
        GridHeader += "- (pi^+ + pi^-) ";
      else if (hadron == "KAp")
        GridHeader += "- K^+ ";
      else if (hadron == "KAm")
        GridHeader += "- K^- ";
      else if (hadron == "KAsum")
        GridHeader += "- (K^+ + K^-) ";
      else
        GridHeader += "- Unknown species ";
      GridHeader += "FF fit at " + std::string(es.PerturbativeOrder, 'N') + "LO - mem=0 => average over replicas, ";
      GridHeader += "mem=1-" + std::to_string(Nmembers) + " => Monte Carlo replicas - set generated with APFEL++'\n";

      GridHeader += "SetIndex: 0000000\n";
      GridHeader += "SetType: fragfn\n";
      GridHeader += "Authors: R. Abdul Khalek, V. Bertone, A. Khoudli, E. R. Nocera\n";
      GridHeader += "Reference: arXiv:xxxx.xxxxx\n";
      GridHeader += "Format: lhagrid1\n";
      GridHeader += "DataVersion: 1\n";
      if (hadron.substr(0, 2) == "PI")
        GridHeader += "Particle: 211\n";
      else if (hadron.substr(0, 2) == "KA")
        GridHeader += "Particle: 321\n";
      else
        GridHeader += "Particle: 000\n";
      GridHeader += "FlavorScheme: variable\n";
      GridHeader += "ErrorType: replicas";

      // Feed it to the initialisation class of APFEL++ and create a grid
      apfel::InitialiseEvolution evpdf{es, true, GridHeader};


      std::string OutName = TokenOutName + hadron;

      // Move set into the result folder if the set does not exist yet
      std::cout << "Moving " << OutName << " to " << ResultFolder << std::endl;
      std::rename(OutName.c_str(), (ResultFolder + "/" + OutName).c_str());

      TotalSize += HadronOutputSize;
    }

  return 0;
}

[[nodiscard]] apfel::EvolutionSetup initEvSetup(YAML::Node const& config, std::string const& name)
{
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
  es.name              = name;
  es.GridParameters    = {{100, 1e-2, 3}, {100, 1e-1, 3}, {50, 6e-1, 3}, {50, 8e-1, 3}};
  es.InSet.clear();

  return es;
}


void fillEvSet(NNxFunc& NNfunc,
               std::string const& hadron,
               nnad::Matrix<double> const& SplitMatrix,
               nnad::Matrix<double> const& FlavRoatation,
               VV const& BestPars,
               apfel::EvolutionSetup& es)
{
  //TODO Can we get replica_0 from replica_i?
  EvMapApfel replica_0 = [&] (double const& x, double const&) -> std::map<int, double>
  {
    // Initialise map in the QCD evolution basis
    std::map<int, double> EvMap{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}};

    // Number of replicas used for the average
    const int nr = BestPars.size();

    // Now run over all replicas and accumulated
    for (auto p: BestPars)
      {
        auto nnx = NNfunc(x, p);

        // Selects the specific hadron output
        nnad::Matrix<double> SplitOutput = SplitMatrix * nnad::Matrix(nnx.size(), 1, nnx);

        // Rotate into the evolution basis
        const nnad::Matrix<double> nnv = FlavRoatation * SplitOutput;

        // Fill in map
        for (int i = 0; i < 13; i++)
          {
            if (!hadron.compare("PIm") || !hadron.compare("KAm"))
              EvMap[i] += (!(i % 2) && i != 0 ? -1 : 1) * nnv.GetElement(i, 0) / nr;
            else if (!hadron.compare("PIsum") || !hadron.compare("KAsum"))
              EvMap[i] += (!(i % 2) && i != 0 ? 0 : 2) * nnv.GetElement(i, 0) / nr;
            else // PIp or KAp
              EvMap[i] += nnv.GetElement(i, 0) / nr;
          }
      }
    return EvMap;
  };
  std::function<EvMapApfel(std::vector<double> const&)> replica_i = [&] (std::vector<double> const& p) -> EvMapApfel
  {
    return [&,p] (double const& x, double const&) -> std::map<int, double>
    {
      // Initialise map in the QCD evolution basis
      std::map<int, double> EvMap{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}};

      auto nnx = NNfunc(x, p);

      // Selects the specific hadron output
      nnad::Matrix<double> SplitOutput = SplitMatrix * nnad::Matrix(nnx.size(), 1, nnx);

      // Rotate into the evolution basis
      const nnad::Matrix<double> nnv = FlavRoatation * SplitOutput;

      // Fill in map (ignoring top)
      for (int i = 0; i < 13; i++)
        {
          if (!hadron.compare("PIm") || !hadron.compare("KAm"))
            EvMap[i] += (!(i % 2) && i != 0 ? -1 : 1) * nnv.GetElement(i, 0);
          else if (!hadron.compare("PIsum") || !hadron.compare("KAsum"))
            EvMap[i] += (!(i % 2) && i != 0 ? 0 : 2) * nnv.GetElement(i, 0);
          else // PIp or KAp
            EvMap[i] += nnv.GetElement(i, 0);
        }
      return EvMap;
    };
  };

  es.InSet.push_back(replica_0);
  for (auto p: BestPars)
    es.InSet.push_back(replica_i(p));
}

