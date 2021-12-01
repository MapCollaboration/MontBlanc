//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//          Hakim Khoudli: hakim.khoudli@polytechnique.org

#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"

#include <NangaParbat/cutfactory.h>
#include <unistd.h>
#include <getopt.h>
#include <fstream>
#include <sys/stat.h>

std::string GetCurrentWorkingDir()
{
  char buff[FILENAME_MAX];
  getcwd(buff, FILENAME_MAX);
  return buff;
}

void usage_error_message(std::string call_name)
{
  std::cerr << "Usage: " << call_name << " [-a|--all_replicas] [-i|--index_result] <path to fit folder> [<member_index> (default: 0)] [<set name> (default: LHAPDFSet)]" << std::endl;
}

void compute_chi2s(std::string ResultFolder, int member_index, std::string LHAPDFSet, std::string ResultName = "Chi2s.yaml");

std::string i_to_fixed_length_str(int value, int digits_count)
{
  std::ostringstream os;
  os<<std::setfill('0')<<std::setw(digits_count)<<value;
  return os.str();
}

int main(int argc, char *argv[])
{
  const char* const short_opts = "ai";
  const option long_opts[] =
  {
    {"all_replicas", no_argument, nullptr, 'a'},
    {"index_result", no_argument, nullptr, 'i'}
    //{"replica member", required_argument, nullptr, 'r'},
  };

  bool compute_all_replicas = false;
  bool specify_index_result = false;

  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

      if (opt == -1)
        break;

      switch (opt)
        {
        case 'a':
          compute_all_replicas = true;
          break;
        case 'i':
          specify_index_result = true;
          break;
        case '?': // Unrecognized option
        default: // Unhandled option
          usage_error_message(argv[0]);
          exit(-1);
        }
    }

  if ((argc - optind) < 1)
    {
      usage_error_message(argv[0]);
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[optind];

  // Member index
  int member_index = 0;
  if ((argc - optind) >= 2)
    member_index = atoi(argv[optind+1]);

  // Name of the set
  std::string LHAPDFSet = "LHAPDFSet";
  if ((argc - optind) >= 3)
    LHAPDFSet = argv[optind+2];

  if (compute_all_replicas)
    {
      std::cout << mkdir((ResultFolder+"/IndivChi2s").c_str(), 0777) <<"ResultFolder/IndivChi2s" << "\n";
      //std::cout << (ResultFolder+"/LHAPDFSet/LHAPDFSet_00" + std::to_string(10)+".dat").c_str() << "\n";
      //std::cout << access((ResultFolder+"/LHAPDFSet/LHAPDFSet_" + i_to_fixed_length_str(10,4)+".dat").c_str(), F_OK) << "\n";
      //std::cout << access((ResultFolder).c_str(), F_OK) << "\n";

      for(size_t i=1; access((ResultFolder+"/LHAPDFSet/LHAPDFSet_" + i_to_fixed_length_str(i,4)+".dat").c_str(), F_OK) != -1; i++)
        {
          //std::cout<<(ResultFolder+"/LHAPDFSet/" + std::to_string(i));
          compute_chi2s(ResultFolder, i, LHAPDFSet, "IndivChi2s/Chi2sReplica" + std::to_string(i));
        }
    }
  else if (specify_index_result)
    {
      mkdir((ResultFolder+"/IndivChi2s").c_str(), 0777);
      compute_chi2s(ResultFolder, member_index, LHAPDFSet, "IndivChi2s/Chi2sReplica" + std::to_string(member_index));
    }
  else
    {
      compute_chi2s(ResultFolder, member_index, LHAPDFSet);
    }
  return 0;
}

void compute_chi2s(std::string ResultFolder, int member_index, std::string LHAPDFSet, std::string ResultName)
{
  // Timer
  apfel::Timer t;

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
    LHAPDF::pathsPrepend(ResultFolder + "/");
  else
    LHAPDF::pathsPrepend(GetCurrentWorkingDir() + "/" + ResultFolder + "/");

  // LHAPDF Parameterisation
  NangaParbat::Parameterisation *LHAPDF_FFs = new MontBlanc::LHAPDFparameterisation(LHAPDFSet, g, member_index);

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
    }
  std::cout << "\n- Total:\n";
  std::cout << "  - Npt = " << chi2->GetDataPointNumberAfterCuts()
	    << ", chi2 / Npt = " << chi2->Evaluate() << std::endl;
  emitter << YAML::BeginMap << YAML::Key << "Total" << YAML::Value;
  emitter << YAML::Flow << YAML::BeginMap;
  emitter << YAML::Key << "chi2" << YAML::Value << chi2->Evaluate();
  emitter << YAML::Key << "Npt" << YAML::Value << chi2->GetDataPointNumberAfterCuts();
  emitter << YAML::EndMap;
  emitter << YAML::EndSeq;

  // Print YAML:Emitter to file
  std::ofstream fout(ResultFolder +"/"+ ResultName);
  fout << emitter.c_str();
  fout.close();

  // Delete random-number generator
  gsl_rng_free(rng);

  t.stop(true);
}
