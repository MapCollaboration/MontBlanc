#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"

#include <LHAPDF/LHAPDF.h>

#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <unistd.h>
#include <fstream>
#include <filesystem>

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
      std::cerr << "Usage: " << argv[0] << " <LHAPDF set location> <input card> <path to data> " << std::endl;
      exit(-1);
    }


  // Input information

  const std::string LHAPDFSetLoc = argv[1];
  const std::string InputCardPath = argv[2];
  const std::string DataFolder = (std::string) argv[3] + "/";
  const std::string Weights       = "fit/MAPFF20_PI_NNLO_Q1_00/Weights/";
 // const int replica = std::stoi(argv[4]);
  std::vector<float> weight_vect;
  float weight;

   if (!std::filesystem::exists(Weights)) 
   {
     std::filesystem::create_directory(Weights);
     std::cout << "Folder created successfully." << std::endl;
   }

  // Timer
  apfel::Timer t;
  double den = 0.;

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

  // Set PDF set member to the given replica
  //config["Predictions"]["pdfset"]["member"] = replica;

  //Loop over PDFs replica
  int N_rep = 1000;


  for(int i = 1; i <= N_rep; i++)
    {
      config["Predictions"]["pdfset"]["member"] = i;
     
  // Vectors of DataHandler-ConvolutionTable pairs to be fed to the chi2
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect;

  // Run over the data set
  for (auto const& ds : config["Data"]["sets"])
    {
       if(ds["name"].as<std::string>().find("COMPASS") != std::string::npos || 
              ds["name"].as<std::string>().find("HERMES") != std::string::npos)
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
    }

  // Include new search path in LHAPDF
  LHAPDF::setPaths(GetCurrentWorkingDir() + "/" + LHAPDFSetLoc + "/");

  // Load LHAPDF set to get the number of members
  //const LHAPDF::PDFSet ffset("MySet");
  const std::vector<LHAPDF::PDF*> sets = LHAPDF::mkPDFs("MySetForPim");
  //for (int irep = 1; irep < ffset.get_entry_as<int>("NumMembers"); irep++)
    //{
      // Initialiase chi2 object LHAPDF Parameterisation
      NangaParbat::ChiSquare *chi2 = new MontBlanc::AnalyticChiSquare{DSVect, new MontBlanc::LHAPDFparameterisation(sets, g, 0)};

      // Compute weight
      const int    np = chi2->GetDataPointNumber();
      const double c2 = chi2->Evaluate();
      const double lw = ( np - 1 ) * log(c2*np) / 2 - c2*np / 2 - 0.5*(np - 1) * log(np) + log(N_rep);
      
      den += pow(c2, 0.5*(np - 1)) * exp(- 0.5 * c2*np); 
      weight_vect.push_back(lw);
      //std::cout << std::scientific << np << "\t" << lw <<"\t"<<log(den)<< std::endl;
      // i ounti sono 838
      
     /* emitter << YAML::Flow << YAML::BeginMap;
      emitter << YAML::Key << "log num" << YAML::Value << lw;
       emitter << YAML::Key << "chi2" << YAML::Value << c2*np;
        emitter << YAML::Key << "den" << YAML::Value << den;
       emitter << YAML::EndMap;*/

    }
   /*   emitter<< YAML::EndSeq;
  emitter << YAML::EndMap;
  std::ofstream fout(Weights + "weight_for_PDFS.yaml");
  fout << emitter.c_str();
  fout.close();  */
      //una volta raccolto i 100 chi2 calcolo e stampo i pesi 
  YAML::Emitter emitter;
  emitter <<YAML::BeginMap <<  YAML::Key << "Weights" << YAML::Value << YAML::BeginSeq;
  
  for (int i = 0; i < N_rep; i++)
    {
      emitter << YAML::Flow << YAML::BeginMap;
      weight = exp(weight_vect[i] - log(den));
      emitter << YAML::Key << "Weights replica " + std::to_string(i+1) << YAML::Value << weight;
       emitter << YAML::EndMap;
    }
  emitter<< YAML::EndSeq;
  emitter << YAML::EndMap;
  std::ofstream fout(Weights + "weight_for_PDFS.yaml");
  fout << emitter.c_str();
  fout.close();  
  
      // Print total chi2
      //std::cout << "Replica " << irep << ", Total chi2 / Npt = " << chi2->Evaluate() << " (Npt = " << chi2->GetDataPointNumberAfterCuts() << ")" << std::endl;
      //std::cout << "Replica " << irep << ", Total chi2 / Npt = " << chi2->Evaluate() << ", log(weight) = " << lw << std::endl;
     
   // }

  // Delete random-number generator
  gsl_rng_free(rng);

    
    

  t.stop(true);
  return 0;
}
