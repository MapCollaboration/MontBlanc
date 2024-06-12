//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//          Hakim Khoudli: hakim.khoudli@polytechnique.org
//

#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"

#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <fstream>

#include <getopt.h>

std::string GetCurrentWorkingDir()
{
  char buff[FILENAME_MAX];
  getcwd(buff, FILENAME_MAX);
  return buff;
}

int main(int argc, char *argv[])
{
  //Command line options handling
  const char* const short_opts = "u";
  const option long_opts[] =
  {
    {"force_uncertainties", no_argument, nullptr, 'u'}, //Force the computation of uncertainties even if the PDF set is not LHAPDFSet
  };

  bool force_uncertainties = false;

  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

      if (opt == -1)
        break;

      switch (opt)
        {
        case 'u':
          force_uncertainties = true;
          break;
        case '?': // Unrecognized option
        default: // Unhandled option
          std::cerr << "Usage: " << argv[0] << " [-u|--force_uncertainties] <path to fit folder> [<set name> (default: LHAPDFSet)]" << std::endl;
          exit(-1);
        }
    }

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
  //const std::string OutputFile    = ResultFolder + "/Predictions.yaml";
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
  //if (config["Predictions"]["pdfset"]["member"]) 
    //{
      //const int replica = config["Predictions"]["pdfset"]["member"].as<int>();
    //}
  //config["Predictions"]["pdfset"]["member"] = replica;

  // APFEL++ x-space grid
  std::vector<apfel::SubGrid> vsg;
  for (auto const& sg : config["Predictions"]["xgrid"])
    vsg.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> g(new const apfel::Grid{vsg});
 
  // Create yaml file for each experiments
  for (auto const& dn : config["Data"]["sets"])
    {
      if(dn["name"].as<std::string>().find("COMPASS") != std::string::npos || 
              dn["name"].as<std::string>().find("HERMES") != std::string::npos)
       {
         const std::string OutputFile= ResultFolder + "/Pred_" +  dn["name"].as<std::string>() + ".yaml";
         std::ofstream fout(OutputFile);
         YAML::Node node;
         node["exp"] = dn["name"].as<std::string>();
         fout << node;
         fout.close();
       }
    }

  std::string pdfset = config["Predictions"]["pdfset"]["name"].as<std::string>();
  //const std::vector<LHAPDF::PDF*> PDFs = set.mkPDFs();
  const std::vector<LHAPDF::PDF*> PDFs = LHAPDF::mkPDFs(pdfset);
  const int n_rep = PDFs.size() - 1;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  for (int jrep = 1; jrep <= n_rep; jrep++)
    { 
      config["Predictions"]["pdfset"]["member"] = jrep;
      
      // Run over the data set and gather pairs of DatHandler and
      // PredictionHandler pairs. Do not impose cuts to compute
      // predictions for all available points.
      std::cout << "Computing tables..." << std::endl;
      std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect;
      for (auto const& ds : config["Data"]["sets"])
        {
          if(ds["name"].as<std::string>().find("COMPASS") != std::string::npos || 
              ds["name"].as<std::string>().find("HERMES") != std::string::npos)
           {
             std::cout << "#Experiment = " << ds["name"].as<std::string>() << std::endl;
             // Dataset
             NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(datafolder + ds["file"].as<std::string>())};

             // Add block to the chi2
             DSVect.push_back(std::make_pair(DH, new MontBlanc::PredictionsHandler{config["Predictions"], *DH, g}));
           }
          else continue;
        }


      // Run over the experiments, compute central values and standard
      // deviations (of the shifted predictions) over the replicas. Save
      // results in a YAML:Emitter.
      std::cout << "\nComputing predictions..." << std::endl;
      //YAML::Emitter emitter;
      //emitter << YAML::BeginSeq;
      for (int iexp = 0; iexp < (int) DSVect.size(); iexp++)
        {
          const std::string OutputFile= ResultFolder + "/Pred_" + DSVect[iexp].first->GetName() + ".yaml";
          std::cout << "#Experiment = " << DSVect[iexp].first->GetName() << std::endl;

          // Get experimental central values and uncorrelated ucertainties
          const std::vector<double> mvs = DSVect[iexp].first->GetMeanValues();
          const std::vector<double> unc = DSVect[iexp].first->GetUncorrelatedUnc();

          // Get binning
          const std::vector<NangaParbat::DataHandler::Binning> bins = DSVect[iexp].first->GetBinning();

          // Initialise averages
          std::vector<double> av(bins.size(), 0);
          std::vector<double> std(bins.size(), 0);
          std::vector<double> avor(bins.size(), 0);
          std::vector<double> stdor(bins.size(), 0);

          // Get LHAPDF set
          const std::vector<LHAPDF::PDF*> sets = LHAPDF::mkPDFs(LHAPDFSet);

          // If the default "LHAPDFSet" we know it is a Monte Carlo set
          // thus compute central values and uncertanties as averages and
          // standard deviations...
          if (LHAPDFSet == "LHAPDFSet" || force_uncertainties)
           {
             std::vector<double> av2(bins.size(), 0);
             std::vector<double> avor2(bins.size(), 0);
             // Run over replicas
             const int nrep = sets.size() - 1;
             for (int irep = 1; irep <= nrep; irep++)
               {
                 // Construct chi2 object with the irep-th replica
                 MontBlanc::AnalyticChiSquare chi2{DSVect, new MontBlanc::LHAPDFparameterisation{sets, g, irep}};
                 const std::vector<double> prds = DSVect[iexp].second->GetPredictions([](double const &, double const &, double const &) -> double { return 0; });

                 const std::pair<std::vector<double>, double> shifts = chi2.GetSystematicShifts(iexp);
                 for (int i = 0; i < (int) bins.size(); i++)
                   {
                     av[i] += ( prds[i] + shifts.first[i] ) / nrep;
                     av2[i] += pow(prds[i] + shifts.first[i], 2) / nrep;
                     avor[i] += prds[i] / nrep;
                     avor2[i] += pow(prds[i], 2) / nrep;
                   }
               }
             for (int i = 0; i < (int) bins.size(); i++)
               {
                 std[i] = sqrt(av2[i] - pow(av[i], 2));
                 stdor[i] = sqrt(avor2[i] - pow(avor[i], 2));
               }
           }
          // ...otherwise only compute predictions for member 0 with no
          // uncertainty.
          else
            {
              // Construct chi2 object with the 0-th replica
              MontBlanc::AnalyticChiSquare chi2{DSVect, new MontBlanc::LHAPDFparameterisation{sets, g, 0}};
              const std::vector<double> prds = DSVect[iexp].second->GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
              const std::pair<std::vector<double>, double> shifts = chi2.GetSystematicShifts(iexp);
              for (int i = 0; i < (int) bins.size(); i++)
                {
                  av[i] += prds[i] + shifts.first[i];
                  avor[i] += prds[i];
                }
            }

          //emitter << YAML::BeginMap << YAML::Key << DSVect[iexp].first->GetName() << YAML::Value << YAML::BeginSeq;
          
          YAML::Node node = YAML::LoadFile(OutputFile);
          node["replica" +  std::to_string(jrep)] =  YAML::Node(YAML::NodeType::Sequence);
          for (int i = 0; i < (int) bins.size(); i++)
            {
              const NangaParbat::DataHandler::Binning b = bins[i];
              YAML::Node subnode;
              subnode =  YAML::Node(YAML::NodeType::Map);
              subnode["index"] = i;
              //subnode["Qmin"] = b.Qmin;
              //subnode["Qmax"] = b.Qmax;
              //subnode["yav"] = b.yav;
              //subnode["ymin"] = b.ymin;
              //subnode["ymax"] = b.ymax;
              //subnode["xav"] = b.xav;
              //subnode["xmin"] = b.xmin;
              //subnode["xmax"] = b.xmax;
              //subnode["zav"] = b.zav;
              //subnode["zmin"] = b.zmin;
              //subnode["zmax"] = b.zmax;
              //subnode["exp. central values"] = mvs[i];
              //subnode["exp. unc."] = unc[i];
              subnode["prediction"] = av[i];
              subnode["pred. unc."] = std[i];
              subnode["unshifted prediction"] = avor[i];
              subnode["unshifted pred. unc."] = stdor[i];
              subnode.SetStyle(YAML::EmitterStyle::Flow);
              node["replica" +  std::to_string(jrep)].push_back(subnode);
              std::ofstream fout(OutputFile);
              fout << node;
              fout.close();
              //emitter << YAML::Flow << YAML::BeginMap;
              //emitter << YAML::Key << "index" << YAML::Value << i;
              //emitter << YAML::Key << "Qav" << YAML::Value << b.Qav << YAML::Key << "Qmin" << YAML::Value << b.Qmin << YAML::Key << "Qmax" << YAML::Value << b.Qmax;
              //emitter << YAML::Key << "yav" << YAML::Value << b.yav << YAML::Key << "ymin" << YAML::Value << b.ymin << YAML::Key << "ymax" << YAML::Value << b.ymax;
              //emitter << YAML::Key << "xav" << YAML::Value << b.xav << YAML::Key << "xmin" << YAML::Value << b.xmin << YAML::Key << "xmax" << YAML::Value << b.xmax;
              //emitter << YAML::Key << "zav" << YAML::Value << b.zav << YAML::Key << "zmin" << YAML::Value << b.zmin << YAML::Key << "zmax" << YAML::Value << b.zmax;
              //emitter << YAML::Key << "exp. central value" << YAML::Value << mvs[i] << YAML::Key << "exp. unc." << YAML::Value << unc[i];
              //emitter << YAML::Key << "prediction" << YAML::Value << av[i] << YAML::Key << "pred. unc." << YAML::Value << std[i];
              //emitter << YAML::Key << "unshifted prediction" << YAML::Value << avor[i] << YAML::Key << "unshifted pred. unc." << YAML::Value << stdor[i];
              //emitter << YAML::EndMap;
           }
           node.reset();
          //emitter << YAML::EndSeq;
          //emitter << YAML::EndMap;
       }
      //emitter << YAML::EndSeq;

      // Print YAML:Emitter to file
      //std::ofstream fout(OutputFile);
      //fout << emitter.c_str();
      //fout.close();
   }
  t.stop(true);
  return 0;
}
