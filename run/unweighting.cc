#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"
#include <Eigen/Dense>
// NangaParbat
#include <NangaParbat/chisquare.h>
#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <NangaParbat/direxists.h>

#include <fstream>
#include <algorithm>
#include <random>   
#include <getopt.h>

#include <filesystem>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <plot.h>
#include <stdio.h>
#include <stdlib.h>

std::string GetCurrentWorkingDir()
{
  char buff[FILENAME_MAX];
  getcwd(buff, FILENAME_MAX);
  return buff;
}

int main(int argc, char *argv[])
{

  // Timer
  apfel::Timer t;
 
  if ((argc - optind) < 3)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder>" << std::endl;
      std::cerr << "Usage: " << argv[1] << " <path to weights>" << std::endl;
      std::cerr << "Usage: " << argv[2] << " <number of unweighted replicas>" << std::endl;
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[optind];
  
  // Input information
  const std::string Weights = ResultFolder + argv[optind + 1] + "weight_for_PDFS.yaml";
  const int N_rep_new = std::stoi(argv[optind + 2]);
  const std::string InputCardPath = ResultFolder + "/config.yaml";
  const std::string BestParameter = ResultFolder + "/BestParameters.yaml";
  const std::string Unweighted_set = ResultFolder + "/Unweighted_set/";
 
  
  if (!std::filesystem::exists(Unweighted_set)) 
   {
     std::filesystem::create_directory(Unweighted_set);
     std::cout << "Folder created successfully." << std::endl;
   }

   YAML::Node config = YAML::LoadFile(InputCardPath);

 // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

//std::vector<double> weight_vect;
std::vector<float> cum_prob;
std::vector<double> weight_vect;
std::vector<double> weight_new_vect;
std::vector<std::pair< int ,float>> cum_prob_k;
std::vector<std::pair< int ,double>> w_k;
std::vector<std::pair< int ,double>> w_k_new;

int N_rep;
YAML::Node W = YAML::LoadFile(Weights);
int i = 0;
for(auto const& rep: W["Weights"])
  {
    i++;
    w_k.push_back(std::make_pair(i,rep["Weights replica " + std::to_string(i)].as<double>()));
  }

//Check the normalization
N_rep = w_k.size();
double w_sum = 0.;
for(int i = 0; i< N_rep; i++)
  {
    w_sum += w_k[i].second;
  }
  std::cout<<"La somma dei pesi è "<<w_sum<<std::endl;


// shuffle vector
std::default_random_engine e(config["Data"]["seed"].as<int>());
shuffle(w_k.begin(), w_k.end(), e); 
  
//compute cumulative probability
cum_prob_k.push_back(std::make_pair(0, 0.));
cum_prob.push_back(0.);

for(int j = 1; j<= N_rep; j++)
  {
    cum_prob.push_back(0.);
    cum_prob[j] = cum_prob[j-1] + w_k[j-1].second/(float) N_rep;
    cum_prob_k.push_back(std::make_pair(w_k[j-1].first, cum_prob[j]));
  }
 
//compute new weights
for(int k = 1; k<= N_rep; k++)
{
   weight_new_vect.push_back(0.); 

for(int j = 1; j<= N_rep_new; j++)
  {
    weight_new_vect[k-1] += ((j/(float)N_rep_new - cum_prob_k[k-1].second) > 0. ? 1. : 0.) * ((cum_prob_k[k].second - j/(float)N_rep_new) > 0. ? 1. : 0.);
  }
   
   w_k_new.push_back(std::make_pair(cum_prob_k[k].first, weight_new_vect[k-1]));

}

//Check the normalization

double w_sum_new = 0.;
for(int i = 0; i< (int) weight_new_vect.size(); i++)
  {
    w_sum_new += w_k_new[i].second;
  }
  std::cout<<"La somma dei nuovi pesi è "<<w_sum_new<<std::endl;

YAML::Emitter emitter;
  emitter <<YAML::BeginMap <<  YAML::Key << "PDF members" << YAML::Value << YAML::BeginSeq;
  
  for (int i = 0; i < N_rep_new; i++)
    {
      emitter << YAML::Flow << YAML::BeginMap;
      emitter << YAML::Key << "replica" << YAML::Value << w_k_new[i].first;
      emitter << YAML::Key << "weight" << YAML::Value << (int) w_k_new[i].second;
      emitter << YAML::EndMap;
    }
  emitter<< YAML::EndSeq;
  emitter << YAML::EndMap;
  std::ofstream fout(Unweighted_set + "unweighting.yaml");
  fout << emitter.c_str();
  fout.close(); 

  YAML::Node Parameters = YAML::LoadFile(BestParameter);
  YAML::Emitter emitter_2;
   emitter_2 << YAML::BeginSeq;
      
  for(int i= 0; i < N_rep_new; i++)
    { 
      for(auto const& val: Parameters)
        {
          if(w_k_new[i].first == val["replica"].as<int>())
            {
              for(int j = 0; j<(int) w_k_new[i].second; j++)
                {
             
              emitter_2 << YAML::Flow << YAML::BeginMap;
                emitter_2 << YAML::Key << "replica" << YAML::Value << val["replica"];
              emitter_2 << YAML::Key << "PDFmember"<< YAML::Value << val["PDFmember"];
              emitter_2 << YAML::Key << "total chi2" << YAML::Value <<val["total chi2"];
               emitter_2 << YAML::Key << "training chi2" << YAML::Value << val["training chi2"];
              emitter_2 << YAML::Key << "validation chi2" << YAML::Value << val["validation chi2"];
                emitter_2 << YAML::Key << "best iteration" << YAML::Value <<val["best iteration"];
                 emitter_2 << YAML::Key << "best validation chi2" << YAML::Value << val["best validation chi2"];
                  emitter_2 << YAML::Key << "parameters" << YAML::Value << YAML::Flow << val["parameters"];
                 emitter_2 << YAML::EndMap;
                
                 //emitter_2 << YAML::Newline;

                }

            }
        }
    }
     emitter_2 << YAML::EndSeq;
  std::ofstream fout_2( ResultFolder + "/UnweightingBestParameters.yaml");
  fout_2 << emitter_2.c_str();
  fout_2.close(); 


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Lets try to plot the PDFs
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 // Include new search path in LHAPDF
  LHAPDF::setPaths(GetCurrentWorkingDir() + "/" + ResultFolder + "/");

   // Include new search path in LHAPDF
  //LHAPDF::setPaths(GetCurrentWorkingDir() + "/" + ResultFolder + ");

 const std::vector<LHAPDF::PDF*> w_sets = LHAPDF::mkPDFs(config["Predictions"]["pdfset"]["name"].as<std::string>());
 const std::vector<LHAPDF::PDF*> unw_sets = LHAPDF::mkPDFs("UnweightedSet");
 float Q = 2.45;
 std::vector<double> w_pdf;
 std::vector<double> unw_pdf;
   int k = 0;
   for(int x = 0 ; x<=1; x+= 0.02 )
   {
      w_pdf.push_back(0.);
      unw_pdf.push_back(0.);
 for (int irep = 0; irep < (int) w_k.size(); irep++)
    {
       w_pdf[k] += w_k[irep].second * w_sets[w_k[irep].first]->xfxQ(2,x,Q);
      unw_pdf[k] += unw_sets[w_k[irep].first]->xfxQ(2, x, Q);
    }
      w_pdf[k] /= 100.;
      unw_pdf[k] /= 100.;
      k++;  
    }

  


// Delete random-number generator
  gsl_rng_free(rng);

t.stop(true);
  
  return 0;
}