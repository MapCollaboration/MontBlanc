#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
#include "MontBlanc/LHAPDFparameterisation.h"
#include <Eigen/Dense>

#include <fstream>
#include <algorithm>
#include <random>   
#include <getopt.h>

#include <filesystem>
#include <sstream>

#include "LHAPDF/Paths.h"
#include "LHAPDF/LHAPDF.h"

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
  const std::string InputCardPath = ResultFolder + "/config.yaml";
  const std::string BestParameter = ResultFolder + "/BestParameters.yaml";
  const std::string Unweighting_files = ResultFolder + "/Unweighting_files/";
  const std::string UnweightedSet = ResultFolder + "/UnweightedSet/";
  const int N_rep_new = std::stoi(argv[optind + 2]);

   // Name of the set
  std::string hadron = "PIp";
  
 
  if (!std::filesystem::exists(Unweighting_files)) 
   {
     std::filesystem::create_directory(Unweighting_files);
     std::cout << "Folder created successfully." << std::endl;
   }
  
  if (!std::filesystem::exists(UnweightedSet)) 
   {
     std::filesystem::create_directory(UnweightedSet);
     std::cout << "Folder created successfully." << std::endl;
   }
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

  // Define vectors and pairs
  std::vector<float> cum_prob;
  std::vector<double> weight_vect;
  std::vector<double> weight_new_vect;
  std::vector<std::pair< int ,float>> cum_prob_k;
  std::vector<std::pair< int ,double>> w_k;
  std::vector<std::pair< int ,double>> w_k_unshuffled;
  std::vector<std::pair< int ,double>> w_k_new;

  int N_rep;
  int i = 0;

  YAML::Node W = YAML::LoadFile(Weights);
  
  for(auto const& rep: W["Weights"])
    {
      i++;
      w_k.push_back(std::make_pair(i, rep["Weights replica " + std::to_string(i)].as<double>()));
      w_k_unshuffled.push_back(std::make_pair(i, rep["Weights replica " + std::to_string(i)].as<double>()));      
    }
    
  // Check the normalization and evaluate N_eff for the reweighted set
  N_rep = w_k.size();
  int N_eff_w = 0;
  float N_eff_sum = 0.;
  double w_sum = 0.;
  for(int i = 0; i< N_rep; i++)
    {
      w_sum += w_k[i].second;
      N_eff_sum += w_k[i].second * log((float) N_rep / w_k[i].second);
    }
  
  N_eff_w = (int) exp(N_eff_sum / (float) N_rep);

  // Shuffle vector
  std::default_random_engine e(config["Data"]["seed"].as<int>());
  shuffle(w_k.begin(), w_k.end(), e); 
  
  // Compute cumulative probability
  cum_prob_k.push_back(std::make_pair(0, 0.));
  cum_prob.push_back(0.);

  for(int j = 1; j<= N_rep; j++)
    {      
      cum_prob.push_back(cum_prob[j-1] + w_k[j-1].second/(float) N_rep);
      cum_prob_k.push_back(std::make_pair(w_k[j-1].first, cum_prob[j]));
    }

  // Compute Shannon Entropy 
  std::vector<float> H_R;
  
  float h_r = 0;
  for(int N_r = 1; N_r <= N_rep_new; N_r++ )
    {
      for(int k = 1; k<= N_rep; k++)
        {
           weight_new_vect.push_back(0.); 

           for(int j = 1; j<= N_r; j++)
             {
               weight_new_vect[k-1] += ((j/(float)N_r - cum_prob_k[k-1].second) >= 0. ? 1. : 0.) * ((cum_prob_k[k].second - j/(float)N_r) >= 0. ? 1. : 0.);
             }
   
           w_k_new.push_back(std::make_pair(cum_prob_k[k].first, weight_new_vect[k-1]));
        }
      
      for(int k = 0; k < N_rep; k++)
        {
          h_r += (w_k_new[k].second / N_r) * (log(w_k_new[k].second / N_r) - log(w_k[k].second / N_rep));
          
        }
      
      H_R.push_back(h_r);
      h_r = 0.;
    }

  // Choose N_eff for the unweighted set
  int N_eff = (N_eff_w / 10) * 10 ;
    
  // Clear before evaluating new weights
  weight_new_vect.clear();
  w_k_new.clear();
 
  // Compute new weights
  for(int k = 1; k<= N_rep; k++)
    {
      weight_new_vect.push_back(0.); 
      for(int j = 1; j<= N_eff; j++)
        {
          weight_new_vect[k-1] += ((j/(float)N_eff - cum_prob_k[k-1].second) >= 0. ? 1. : 0.) * ((cum_prob_k[k].second - j/(float)N_eff) >= 0. ? 1. : 0.);
        }
   
      w_k_new.push_back(std::make_pair(cum_prob_k[k].first, weight_new_vect[k-1]));
    }

  // Check the normalization
  double w_sum_new = 0.;
  for(int i = 0; i< (int) weight_new_vect.size(); i++)
    {
      w_sum_new += w_k_new[i].second;
    }
  
  // Print yaml file with replicas and associated weights
  YAML::Emitter emitter_1;
  emitter_1 <<YAML::BeginMap <<  YAML::Key << "PDF members" << YAML::Value << YAML::BeginSeq;
  
  for (int i = 0; i < N_rep; i++)
    {
      emitter_1 << YAML::Flow << YAML::BeginMap;
      emitter_1 << YAML::Key << "replica" << YAML::Value << w_k_new[i].first;
      emitter_1 << YAML::Key << "weight" << YAML::Value << (int) w_k_new[i].second;
      emitter_1 << YAML::EndMap;
    }
  emitter_1 << YAML::EndSeq;
  emitter_1 << YAML::EndMap;
  std::ofstream fout_1(Unweighting_files + "unweighting.yaml");
  fout_1 << emitter_1.c_str();
  fout_1.close(); 
    
  // Build Unweighted PDF set  
  int r=1;

  for(int k = 1; k <= N_rep; k++)
    {
      if(w_k_new[k - 1].second != 0)
       {          
         for(int n = 0; n < (int) w_k_new[k - 1].second; n++)
           {
             std::stringstream ds;
             ds << std::setw(4) << std::setfill('0') << r;
             std::string path_to_pdf = LHAPDF::findpdfmempath (config["Predictions"]["pdfset"]["name"].as<std::string>(),w_k_new[k - 1].first );
             
             std::ifstream sourceFile(path_to_pdf, std::ios::binary);
             std::ofstream destFile(UnweightedSet +"UnweightedSet" +"_"+ ds.str() + ".dat", std::ios::binary);
   
             destFile << sourceFile.rdbuf();

             sourceFile.close();
             destFile.close();

             if (r == 1)
              {
                std::string path_to_info = LHAPDF::findpdfsetinfopath(config["Predictions"]["pdfset"]["name"].as<std::string>());
                std::string line;
                std::fstream infofile(path_to_info, std::ios::in | std::ios::out);
                std::ofstream destFile("temp.txt"); // Create a temporary file

                while (std::getline(infofile, line)) 
                  {
                    if (line.find("SetDesc:") != std::string::npos) 
                     {
                       destFile << "SetDesc: " << "NNPDF3.1 NLO global fit (perturbative charm), alphas(MZ)=0.118. mem=0 => average on replicas; mem=1-" + std::to_string(N_eff) +" => PDF replicas"<<std::endl;
                     }
                     else if (line.find("Authors:") != std::string::npos) 
                     {
                       destFile << "Authors: " << "L. Canzian"<< std::endl;  // Write the modified line to temp file
                     } 
                    else if (line.find("NumMembers:") != std::string::npos) 
                     {
                       destFile << "NumMembers: " << N_eff + 1 << std::endl;  // Write the modified line to temp file
                     } 
                    else 
                     {
                       destFile << line << std::endl;  // Copy the original line to temp file
                     }
                  }

                infofile.close();
                destFile.close();

                std::remove((UnweightedSet +"UnweightedSet" + ".info").c_str());  // Remove the original file
                std::rename("temp.txt", (UnweightedSet +"UnweightedSet" + ".info").c_str());  // Rename the temp file to the original file
           
              } 
         
             r++;
           }
        }
    } 
  
  // Build replica 0 of the new PDF set
  
  std::vector<int> skipline;
  int s = 0;
  std::ofstream out;
  out << std::scientific;
  std::string filename = ResultFolder + "/" + "UnweightedSet" + "/" + "UnweightedSet" + "_0000";
  out.open(filename + ".dat");
  
  out << "PdfType: central\n";
  
  out << "Format: lhagrid1\n";
 
  while(true)
  {
    out << "---\n";

    // Write x-grid and Q-grid
    std::fstream file(UnweightedSet + "UnweightedSet" +"_0001" + ".dat", std::ios::in | std::ios::out);
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);
                    
    int i;

    // Array of distributions
   
    std::vector<double> Thresholds = config["Predictions"]["thresholds"].as<std::vector<double>>();
    const int nd = 2 * Thresholds.size() + 1;
    i = 0;
    while(std::getline(file, line) && s == 0)
      {
        if(line.empty())
            break;
        i++;
        if(line == "---")
         {
          skipline.push_back(i-4);
          i = 0;          
         }
      }

    file.close();
    //file.seekg(0);
    std::fstream file1(UnweightedSet + "UnweightedSet" +"_0001" + ".dat", std::ios::in | std::ios::out);
    std::getline(file1, line);
    std::getline(file1, line);
    std::getline(file1, line);
     
    const int np = skipline[s];
     
    Eigen::MatrixXd dist =  Eigen::MatrixXd::Zero(np, nd);
    int sum = std::accumulate(skipline.begin(), skipline.begin() + s , 0);   
    int dum;
    dum = (s == 0 ? 0 : sum + 4*s );
    

    for (int i = 0; i < dum ; i++) 
          {
            std::string dummy;
            std::getline(file1, dummy);
          }
    
    i = 0;
   
    while (std::getline(file1, line))
      {
        std::istringstream iss(line);
        double num;
        while (iss >> num)  
          out << num << " ";
            
        out << "\n";
        if(i == 1)
          break;
        i++;
      }  
    
    // Write flavour indices
    for (int i = - (int) Thresholds.size(); i <= (int) Thresholds.size(); i++)
      out << (i == 0 ? 21 : i) << " ";
    out << "\n";
        
    file1.close();
    
    // Gather tabulated distributions
    
    for(int k=1; k<= N_eff ; k++)
      {        
        std::stringstream rp;
        rp << std::setw(4) << std::setfill('0') << k;
    
        std::ifstream file1(UnweightedSet + "UnweightedSet"  + "_"+ rp.str() + ".dat");
        std::getline(file1, line);
        std::getline(file1, line);
        
        std::string line1;
        
        int skip = (s == 0 ? 4 : sum + 4*(s + 1) );
        
        for (int i = 0; i < skip ; i++) 
          {
            std::string dummy;
            std::getline(file1, line1);
          }

        int j;
        i = 0;
        while (std::getline(file1, line1) && i < np) 
          {
            std::istringstream iss1(line1);
            j = 0;
            double num1;
            while (iss1 >> num1) 
              {                
                dist(i,j) += num1 / N_eff;
                j++;                               
              }             
            i++;                        
          }        
        file1.close();
      }
    
    // Print distributions
    for (int jp = 0; jp < np; jp++)
      {
        for (int jd = 0; jd < nd; jd++)
          if ( dist(jp, jd) < 0.)
            out <<" "<< dist(jp,jd);
          else
            out <<"  "<< dist(jp,jd);
        out << "\n";
      }
        
    if (skipline[s] == skipline.back())
      break;
    s++;         
  }

    out << "---\n";
    out.close();
     
// Delete random-number generator
  gsl_rng_free(rng);

t.stop(true);
  
  return 0;
}