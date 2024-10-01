#include "MontBlanc/predictionshandler.h"
#include "MontBlanc/AnalyticChiSquare.h"
// NangaParbat
#include <NangaParbat/chisquare.h>
#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <NangaParbat/direxists.h>

#include <fstream>
#include <getopt.h>
#include <Eigen/Dense>
#include <filesystem>

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
 
  if ((argc - optind) < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder>" << std::endl;
      std::cerr << "Usage: " << argv[1] << " <path to predictions>" << std::endl;
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[optind];

  // Input information
  const std::string InputCardPath = ResultFolder + "/config.yaml";
  const std::string datafolder    = ResultFolder + "/data/";
  const std::string Predictions   = ResultFolder + "/" + argv[optind + 1] + "/";
  const std::string Weights       = ResultFolder + "/Weights/";

  
  
  // Create Weights folder
  if (!std::filesystem::exists(Weights)) 
   {
     std::filesystem::create_directory(Weights);
     std::cout << "Folder created successfully." << std::endl;
   }

  // Load input card
  YAML::Node config = YAML::LoadFile(InputCardPath);

 
  // Define vectors and matrices 

  std::vector<double> weight_vect;                      // vector containing not normalised weights
  std::vector<float> data_unc_uncorr;                   // uncorrelated uncertanties of data
  std::vector<std::vector<float>> data_unc_corr;        // correlated uncertanties of data
  std::vector<std::vector<float>> Q_values_blocks;      // Q_values for each block
  std::vector<std::vector<float>> z_values_low_blocks;  // low values of the bins in z for each block
  std::vector<std::vector<float>> z_values_high_blocks; // high values of the bins in z for each block
  std::vector<Eigen::MatrixXd> Cov_blocks;              // covariance matrix of each experiment
  std::vector<Eigen::VectorXd> data_blocks;             // data for each block
  std::vector<Eigen::VectorXd> data_fluctuated_blocks;  // fluctuated data inside uncerntanties for each block
  std::vector<Eigen::VectorXd> pred_blocks;             // prediction for each block
  std::vector<std::string> file_names;

  // storage vectors for one block
  std::vector<float> Q_values;                          
  std::vector<float> Q_cuts;
  std::vector<float> z_val_low;
  std::vector<float> z_val_high;
  std::vector<float> z_cuts_min;
  std::vector<float> z_cuts_max;
  std::vector<double> dati_fluc;
  
  // temporary variables
  float chi2;
  float weight;
  int number_data;
  const int N_rep = 1000;
  double norm = 0.;
  
  

  //Cycle over PDF replicas
  for (int i = 0; i < N_rep; i++) 
    {
      // Initialise GSL random-number generator
      gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
      gsl_rng_set(rng, config["Data"]["seed"].as<int>());
      number_data = 0; // total number of data for SIDIS

      for (auto const& ds : config["Data"]["sets"])
        {
          // Initialize an Eigen matrix 
          Eigen::MatrixXd corr_err_matrix(1,1);
          Eigen::VectorXd predizioni(1);
          Eigen::VectorXd dati(1);

          // Dimension of columns for the experiment in the correlated error matrix
          int dim_col; // number of correlated uncertanties
          int dim_row; // number of point
          
          if(ds["name"].as<std::string>().find("COMPASS") != std::string::npos || 
              ds["name"].as<std::string>().find("HERMES") != std::string::npos)
           { 
            
            std::string s =  ds["file"].as<std::string>();
            int pos = s.find(".");
            file_names.push_back(s.substr(0, pos));
            YAML::Node exp = YAML::LoadFile(Predictions +"Predictions_" + file_names.back() +"/Predictions_replica_" + std::to_string(i+1) + ".yaml");
            
             // Get info about z and Q cuts ranges and set dimensions for the corr error matrix 
             if(ds["name"].as<std::string>().find("HERMES") != std::string::npos)
               {
                 z_cuts_min.push_back(0.2);
                 z_cuts_max.push_back(0.8);
                 Q_cuts.push_back(2.);
                 dim_col = 54;
                 dim_row = 54;
                 number_data += 54;
               }

             if( ds["name"].as<std::string>().find("COMPASS") != std::string::npos)
               {  
                 z_cuts_min.push_back(0.);
                 z_cuts_max.push_back(0.);
                 Q_cuts.push_back(2.);
                 dim_col = 1;
                 dim_row = 311;
                 number_data += 311;
               }

             //Resize of prediction and data Eigen vectors
             predizioni.resize(dim_row);
             dati.resize(dim_row);
           
             // storage of predictions for one block (experiment)
             int k = 0;
             for (auto const& p : exp[ds["name"].as<std::string>()])
               {
                 predizioni(k) = p["unshifted prediction"].as<float>();
                 k++;
               }
             
             // Add the block to the predictions
             pred_blocks.push_back(predizioni);

             // Resize the matrix of correlated errors
             corr_err_matrix.resize(dim_row, dim_col);

             // Parsing of data and their uncertanties 
             YAML::Node unc = YAML::LoadFile(datafolder + ds["file"].as<std::string>());

             for(const auto& Qz : unc["independent_variables"])
               {
                 if (Qz["header"]["name"].as<std::string>() == "Q2") 
                  {
                    for (const auto& value : Qz["values"]) 
                      {
                        if(ds["name"].as<std::string>().find("HERMES") != std::string::npos)
                         {
                           Q_values.push_back(sqrt(value["low"].as<double>()));
                         }
                        else if(ds["name"].as<std::string>().find("COMPASS") != std::string::npos)
                         {
                           Q_values.push_back(sqrt(value["value"].as<double>()));
                         }
                      }
                  }

                 if (Qz["header"]["name"].as<std::string>() == "z") 
                  {
                    for (const auto& value : Qz["values"]) 
                      {
                        z_val_low.push_back(value["low"].as<double>());
                        z_val_high.push_back(value["high"].as<double>());
                      }
                  }
               }
             
             // Add the block
             Q_values_blocks.push_back(Q_values);
             z_values_low_blocks.push_back(z_val_low);
             z_values_high_blocks.push_back(z_val_high);

             for (const auto& point : unc["dependent_variables"])
               {
                 int i = 0;
                 for ( const auto& val : point["values"])
                   { 
                     dati(i) = val["value"].as<float>();
                     int j = 0;
                     for (auto const& error : val["errors"])
                       {
                         if(error["label"].as<std::string>() == "unc")
                          { 
                            data_unc_uncorr.push_back(error["value"].as<float>());
                          }
                         if(error["label"].as<std::string>() == "add")
                          { 
                            corr_err_matrix(i,j) = error["value"].as<float>() * dati(i);
                            j++;
                          }
                         if(error["label"].as<std::string>() == "mult")
                          {
                            corr_err_matrix(i,j) = error["value"].as<float>() * dati(i);
                            j++;
                          }
                       }
                     i++;
                   }    
               }
             
             // Add data to the block
             data_blocks.push_back(dati);
            
             // Compute the square of uncorrelated errors
             for (int j = 0; j < (int) data_unc_uncorr.size(); j++)
               { 
                 data_unc_uncorr[j] *= data_unc_uncorr[j]; 
               }

             // Evaluate the off-diagonal term of the covariance matrix 
             Eigen::MatrixXd corr_err_matrix_square = corr_err_matrix * corr_err_matrix.transpose();

             //covariance matrix
             Eigen::MatrixXd covariance_matrix = Eigen::MatrixXd::Zero(dim_row, dim_row);

             for(int k = 0; k < dim_row; k++) 
               {
                 for(int j = 0; j < dim_row; j++)
                   { 
                     if(k == j)
                      {
                        covariance_matrix(k,j) = data_unc_uncorr[k] + corr_err_matrix_square(k,j);
                      }
                     else
                        {     
                          covariance_matrix(k,j) =  corr_err_matrix_square(k,j);
                        }
                   }
               } 

             // Define and evaluate the inverse of the covariance matrix
             Eigen::MatrixXd covariance_matrix_inverse = covariance_matrix.inverse();
 
             // Add the inverse covariance matrix to the block
             Cov_blocks.push_back(covariance_matrix_inverse);  
                          
             // Fluctuate dataset
             NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(datafolder + ds["file"].as<std::string>()), rng, 0};

             dati_fluc = DH->GetFluctutatedData();

             Eigen::VectorXd dati_fluctuated(dim_row);

             for (int i = 0; i< dim_row ; i++)
               {
                 dati_fluctuated(i) = dati_fluc[i];
               }      
             data_fluctuated_blocks.push_back(dati_fluctuated);
             delete(DH);

             // Clear some vectors
             data_unc_uncorr.clear();
             Q_values.clear();
             z_val_low.clear();
             z_val_high.clear();
             dati_fluc.clear();               
           }          
        } 
            
      // Compute chi2
      
      int n = 0;          // Cut block iterator
      float tmp = 0.;
      float c2 = 0.;      // Chi2/NumberOfPoint
      chi2 = 0.;      
      //number_data = 838;  // Total number of data for SIDIS
     
       
      // For each block evaluate chi2 and sum 
      for(const auto& matrix : Cov_blocks)
        { 
          Q_values = Q_values_blocks[n];
          z_val_low = z_values_low_blocks[n];
          z_val_high = z_values_high_blocks[n];
          std::string name = file_names[n];
          Eigen::VectorXd dati = data_blocks[n];
          Eigen::VectorXd dati_fluttuati = data_fluctuated_blocks[n];
          Eigen::VectorXd predizioni = pred_blocks[n];
          Eigen::VectorXd dati_post_cut(Q_values.size());
          Eigen::VectorXd predizioni_post_cut(Q_values.size());
                    
          for(int k = 0; k < (int) Q_values.size(); k++) 
            {             
              if( name.find("COMPASS") != std::string::npos)
               {
                 if(Q_values[k] > Q_cuts[n])
                  {                    
                    dati_post_cut(k) = dati_fluttuati(k);                    
                    predizioni_post_cut(k) = predizioni(k);                      
                  }
                 else
                  {
                    dati_post_cut(k) = dati_fluttuati(k);
                    predizioni_post_cut(k) = dati(k);
                  }        
               }

              if( name.find("HERMES") != std::string::npos)
               {                 
                 if((Q_values[k]> Q_cuts[n]) && (z_cuts_min[n]< z_val_low[k] && z_cuts_max[n] > z_val_high[k] ))
                  {                    
                    dati_post_cut(k) = dati_fluttuati(k);                   
                    predizioni_post_cut(k) = predizioni(k);                      
                  }

                 else
                  {                  
                    dati_post_cut(k) = dati_fluttuati(k);                    
                    predizioni_post_cut(k) = dati(k);                   
                  }        
               }           
            }
          
          // Define residual
          Eigen::VectorXd residual = predizioni_post_cut - dati_post_cut;
          
          tmp = ((residual.transpose() * matrix * residual));          
          chi2 += tmp;
          
          // Clear vectors
          Q_values.clear();
          z_val_low.clear();
          z_val_high.clear();
          n++;
          tmp = 0.;
        }
      
      c2 = chi2 / number_data;
  
      const double lw = ( number_data - 1 ) * log(c2*number_data) / 2 - c2*number_data / 2 - 0.5*(number_data - 1) * log(number_data) + log(N_rep);
      
      norm += pow(c2, 0.5*(number_data - 1)) * exp(- 0.5 * c2*number_data); 
      weight_vect.push_back(lw);
      
      // Blocks clear
      Cov_blocks.clear();
      data_blocks.clear();
      pred_blocks.clear(); 
      file_names.clear(); 
      gsl_rng_free(rng);      
    } 

  YAML::Emitter emitter;
  emitter <<YAML::BeginMap <<  YAML::Key << "Weights" << YAML::Value << YAML::BeginSeq;
  
  for (int i = 0; i < N_rep; i++)
    {
      emitter << YAML::Flow << YAML::BeginMap;
      weight = exp(weight_vect[i] - log(norm));
      emitter << YAML::Key << "Weights replica " + std::to_string(i+1) << YAML::Value << weight;
      emitter << YAML::EndMap;
    }
  emitter<< YAML::EndSeq;
  emitter << YAML::EndMap;
  std::ofstream fout(Weights + "weight_for_PDFS.yaml");
  fout << emitter.c_str();
  fout.close(); 
  
  t.stop(true);
  
  return 0;
}