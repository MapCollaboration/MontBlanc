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

#include <getopt.h>

#include <filesystem>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

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
  
  if (!std::filesystem::exists(Weights)) 
   {
     std::filesystem::create_directory(Weights);
     std::cout << "Folder created successfully." << std::endl;
   }

  YAML::Node config = YAML::LoadFile(InputCardPath);

  // here we define the vector we need

  std::vector<float> predictions;
  std::vector<float> predictions_unc;
  std::vector<float> data;
  std::vector<float> data_unc_uncorr;
  std::vector<std::vector<float>> data_unc_corr;
  std::vector<float> data_unc_corr_raw; 
  std::vector<gsl_matrix*> Cov_blocks;
  std::vector<std::vector<float>> data_blocks;
  std::vector<std::vector<float>> pred_blocks;
  std::vector<float> chi2_vect;
  float chi2;
  float weight;
  std::vector<float> weights;
  int number_data = 0;
  int N_rep = 100;

  //Ciclo sulle repliche, si può fare in parallelo
  for (int i = 0; i < N_rep; i++)
    {
      for (auto const& ds : config["Data"]["sets"])
        {
         
          if(ds["name"].as<std::string>().find("COMPASS") != std::string::npos || 
              ds["name"].as<std::string>().find("HERMES") != std::string::npos)
           { 
             YAML::Node exp = YAML::LoadFile(Predictions + "Predictions_exp " + ds["name"].as<std::string>() +"/Predictions_replica " + std::to_string(i+1) + ".yaml");
             for (auto const& p : exp[ds["name"].as<std::string>()])
               {
                 predictions.push_back(p["unshifted prediction"].as<float>());
               }
             pred_blocks.push_back(predictions);
             predictions.clear();
             // Qui apro i data file per ogni esperimento e raccolgo i punti sperimentali e i relativi errori (separando le inc add e unc)
             YAML::Node unc = YAML::LoadFile(datafolder + ds["file"].as<std::string>());
             for(const auto& val : unc["dependent_variables"])
              {
                for ( const auto& u : val["values"])
                  { 
                    data.push_back(u["value"].as<float>());
                    for (auto const& error : u["errors"])
                      {
                        if(error["label"].as<std::string>() == "unc")
                         { 
                           data_unc_uncorr.push_back(error["value"].as<float>());
                         }
                        if(error["label"].as<std::string>() == "add")
                         { 
                           data_unc_corr_raw.push_back(error["value"].as<float>()*data.back());
                         }
                        if(error["label"].as<std::string>() == "mult")
                         {
                           data_unc_corr_raw.push_back(error["value"].as<float>()*data.back());
                         }
                      }
                    //prima di passare al valore successivo appendo la raw nella matrice con le inc correlate e libero il container della riga 
                    data_unc_corr.push_back(data_unc_corr_raw);
                    data_unc_corr_raw.clear();
                  }    
              }
             // a questo punto appendo al blocco dei dati il vettore contenente i dati dell'esperimento
             data_blocks.push_back(data);
             //libero il vettore che contiene i dati
             data.clear();

             //Ora calcolo la prima parte della matrice di covarianza cioè il quadrato degli errori unc
             for (int j = 0; j < (int) data_unc_uncorr.size(); j++)
               {
                 data_unc_uncorr[j] *= data_unc_uncorr[j];   
               }
         
             //transpose of the matrix che contiene gli errori correlati
             int rows = data_unc_corr.size();
             int columns = data_unc_corr[0].size();
             // inizializziamo matrice trasposta
             std::vector<std::vector<float> > data_unc_corr_transposed(columns, std::vector<float>(rows));
       
             for (int k = 0; k < rows; k++) 
               {
                 for (int j = 0; j < columns; j++)
                   {
                     data_unc_corr_transposed[j][k] = data_unc_corr[k][j];
                   }
               }
        
             // ora calcoliamo il secondo pezzo che è il prodotto della matrice con la sua trasposta
             std::vector<std::vector<float> > data_unc_corr_square(rows, std::vector<float>(rows));

             for (int l = 0; l < rows; l++) 
               {
                 for (int j = 0; j < rows; j++)
                   {
                     for (int k = 0; k < columns; k++)
                       {
                         data_unc_corr_square[l][j] += data_unc_corr[l][k] * data_unc_corr_transposed[k][j] ;
                       }
                   }
               }
              
             gsl_matrix* covariance_matrix = gsl_matrix_alloc(rows, rows); // Allocate a rows x rows matrix

             for (int i = 0; i < rows; i++) 
               {
                 for (int j = 0; j < rows; j++) 
                    {
                      if (i == j)
                        {
                          gsl_matrix_set(covariance_matrix, i, j, (data_unc_uncorr[i] +  data_unc_corr_square[i][j]));
                        }
                      else 
                         {
                           gsl_matrix_set(covariance_matrix, i, j, data_unc_corr_square[i][j]);
                         }
                    }
               } 

              int status;
              status = gsl_linalg_cholesky_decomp(covariance_matrix);
    
              if (status == 0) {
              status = gsl_linalg_cholesky_invert(covariance_matrix);
              }
             
             
             /*//covariance matrix
             Eigen::MatrixXd covariance_matrix = Eigen::MatrixXd::Zero(rows, rows);

             for (int k = 0; k < rows; k++) 
               {
                 for (int j = 0; j < rows; j++)
                   { 
                     if (k == j)
                      {
                        covariance_matrix(k,j) = data_unc_uncorr[k] +  data_unc_corr_square[k][j] ;
                      }
                     else
                        {     
                          covariance_matrix(k,j) = data_unc_corr_square[k][j];
                        }
                   }
               } */
              /*
              std::cout << "il num di righe è " <<  covariance_matrix.rows()<<std::endl;
              std::cout << "il num di cols è " <<  covariance_matrix.cols()<<std::endl;
              for(int k = 0; k < (int) covariance_matrix.rows(); k++)
            {
              for(int j = 0; j < (int) covariance_matrix.cols(); j++) 
                {
                  std::cout <<covariance_matrix(k,j)<<" ";
                }
                std::cout<<std::endl;
            }  
             // calcolo matrice inversa
             Eigen::MatrixXd covariance_matrix_inverse = covariance_matrix.inverse();  
             // inserisco la matrice di cov per l'esp nel blocco
             Cov_blocks.push_back(covariance_matrix_inverse);  */
             Cov_blocks.push_back(covariance_matrix);
             data_unc_uncorr.clear();
             gsl_matrix_free(covariance_matrix);      

             for (auto& innerVector : data_unc_corr_square )
               {
                 innerVector.clear();
               }
             data_unc_corr_square.clear();

             for (auto& innerVector : data_unc_corr_transposed )
               {
                 innerVector.clear();
               }
             data_unc_corr_transposed.clear();

             for (auto& innerVector : data_unc_corr)
               {
                 innerVector.clear();
               }
             data_unc_corr.clear();    
           }
        }  
      //compute chi2
      int n = 0;
      chi2 = 0;
      number_data = 0;
      //per ogni matrice di covarianza del blocco calcolo chi2 e poi sommo
      for(const auto& matrix : Cov_blocks)
        {  
          chi2 = 0;
          data = data_blocks[n];
          predictions = pred_blocks[n];
          number_data += data.size(); 
          int rowss = matrix->size1;
          int colss = matrix->size2;
          n++;
          for(int k = 0; k < rowss; k++)
            {
              for(int j = 0; j < colss; j++) 
                {
                  chi2 += (data[k] - predictions[k]) *  gsl_matrix_get(matrix, k, j) * (data[j] - predictions[j]);
                }
            }
          data.clear();
          predictions.clear();
         
        }
      //std::cout <<"il chi2 è = "<< chi2 << std::endl;  
      chi2_vect.push_back(chi2);

      Cov_blocks.clear();
      data_blocks.clear();
      pred_blocks.clear();       
   } 
  //una volta raccolto i 100 chi2 calcolo e stampo i pesi 
  YAML::Emitter emitter;
  emitter <<  YAML::BeginSeq << YAML::Key << "Weights" << YAML::Value << YAML::BeginMap;
  float den = 0; 
  for(int k = 0 ; k < N_rep; k++)
    {           
      den += pow(chi2_vect[k], 0.5*(number_data - 1)) * exp(- 0.5 * chi2_vect[k]); 
    }
  
  for (int i = 0; i< N_rep; i++)
    {
      weight = pow(chi2_vect[i], 0.5*(number_data - 1)) * exp(- 0.5 * chi2_vect[i]) /((1/N_rep) * den);
      emitter << YAML::Key << "Weights replica " + std::to_string(i+1) << YAML::Value << weight;
    }
  emitter << YAML::EndSeq;
  emitter << YAML::EndMap;
  std::ofstream fout(Weights + "weight_for_PDFS.yaml");
  fout << emitter.c_str();
  fout.close();   
  t.stop(true);
  return 0;
}