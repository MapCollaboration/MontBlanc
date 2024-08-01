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

gsl_matrix *invert_a_matrix(gsl_matrix *matrix, int);
void print_mat_contents(gsl_matrix *matrix, int);

//Inverse of the covariance matrix with gsl
gsl_matrix *invert_a_matrix(gsl_matrix *matrix, int size)
{
    gsl_permutation *p = gsl_permutation_alloc(size);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_matrix *inv = gsl_matrix_alloc(size, size);
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);

    return inv;
}

//Print a gsl matrix
void print_mat_contents(gsl_matrix *matrix, int size)
{
    size_t i, j;
    double element;

    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            element = gsl_matrix_get(matrix, i, j);
            printf("%f ", element);
        }
        printf("\n");
    }
}
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

  // here we define the vector and matrices we need

  std::vector<float> predictions;
  std::vector<float> data;
  std::vector<float> data_unc_uncorr;  // uncorrelated uncertanties of data
  std::vector<std::vector<float>> data_unc_corr; //correlated uncertanties of data
  std::vector<float> data_unc_corr_row; 
  std::vector<Eigen::MatrixXd> Cov_blocks; // covariance matrix of each experiment
  std::vector<std::vector<float>> data_blocks; // data for each experiment
  std::vector<std::vector<float>> pred_blocks; // prediction for each experiment
  std::vector<std::vector<float>> Q_values_blocks; // Q_values to introduce cinematic cuts
  std::vector<std::vector<float>> z_values_blocks; // z_values to introduce cinematic cuts
  std::vector<float> chi2_vect; // chi2 for each experiment
  std::vector<float> Q_values;
  std::vector<float> Q_cuts;
  std::vector<float> z_values;
  std::vector<float> z_cuts_min;
  std::vector<float> z_cuts_max;
  std::vector<std::string> exp_name;

  // temporary variables
  float chi2;
  float chi2_test;
  float weight;
  int number_data;
  int N_rep = 100;
  
  //Ciclo sulle repliche, si può fare in parallelo
  for (int i = 0; i < N_rep; i++) //per ora lo faccio solo su una replica
    {
      
      for (auto const& ds : config["Data"]["sets"])
        {
          // initialize an eigen matrix 
          Eigen::MatrixXd corr_err_matrix(1,1);
          
          // dimension of columns for the experiment in the correlated error matrix
          int dim_c;

          if(ds["name"].as<std::string>().find("COMPASS") != std::string::npos || 
              ds["name"].as<std::string>().find("HERMES") != std::string::npos)
           { 
             YAML::Node exp = YAML::LoadFile(Predictions + "Predictions_exp " + ds["name"].as<std::string>() +"/Predictions_replica " + std::to_string(0) + ".yaml");
             
             // storage of predictions 
             for (auto const& p : exp[ds["name"].as<std::string>()])
               {
                 predictions.push_back(p["unshifted prediction"].as<float>());
               }
             pred_blocks.push_back(predictions);
             predictions.clear();

             // set experiment name
             exp_name.push_back(ds["name"].as<std::string>());

             // Get info about z and Q cuts range and set dimension for the corr error matrix 
             if(ds["name"].as<std::string>().find("HERMES") != std::string::npos)
               {
                 z_cuts_min.push_back(0.2);
                 z_cuts_max.push_back(0.8);
                 Q_cuts.push_back(2.);
                 dim_c = 54;
               }
             if( ds["name"].as<std::string>().find("COMPASS") != std::string::npos)
               {  
                 z_cuts_min.push_back(0.);
                 z_cuts_max.push_back(0.);
                 Q_cuts.push_back(2.);
                 dim_c = 1;
               } 

             // Number of rows for corr error matrix  
             int dim = pred_blocks.back().size();

             // Resize the matrix
             corr_err_matrix.resize(dim, dim_c);

             // Qui apro i data file per ogni esperimento e raccolgo i punti sperimentali e i relativi errori (separando le inc add e unc)
             YAML::Node unc = YAML::LoadFile(datafolder + ds["file"].as<std::string>());
             for(const auto& Q : unc["independent_variables"])
               {
                 if (Q["header"]["name"].as<std::string>() == "Q2") 
                 {
                   for (const auto& value : Q["values"]) 
                     {
                       Q_values.push_back(sqrt(value["value"].as<double>()));
                     }
                 }
                 if (Q["header"]["name"].as<std::string>() == "z") 
                 {
                   for (const auto& value : Q["values"]) 
                     {
                       z_values.push_back(value["value"].as<double>());
                     }
                 }
               }

             Q_values_blocks.push_back(Q_values);
             z_values_blocks.push_back(z_values);

             for(const auto& val : unc["dependent_variables"])
               {
                 int i = 0;
                 for ( const auto& u : val["values"])
                   { 
                     data.push_back(u["value"].as<float>());
                     int j = 0;
                     for (auto const& error : u["errors"])
                       {
                         if(error["label"].as<std::string>() == "unc")
                          { 
                            data_unc_uncorr.push_back(error["value"].as<float>());
                          }
                         if(error["label"].as<std::string>() == "add")
                          { 
                            corr_err_matrix(i,j) = error["value"].as<float>() * data.back();
                            j++;
                          }
                         if(error["label"].as<std::string>() == "mult")
                          {
                            corr_err_matrix(i,j) = error["value"].as<float>() * data.back();
                            j++;
                          }
                       }
                     i++;
                   }    
               }
             
             // A questo punto appendo al blocco dei dati il vettore contenente i dati dell'esperimento
             data_blocks.push_back(data);
            
             //libero il vettore che contiene i dati
             data.clear();

             int rows = corr_err_matrix.rows();
             int columns = corr_err_matrix.cols();
            
             //Ora calcolo la prima parte della matrice di covarianza cioè il quadrato degli errori unc
             for (int j = 0; j < (int) data_unc_uncorr.size(); j++)
               {
                 data_unc_uncorr[j] *= data_unc_uncorr[j]; 
                 //for(int i = 0 ; i < columns ; i++)
                   //{
                     //data_unc_uncorr[j] += corr_err_matrix(j,i);
                  // }
               }

             // Evaluate the of diagonal term of the covariance matrix 
             Eigen::MatrixXd corr_err_matrix_square = corr_err_matrix * corr_err_matrix.transpose();

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
             
             gsl_matrix *cov_mat = gsl_matrix_alloc(rows, rows);
            
             
             for (int k = 0; k < rows; k++) 
               {
                 for (int j = 0; j < rows; j++)
                   { 
                     if (k == j)
                      {
                        gsl_matrix_set( cov_mat, k, j, data_unc_uncorr[k]) ;
                      }
                     else
                        {     
                          gsl_matrix_set( cov_mat, k, j, 0.);
                        }
                   }
            
               } 

             gsl_matrix *inverse_cov_mat = invert_a_matrix(cov_mat, rows);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

             //covariance matrix
             Eigen::MatrixXd covariance_matrix = Eigen::MatrixXd::Zero(rows, rows);

             for(int k = 0; k < rows; k++) 
               {
                 for(int j = 0; j < rows; j++)
                   { 
                     if(k == j)
                      {
                        covariance_matrix(k,j) = data_unc_uncorr[k] + corr_err_matrix_square(k,j) ;
                      }
                     else
                        {     
                          covariance_matrix(k,j) =  corr_err_matrix_square(k,j);
                        }
                   }
               } 

             // Define and evaluate the inverse of the covariance matrix
             Eigen::MatrixXd covariance_matrix_inverse = covariance_matrix.inverse();   

             // Some troubleshooting
           /*  if( ds["name"].as<std::string>().find("HERMES") != std::string::npos){
              
              chi2_test = 0;
              for(int k = 0; k < (int) covariance_matrix.rows(); k++)
            {
               //if(  (Q_values[k]> Q_cuts.back()) && (z_cuts_min.back()< z_values[k] && z_cuts_max.back() > z_values[k] ))
                //{
                 // std::cout<<data_blocks.back()[k]<<"    "<<pred_blocks.back()[k]<<"    "<<(data_blocks.back()[k] - pred_blocks.back()[k])<<"     "<< pow((data_blocks.back()[k] - pred_blocks.back()[k]),2)<<std::endl<<std::endl;
                //}
              for(int j = 0; j < (int) covariance_matrix.cols(); j++) 
                {
                 if((k ==j)) // && (Q_values[k]> Q_cuts.back() && Q_values[j]> Q_cuts.back()) && (z_cuts_min.back()< z_values[k] && z_cuts_max.back() > z_values[k] && z_cuts_min.back()< z_values[j] && z_values[j] < z_cuts_max.back()))
                  {
                  chi2_test +=  (data_blocks.back()[k] - pred_blocks.back()[k]) *  covariance_matrix_inverse(k,j) * (data_blocks.back()[k] - pred_blocks.back()[k]);
                
                  std::cout <<"l'inversa della matrice di cov è "<< covariance_matrix_inverse(k,j)<<"  mentre la matrice di cov è  "<< covariance_matrix(k,j)<<std::endl<<std::endl;
                 
                  std::cout<<"il contributo al chi2 è "<<(data_blocks.back()[k] - pred_blocks.back()[k]) *  covariance_matrix_inverse(k,j) * (data_blocks.back()[k] - pred_blocks.back()[k]) <<std::endl<<std::endl;
                  }
                }
                
            }  
         
                std::cout<<std::endl<<std::endl;
                for (int j = 0; j< (int) data_blocks.back().size() ; j++)
                 {
                     
                     std::cout<<data_blocks.back()[j]<<"    "<<pred_blocks.back()[j]<<"    "<<(data_blocks.back()[j] - pred_blocks.back()[j])<<"     "<< pow((data_blocks.back()[j] - pred_blocks.back()[j]),2)<<std::endl<<std::endl;
                    
                     std::cout<<"  il chi quadro con la somma  manuale è:  "<< chi2_test<<std::endl; 
                 }
           } */
              
         /* if( ds["name"].as<std::string>().find("COMPASS") != std::string::npos){
              std::cout << "il num di righe è " <<  covariance_matrix.rows()<<std::endl;
              std::cout << "il num di cols è " <<  covariance_matrix.cols()<<std::endl;
              for(int k = 0; k < (int) covariance_matrix.rows(); k++)
            {
              for(int j = 0; j < (int) covariance_matrix.cols(); j++) 
                {
                  if(k == j)
                  {
                  std::cout <<"l'elemento diagonale "<< k << " "<<j<<" "<< covariance_matrix_inverse(k,j)<<std::endl;
                  }
                }
                
            }  
           }
           */
  
            // inserisco la matrice di cov per l'esp nel blocco
            Cov_blocks.push_back(covariance_matrix_inverse);  
             
            data_unc_uncorr.clear();
            Q_values.clear();
            z_values.clear();
            gsl_matrix_free(cov_mat);
            gsl_matrix_free(inverse_cov_mat); 

         }  
       } 
         
      std::cout<<"Replica numero "<<i<<std::endl; 
      // Compute chi2
      int n = 0;
      chi2 = 0;
      number_data = 0;
       
      // Per ogni matrice di covarianza del blocco calcolo chi2 e poi sommo
      for(const auto& matrix : Cov_blocks)
        {  
          chi2 = 0;
          data = data_blocks[n];
          predictions = pred_blocks[n];
          Q_values = Q_values_blocks[n];
          z_values = z_values_blocks[n];
          std::string name = exp_name[n];
          //number_data += data.size(); 
          

          for(int k = 0; k < (int) data.size(); k++)
            {
              for(int j = 0; j < (int) data.size(); j++) 
                {
                  if( name.find("COMPASS") != std::string::npos)
                   {
                     if(Q_values[k] > Q_cuts[n] && Q_values[j] > Q_cuts[n])
                     {
                       chi2 += (data[k] - predictions[k]) *  matrix(k,j) * (data[j] - predictions[j]);
                     }
                   }
                  if( name.find("HERMES") != std::string::npos)
                   {
                     if((Q_values[k]> Q_cuts[n] && Q_values[j]> Q_cuts[n]) && (z_cuts_min[n]< z_values[k] && z_cuts_max[n] > z_values[k] && z_cuts_min[n]< z_values[j] && z_values[j] < z_cuts_max[n]))
                      {
                        chi2 += (data[k] - predictions[k]) *  matrix(k,j) * (data[j] - predictions[j]);
                      }
                   }                  
                }
            }

          data.clear();
          predictions.clear();
          Q_values.clear();
          z_values.clear();
          n++;
          std::cout <<"il chi2 è = "<< chi2 << std::endl; 
          std::cout<<name<<std::endl<<std::endl;
          number_data = 0;
        }
      
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
  
  for (int i = 0; i < N_rep; i++)
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