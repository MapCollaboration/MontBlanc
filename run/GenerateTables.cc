//
// APFEL++ 2017
//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#include <apfel/apfelxx.h>
#include <apfel/sidiscoefficientfunctionsunp.h>
#include <apfel/sidiscoefficientfunctionspol.h>
#include <fstream>

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "Usage: " << argv[0] << " <input card> <output folder>" << std::endl;
      exit(-1);
    }

  // Input card
  const std::string card = argv[1];

  // Destination of tables
  const std::string path = (std::string) argv[2] + (std::string) "/";

  // Read Input Card
  YAML::Node config = YAML::LoadFile(card);

  // x-space and z-space grids
  std::vector<apfel::SubGrid> vsgx;
  for (auto const &sgx : config["Predictions"]["xgrid"])
    vsgx.push_back({sgx[0].as<int>(), sgx[1].as<double>(), sgx[2].as<int>()});
  const apfel::Grid gx{vsgx};

  std::vector<apfel::SubGrid> vsgz;
  for (auto const &sgz : config["Predictions"]["zgrid"])
    vsgz.push_back({sgz[0].as<int>(), sgz[1].as<double>(), sgz[2].as<int>()});
  const apfel::Grid gz{vsgz};

  // Integration accuracy (hardcoded for now)
  const double acc = apfel::eps5;

  std::cout << "Computing unpolarised DoubleOperator objects...\n";
  // Identity
  apfel::Timer t;
  std::cout << "- 1/25\n";
  const apfel::DoubleOperator DoubleIdentity{gx, gz, apfel::DoubleIdentity{}, acc};
  std::cout << "- 2/25\n";
  const apfel::DoubleOperator C1LQ2Q{gx, gz, apfel::C1LQ2Q{}, acc};
  std::cout << "- 3/25\n";
  const apfel::DoubleOperator C1LQ2G{gx, gz, apfel::C1LQ2G{}, acc};
  std::cout << "- 4/25\n";
  const apfel::DoubleOperator C1LG2Q{gx, gz, apfel::C1LG2Q{}, acc};
  std::cout << "- 5/25\n";
  const apfel::DoubleOperator C1TQ2Q{gx, gz, apfel::C1TQ2Q{}, acc};
  std::cout << "- 6/25\n";
  const apfel::DoubleOperator C1TQ2G{gx, gz, apfel::C1TQ2G{}, acc};
  std::cout << "- 7/25\n";
  const apfel::DoubleOperator C1TG2Q{gx, gz, apfel::C1TG2Q{}, acc};
  std::cout << "- 8/25\n";
  const apfel::DoubleOperator C2LQ2G{gx, gz, apfel::C2LQ2G{}, acc};
  std::cout << "- 9/25\n";
  const apfel::DoubleOperator C2LG2G{gx, gz, apfel::C2LG2G{}, acc};
  std::cout << "- 10/25\n";
  const apfel::DoubleOperator C2LG2Q{gx, gz, apfel::C2LG2Q{}, acc};
  std::cout << "- 11/25\n";
  const apfel::DoubleOperator C2LQ2QNS3{gx, gz, apfel::C2LQ2QNS{3}, acc};
  const apfel::DoubleOperator C2LQ2QNS4{gx, gz, apfel::C2LQ2QNS{4}, acc};
  const apfel::DoubleOperator C2LQ2QNS5{gx, gz, apfel::C2LQ2QNS{5}, acc};
  std::cout << "- 12/25\n";
  const apfel::DoubleOperator C2LQ2QPS{gx, gz, apfel::C2LQ2QPS{}, acc};
  std::cout << "- 13/25\n";
  const apfel::DoubleOperator C2LQ2QB{gx, gz, apfel::C2LQ2QB{}, acc};
  std::cout << "- 14/25\n";
  const apfel::DoubleOperator C2LQ2QP1{gx, gz, apfel::C2LQ2QP1{}, acc};
  std::cout << "- 15/25\n";
  const apfel::DoubleOperator C2LQ2QP2{gx, gz, apfel::C2LQ2QP2{}, acc};
  std::cout << "- 16/25\n";
  const apfel::DoubleOperator C2LQ2QP3{gx, gz, apfel::C2LQ2QP3{}, acc};
  std::cout << "- 17/25\n";
  const apfel::DoubleOperator C2TQ2G{gx, gz, apfel::C2TQ2G{}, acc};
  std::cout << "- 18/25\n";
  const apfel::DoubleOperator C2TG2G{gx, gz, apfel::C2TG2G{}, acc};
  std::cout << "- 19/25\n";
  const apfel::DoubleOperator C2TG2Q{gx, gz, apfel::C2TG2Q{}, acc};
  std::cout << "- 20/25\n";
  const apfel::DoubleOperator C2TQ2QNS3{gx, gz, apfel::C2TQ2QNS{3}, acc};
  const apfel::DoubleOperator C2TQ2QNS4{gx, gz, apfel::C2TQ2QNS{4}, acc};
  const apfel::DoubleOperator C2TQ2QNS5{gx, gz, apfel::C2TQ2QNS{5}, acc};
  std::cout << "- 21/25\n";
  const apfel::DoubleOperator C2TQ2QPS{gx, gz, apfel::C2TQ2QPS{}, acc};
  std::cout << "- 22/25\n";
  const apfel::DoubleOperator C2TQ2QB{gx, gz, apfel::C2TQ2QB{}, acc};
  std::cout << "- 23/25\n";
  const apfel::DoubleOperator C2TQ2QP1{gx, gz, apfel::C2TQ2QP1{}, acc};
  std::cout << "- 24/25\n";
  const apfel::DoubleOperator C2TQ2QP2{gx, gz, apfel::C2TQ2QP2{}, acc};
  std::cout << "- 25/25\n";
  const apfel::DoubleOperator C2TQ2QP3{gx, gz, apfel::C2TQ2QP3{}, acc};
  t.stop();

  // Write to file
  std::cout << "Writing DoubleOperator objects to file... ";
  std::ofstream fout;
  t.start();
  fout.open(path + "DoubleIdentity.yaml");
  fout << DoubleIdentity.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C1LQ2Q.yaml");
  fout << C1LQ2Q.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C1LQ2G.yaml");
  fout << C1LQ2G.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C1LG2Q.yaml");
  fout << C1LG2Q.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C1TQ2Q.yaml");
  fout << C1TQ2Q.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C1TQ2G.yaml");
  fout << C1TQ2G.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C1TG2Q.yaml");
  fout << C1TG2Q.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2G.yaml");
  fout << C2LQ2G.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LG2G.yaml");
  fout << C2LG2G.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LG2Q.yaml");
  fout << C2LG2Q.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QNS_nf3.yaml");
  fout << C2LQ2QNS3.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QNS_nf4.yaml");
  fout << C2LQ2QNS4.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QNS_nf5.yaml");
  fout << C2LQ2QNS5.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QPS.yaml");
  fout << C2LQ2QPS.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QB.yaml");
  fout << C2LQ2QB.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QP1.yaml");
  fout << C2LQ2QP1.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QP2.yaml");
  fout << C2LQ2QP2.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2LQ2QP3.yaml");
  fout << C2LQ2QP3.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2G.yaml");
  fout << C2TQ2G.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TG2G.yaml");
  fout << C2TG2G.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TG2Q.yaml");
  fout << C2TG2Q.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QNS_nf3.yaml");
  fout << C2TQ2QNS3.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QNS_nf4.yaml");
  fout << C2TQ2QNS4.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QNS_nf5.yaml");
  fout << C2TQ2QNS5.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QPS.yaml");
  fout << C2TQ2QPS.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QB.yaml");
  fout << C2TQ2QB.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QP1.yaml");
  fout << C2TQ2QP1.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QP2.yaml");
  fout << C2TQ2QP2.EmitDoubleOperator() << std::endl;
  fout.close();
  fout.open(path + "C2TQ2QP3.yaml");
  fout << C2TQ2QP3.EmitDoubleOperator() << std::endl;
  fout.close();
  t.stop();

  return 0;
}
