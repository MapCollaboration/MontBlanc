//
// APFEL++ 2017
//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#include <apfel/apfelxx.h>
#include <apfel/sidiscoefficientfunctionsunp.h>
#include <apfel/sidiscoefficientfunctionspol.h>
#include <yaml-cpp/yaml.h>
#include <memory>

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "Usage: " << argv[0] << " <input card> <output folder> [<task> <ntasks>]" << std::endl;
      exit(-1);
    }

  // Input card
  const std::string card = argv[1];

  // Destination of tables
  const std::string path = (std::string) argv[2] + (std::string) "/";

  const int task   = argc > 4 ? std::stoi(argv[3]) : 0;
  const int ntasks = argc > 4 ? std::stoi(argv[4]) : 1;

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

  const std::vector<std::pair<std::string, std::shared_ptr<apfel::DoubleExpression>>> ops =
  {
    {"DoubleIdentity", std::make_shared<apfel::DoubleIdentity>()},
    {"C1LQ2Q",         std::make_shared<apfel::C1LQ2Q>()},
    {"C1LQ2G",         std::make_shared<apfel::C1LQ2G>()},
    {"C1LG2Q",         std::make_shared<apfel::C1LG2Q>()},
    {"C1TQ2Q",         std::make_shared<apfel::C1TQ2Q>()},
    {"C1TQ2G",         std::make_shared<apfel::C1TQ2G>()},
    {"C1TG2Q",         std::make_shared<apfel::C1TG2Q>()},
    {"C2LQ2G",         std::make_shared<apfel::C2LQ2G>()},
    {"C2LG2G",         std::make_shared<apfel::C2LG2G>()},
    {"C2LG2Q",         std::make_shared<apfel::C2LG2Q>()},
    {"C2LQ2QNS_nf3",   std::make_shared<apfel::C2LQ2QNS>(3)},
    {"C2LQ2QNS_nf4",   std::make_shared<apfel::C2LQ2QNS>(4)},
    {"C2LQ2QNS_nf5",   std::make_shared<apfel::C2LQ2QNS>(5)},
    {"C2LQ2QPS",       std::make_shared<apfel::C2LQ2QPS>()},
    {"C2LQ2QB",        std::make_shared<apfel::C2LQ2QB>()},
    {"C2LQ2QP1",       std::make_shared<apfel::C2LQ2QP1>()},
    {"C2LQ2QP2",       std::make_shared<apfel::C2LQ2QP2>()},
    {"C2LQ2QP3",       std::make_shared<apfel::C2LQ2QP3>()},
    {"C2TQ2G",         std::make_shared<apfel::C2TQ2G>()},
    {"C2TG2G",         std::make_shared<apfel::C2TG2G>()},
    {"C2TG2Q",         std::make_shared<apfel::C2TG2Q>()},
    {"C2TQ2QNS_nf3",   std::make_shared<apfel::C2TQ2QNS>(3)},
    {"C2TQ2QNS_nf4",   std::make_shared<apfel::C2TQ2QNS>(4)},
    {"C2TQ2QNS_nf5",   std::make_shared<apfel::C2TQ2QNS>(5)},
    {"C2TQ2QPS",       std::make_shared<apfel::C2TQ2QPS>()},
    {"C2TQ2QB",        std::make_shared<apfel::C2TQ2QB>()},
    {"C2TQ2QP1",       std::make_shared<apfel::C2TQ2QP1>()},
    {"C2TQ2QP2",       std::make_shared<apfel::C2TQ2QP2>()},
    {"C2TQ2QP3",       std::make_shared<apfel::C2TQ2QP3>()}
  };

  for (int i = task; i < (int) ops.size(); i += ntasks)
    {
      std::cout << "Computing " << ops[i].first << " (" << i + 1 << "/" << ops.size() << ")..." << std::endl;
      const apfel::DoubleOperator O{gx, gz, *ops[i].second, acc};
      O.EmitDoubleOperatorBinary(path + ops[i].first + ".bin");
    }

  return 0;
}
