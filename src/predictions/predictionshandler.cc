//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/predictionshandler.h"

#include <apfel/apfelxx.h>
#include <apfel/sidiscoefficientfunctionsunp.h>
#include <apfel/sidiscoefficientfunctionsew.h>
#include <LHAPDF/LHAPDF.h>
#include <numeric>

// Compiler guard to avoid warnings in the use of the SOURCE_DIR
// The variable SOURCE_DIR is defined during the CMake configuration
#ifndef SOURCE_DIR
  #error "SOURCE_DIR is not defined!"
#endif //TABLE_DIR

namespace MontBlanc
{
  //_________________________________________________________________________
  PredictionsHandler::PredictionsHandler(YAML::Node                                     const& config,
                                         NangaParbat::DataHandler                       const& DH,
                                         std::shared_ptr<const apfel::Grid>             const& gx,
                                         std::shared_ptr<const apfel::Grid>             const& gz,
                                         std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    NangaParbat::ConvolutionTable{},
    _mu0(config["mu0"].as<double>()),
    _Thresholds(config["thresholds"].as<std::vector<double>>()),
    _gx(gx),
    _gz(gz),
    _obs(DH.GetObservable()),
    _bins(DH.GetBinning()),
    _qTfact(DH.GetKinematics().qTfact),
    _cmap(apfel::DiagonalBasis{13})
  {
    // Set silent mode for both apfel=+ and LHAPDF;
    apfel::SetVerbosityLevel(0);
    LHAPDF::setVerbosity(0);

    // Charge map used to discriminate between positive, negative and
    // sum of charged hadrons.
    const int charge = DH.GetCharge();
    if (charge == 1)
      // For positive hadrons, being them the default, the charge map
      // is all one's
      _ChargeMap.resize(13, 1);
    else if (charge == -1)
      {
        // For negative hadrons, charge conjugation leaves the sea-like
        // distributions (Sigma, T3, T8, ...) unchanged but changes the
        // sign of the valence-like ones (V, V3, V8, etc.).
        _ChargeMap.resize(13, 1);
        for (int i = 1; i <= 6; i++)
          _ChargeMap[2 * i] = - 1;
      }
    else if (charge == 0)
      {
        // For the sum of positive and negative hadrons, total-like
        // distributions (Sigma, T3, T8, ...) double while valence-like
        // distributions (V, V3, V8, etc.) cancel.
        _ChargeMap.resize(13, 2);
        for (int i = 1; i <= 6; i++)
          _ChargeMap[2 * i] = 0;
      }
    else
      throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unsupported charge.");

    // Perturbative order
    const int PerturbativeOrder = config["perturbative order"].as<int>();

    // Alpha_s
    const apfel::TabulateObject<double> TabAlphas{*(new apfel::AlphaQCD
      {config["alphas"]["aref"].as<double>(), config["alphas"]["Qref"].as<double>(), _Thresholds, PerturbativeOrder}), 100, 0.9, 1001, 3};
    const auto Alphas = [&] (double const& mu) -> double{ return TabAlphas.Evaluate(mu); };

    // Electromagnetic coupling
    const apfel::AlphaQED alphaem{config["alphaem"]["aref"].as<double>(), config["alphaem"]["Qref"].as<double>(), _Thresholds, {0, 0, 1.777}, 0};
    const auto Alphaem = [&] (double const& mu) -> double{ return alphaem.Evaluate(mu); };

    // Initialize QCD time-like evolution operators and tabulated them
    const std::unique_ptr<const apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabGammaij{new const apfel::TabulateObject<apfel::Set<apfel::Operator>>
      {*(BuildDglap(InitializeDglapObjectsQCDT(*_gz, _Thresholds, true), _mu0, PerturbativeOrder, Alphas)), 100, 1, 100, 3}};

    // Zero operator
    const apfel::Operator Zero{*_gz, apfel::Null{}};

    // Set cuts in the mother class
    this->_cuts = cuts;

    // Compute total cut mask as a product of single masks
    _cutmask.resize(_bins.size(), true);
    for (auto const& c : _cuts)
      _cutmask *= c->GetMask();

    // Center of mass energy
    const double Vs = DH.GetKinematics().Vs;

    // Overall prefactor
    const double pref = DH.GetPrefactor();

    if (DH.GetProcess() == NangaParbat::DataHandler::Process::SIA)
    {
        // Get the strong coupling
        const double as = Alphas(Vs);

        // Get fine-structure constant
        const double aem = Alphaem(Vs);

        // Get evolution-operator objects
        std::map<int, apfel::Operator> Gammaij = TabGammaij->Evaluate(Vs).GetObjects();

        // Get F2 objects at the scale Vs
        const apfel::StructureFunctionObjects F2Obj = apfel::InitializeF2NCObjectsZMT(*_gz, _Thresholds)(Vs, apfel::ElectroWeakCharges(Vs, true));

        // Get skip vector
        const std::vector<int> skip = F2Obj.skip;

        // Intialise container for the FK table
        std::map<int, apfel::Operator> Cj;
        for (int j = 0; j < 13; j++)
          Cj.insert({j, Zero});

        // Initialise total cross section
        double xsec = 0;

        // Loop over the quark components
        for (apfel::QuarkFlavour comp : DH.GetTagging())
          {
            // Combine perturbative contributions to the coefficient
            // functions
            apfel::Set<apfel::Operator> Ki = F2Obj.C0.at(comp);
            if (PerturbativeOrder > 0)
              Ki += ( as / apfel::FourPi ) * F2Obj.C1.at(comp);
            if (PerturbativeOrder > 1)
              Ki += pow(as / apfel::FourPi, 2) * F2Obj.C2.at(comp);

            // Convolute coefficient functions with the evolution
            // operators
            for (int j = 0; j < 13; j++)
              {
                std::map<int, apfel::Operator> gj;
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip.begin(), skip.end(), i) != skip.end()))
                    gj.insert({i, Zero});
                  else
                    gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                // Convolute distributions, combine them and return.
                Cj.at(j) += (Ki * apfel::Set<apfel::Operator> {F2Obj.ConvBasis.at(comp), gj}).Combine();
              }

            // Update total cross sections
            xsec += apfel::GetSIATotalCrossSection(PerturbativeOrder, Vs, as, aem, _Thresholds, comp);
          }

        // If the cross section is not normalised, set the total cross
        // section to one.
        if (!DH.GetNormalised())
          xsec = 1;

        // Combine coefficient functions with the total cross section
        // prefactor, including the overall prefactor. Push the same
        // resulting set of operators into the FK container as many
        // times as bins. This is not optimal but more symmetric with
        // the SIDIS case.
        for (int i = 0; i < (int) _bins.size(); i++)
          if (!_cutmask[i])
            _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
          else
            _FKt.push_back(apfel::Set<apfel::Operator>
            {(pref * apfel::GetSIATotalCrossSection(0, Vs, as, aem, _Thresholds, apfel::QuarkFlavour::TOTAL, true) / xsec ) * apfel::Set<apfel::Operator>{_cmap, Cj}});
    }
    else if (DH.GetProcess() == NangaParbat::DataHandler::Process::SIDIS)
    {
        // PDF set
        const LHAPDF::PDF* PDFs = LHAPDF::mkPDF(config["pdfset"]["name"].as<std::string>(), config["pdfset"]["member"].as<int>());

        // Target isoscalarity
        const double iso = DH.GetTargetIsoscalarity();

        // Adjust PDFs to account for the isoscalarity
        const std::function<std::map<int, double>(double const&, double const&)> tPDFs = [&] (double const& x, double const& Q) -> std::map<int, double>
        {
          const std::map<int, double> pr = PDFs->xfxQ(x, Q);
          std::map<int, double> tg = pr;
          tg.at(1)  = iso * pr.at(1)  + ( 1 - iso ) * pr.at(2);
          tg.at(2)  = iso * pr.at(2)  + ( 1 - iso ) * pr.at(1);
          tg.at(-1) = iso * pr.at(-1) + ( 1 - iso ) * pr.at(-2);
          tg.at(-2) = iso * pr.at(-2) + ( 1 - iso ) * pr.at(-1);
          return tg;
        };

        // Rotate input PDF set into the QCD evolution basis
        const auto RotPDFs = [=] (double const& x, double const& mu) -> std::map<int, double> { return apfel::PhysToQCDEv(tPDFs(x, mu)); };

        // EW charges. Set to zero charges of the flavours that are
        // not tagged.
        std::function<std::vector<double>(double const&)> fBq = [=] (double const& Q) -> std::vector<double> { return apfel::ElectroWeakCharges(Q, false); };

        // Initialise inclusive structure functions
        // TODO
        // Is _gx correct here?
        const auto IF2 = BuildStructureFunctions(InitializeF2NCObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fBq);
        const auto IFL = BuildStructureFunctions(InitializeFLNCObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fBq);

        // Inclusive cross section differential in x and Q as a
        // distribution function of Q
        const std::function<apfel::Distribution(double const&)> IncXSecQ = [&] (double const& Q) -> apfel::Distribution
        {
          // Overall Q-dependent factor of the cross section
          const double fact = 4 * M_PI * pow(Alphaem(Q), 2) / pow(Q, 3);

          // Functions that multiply F2 and FL
          const std::function<double(double const&)> func2 = [=] (double const& x) -> double{ return fact * ( 1 + pow(1 - pow(Q / Vs, 2) / x, 2) ) / x; };
          const std::function<double(double const&)> funcL = [=] (double const& x) -> double{ return - fact * pow(Q / Vs, 4) / pow(x, 3); };

          // Return cross section
          return func2 * IF2.at(0).Evaluate(Q) + funcL * IFL.at(0).Evaluate(Q);
        };

        // Tabulate total inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Distribution> TabIncXSecQ{IncXSecQ, 100, 1, 10, 3, _Thresholds};

        // Path to SIDIS tables
        std::string SIDISTablePath = SOURCE_DIR + std::string("/") + (config["SIDIS tables"] ? config["SIDIS tables"].as<std::string>() : std::string("tables"));

        // -- LO
        const apfel::DoubleOperator OT0ns{YAML::LoadFile(SIDISTablePath + "/DoubleIdentity.yaml"), *_gx, *_gz, apfel::DoubleIdentity{}};
        // -- NLO
        // Transverse
        const apfel::DoubleOperator OT1ns{YAML::LoadFile(SIDISTablePath + "/C1TQ2Q.yaml"), *_gx, *_gz, apfel::C1TQ2Q{}};
        const apfel::DoubleOperator OT1gq{YAML::LoadFile(SIDISTablePath + "/C1TQ2G.yaml"), *_gx, *_gz, apfel::C1TQ2G{}};
        const apfel::DoubleOperator OT1qg{YAML::LoadFile(SIDISTablePath + "/C1TG2Q.yaml"), *_gx, *_gz, apfel::C1TG2Q{}};
        // Longitudinal
        const apfel::DoubleOperator OL1ns{YAML::LoadFile(SIDISTablePath + "/C1LQ2Q.yaml"), *_gx, *_gz, apfel::C1LQ2Q{}};
        const apfel::DoubleOperator OL1gq{YAML::LoadFile(SIDISTablePath + "/C1LQ2G.yaml"), *_gx, *_gz, apfel::C1LQ2G{}};
        const apfel::DoubleOperator OL1qg{YAML::LoadFile(SIDISTablePath + "/C1LG2Q.yaml"), *_gx, *_gz, apfel::C1LG2Q{}};
        // -- NNLO
        // Transverse
        const apfel::DoubleOperator OT2ns_nf3{YAML::LoadFile(SIDISTablePath + "/C2TQ2QNS_nf3.yaml"), *_gx, *_gz, apfel::C2TQ2QNS{3}};
        const apfel::DoubleOperator OT2ns_nf4{YAML::LoadFile(SIDISTablePath + "/C2TQ2QNS_nf4.yaml"), *_gx, *_gz, apfel::C2TQ2QNS{4}};
        const apfel::DoubleOperator OT2ns_nf5{YAML::LoadFile(SIDISTablePath + "/C2TQ2QNS_nf5.yaml"), *_gx, *_gz, apfel::C2TQ2QNS{5}};
        const apfel::DoubleOperator OT2gq{YAML::LoadFile(SIDISTablePath + "/C2TQ2G.yaml"), *_gx, *_gz, apfel::C2TQ2G{}};
        const apfel::DoubleOperator OT2qg{YAML::LoadFile(SIDISTablePath + "/C2TG2Q.yaml"), *_gx, *_gz, apfel::C2TG2Q{}};
        const apfel::DoubleOperator OT2gg{YAML::LoadFile(SIDISTablePath + "/C2TG2G.yaml"), *_gx, *_gz, apfel::C2TG2G{}};
        const apfel::DoubleOperator OT2qbq{YAML::LoadFile(SIDISTablePath + "/C2TQ2QB.yaml"), *_gx, *_gz, apfel::C2TQ2QB{}};
        const apfel::DoubleOperator OT2qpq1{YAML::LoadFile(SIDISTablePath + "/C2TQ2QP1.yaml"), *_gx, *_gz, apfel::C2TQ2QP1{}};
        const apfel::DoubleOperator OT2qpq2{YAML::LoadFile(SIDISTablePath + "/C2TQ2QP2.yaml"), *_gx, *_gz, apfel::C2TQ2QP2{}};
        const apfel::DoubleOperator OT2qpq3{YAML::LoadFile(SIDISTablePath + "/C2TQ2QP3.yaml"), *_gx, *_gz, apfel::C2TQ2QP3{}};
        const apfel::DoubleOperator OT2ps{YAML::LoadFile(SIDISTablePath + "/C2TQ2QPS.yaml"), *_gx, *_gz, apfel::C2TQ2QPS{}};
        // Longitudinal
        const apfel::DoubleOperator OL2ns_nf3{YAML::LoadFile(SIDISTablePath + "/C2LQ2QNS_nf3.yaml"), *_gx, *_gz, apfel::C2LQ2QNS{3}};
        const apfel::DoubleOperator OL2ns_nf4{YAML::LoadFile(SIDISTablePath + "/C2LQ2QNS_nf4.yaml"), *_gx, *_gz, apfel::C2LQ2QNS{4}};
        const apfel::DoubleOperator OL2ns_nf5{YAML::LoadFile(SIDISTablePath + "/C2LQ2QNS_nf5.yaml"), *_gx, *_gz, apfel::C2LQ2QNS{5}};
        const apfel::DoubleOperator OL2gq{YAML::LoadFile(SIDISTablePath + "/C2LQ2G.yaml"), *_gx, *_gz, apfel::C2LQ2G{}};
        const apfel::DoubleOperator OL2qg{YAML::LoadFile(SIDISTablePath + "/C2LG2Q.yaml"), *_gx, *_gz, apfel::C2LG2Q{}};
        const apfel::DoubleOperator OL2gg{YAML::LoadFile(SIDISTablePath + "/C2LG2G.yaml"), *_gx, *_gz, apfel::C2LG2G{}};
        const apfel::DoubleOperator OL2qbq{YAML::LoadFile(SIDISTablePath + "/C2LQ2QB.yaml"), *_gx, *_gz, apfel::C2LQ2QB{}};
        const apfel::DoubleOperator OL2qpq1{YAML::LoadFile(SIDISTablePath + "/C2LQ2QP1.yaml"), *_gx, *_gz, apfel::C2LQ2QP1{}};
        const apfel::DoubleOperator OL2qpq2{YAML::LoadFile(SIDISTablePath + "/C2LQ2QP2.yaml"), *_gx, *_gz, apfel::C2LQ2QP2{}};
        const apfel::DoubleOperator OL2qpq3{YAML::LoadFile(SIDISTablePath + "/C2LQ2QP3.yaml"), *_gx, *_gz, apfel::C2LQ2QP3{}};
        const apfel::DoubleOperator OL2ps{YAML::LoadFile(SIDISTablePath + "/C2LQ2QPS.yaml"), *_gx, *_gz, apfel::C2LQ2QPS{}};

        const apfel::DoubleOperator OZero{*_gx, *_gz, apfel::DoubleNull{}};

        // Rotation Matrix from evolution to physical basis
        std::map<int, std::map<int, double>> Tqi;
        for (int q = -6; q <= 6; q++)
          {
            if (q == 0)
              continue;
            Tqi.insert({q, std::map<int, double>{}});
            for (int j = 1; j <= 12; j++)
              Tqi[q].insert({j, apfel::RotQCDEvToPhysFull[q+6][j]});
          }

        // Defining the quark hard x-sec
        const std::function<apfel::Set<apfel::DistributionOperator>(double const&)> Ki = [&] (double const& Q) -> apfel::Set<apfel::DistributionOperator>
        {
          // Coupling constant at NLO
          const double as  = Alphas(Q) / apfel::FourPi;
          const double as2  = as * as;

          // Number of active flavours
          const int nf = apfel::NF(Q, _Thresholds);

          // Get charges
          const std::vector<double> Bq = fBq(Q);

          // Overall Q-dependent factor of the cross section
          const double fact = ( 4 * M_PI * pow(Alphaem(Q), 2) / pow(Q, 3) );

          // Functions that multiply FT and FL
          const std::function<double(double const&, double const&)> funcL = [=] (double const& x, double const&) -> double{ return fact * 2 * ( 1 - pow(Q / Vs, 2) / x ) / x; };
	        const std::function<double(double const&, double const&)> funcT = [=] (double const& x, double const&) -> double{ return fact * ( 1 + pow(1 - pow(Q / Vs, 2) / x, 2) ) / x; };

          // Transverse
          apfel::DoubleOperator OTns   = OT0ns;
          apfel::DoubleOperator OTgq   = OZero;
          apfel::DoubleOperator OTqg   = OZero;
          apfel::DoubleOperator OTgg   = OZero;
          apfel::DoubleOperator OTps   = OZero;
          apfel::DoubleOperator OTqbq  = OZero;
          apfel::DoubleOperator OTqpq1 = OZero;
          apfel::DoubleOperator OTqpq2 = OZero;
          apfel::DoubleOperator OTqpq3 = OZero;

          // Longitudinal
          apfel::DoubleOperator OLns   = OZero;
          apfel::DoubleOperator OLgq   = OZero;
          apfel::DoubleOperator OLqg   = OZero;
          apfel::DoubleOperator OLgg   = OZero;
          apfel::DoubleOperator OLps   = OZero;
          apfel::DoubleOperator OLqbq  = OZero;
          apfel::DoubleOperator OLqpq1 = OZero;
          apfel::DoubleOperator OLqpq2 = OZero;
          apfel::DoubleOperator OLqpq3 = OZero;

          // Define operators that multiply all channels
          if (PerturbativeOrder >= 1)
            {
              OTns += as * OT1ns;
              OTgq += as * OT1gq;
              OTqg += as * OT1qg;
              OLns += as * OL1ns;
              OLgq += as * OL1gq;
              OLqg += as * OL1qg;
            }
          if (PerturbativeOrder >= 2)
            {
              if (nf == 3)
                {
                  OTns += as2 * OT2ns_nf3;
                  OLns += as2 * OL2ns_nf3;
                }
              else if (nf == 4)
                {
                  OTns += as2 * OT2ns_nf4;
                  OLns += as2 * OL2ns_nf4;
                }
              else if (nf == 5)
                {
                  OTns += as2 * OT2ns_nf5;
                  OLns += as2 * OL2ns_nf5;
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown number of active flavours.");

              OTgq   += as2 * OT2gq;
              OTqg   += as2 * OT2qg;
              OTgg   += as2 * OT2gg;
              OTps   += as2 * OT2ps;
              OTqbq  += as2 * OT2qbq;
              OTqpq1 += as2 * OT2qpq1;
              OTqpq2 += as2 * OT2qpq2;
              OTqpq3 += as2 * OT2qpq3;
              OLgq   += as2 * OL2gq;
              OLqg   += as2 * OL2qg;
              OLgg   += as2 * OL2gg;
              OLps   += as2 * OL2ps;
              OLqbq  += as2 * OL2qbq;
              OLqpq1 += as2 * OL2qpq1;
              OLqpq2 += as2 * OL2qpq2;
              OLqpq3 += as2 * OL2qpq3;
            }

          // Produce a map of distributions out of the PDFs in the physical basis
          // This line envelops the PDFs into a map of apfel::Distribution objects. These
          // objects will then be convoluted.
          const std::map<int, apfel::Distribution> DistPDFs = apfel::DistributionMap(*_gx, tPDFs, Q);

          // Initialise a map of double objects to be used to construct
          // a set
          std::map<int, apfel::DistributionOperator> KiMap;

          // gq channel (NLO / NNLO)
          // -----------------------
          apfel::Distribution eqfq = Bq[0] * ( DistPDFs.at(1) + DistPDFs.at(-1) );
          for (int q = 2; q <= 5; q++)
            eqfq += Bq[q-1] * ( DistPDFs.at(q) + DistPDFs.at(-q) );
          const apfel::DistributionOperator CT_gq_DO = OTgq.MultiplyFirstBy(eqfq);
          const apfel::DistributionOperator CL_gq_DO = OLgq.MultiplyFirstBy(eqfq);

          // gg channel (NNLO)
          // -----------------
          const double etot = std::accumulate(Bq.begin(), Bq.begin() + nf, 0.);
          const apfel::Distribution eqfgTgi = etot * DistPDFs.at(21);
          const apfel::DistributionOperator CT_gg_DO = OTgg.MultiplyFirstBy(eqfgTgi);
          const apfel::DistributionOperator CL_gg_DO = OLgg.MultiplyFirstBy(eqfgTgi);

          // Sum the gg and gq channels and insert into Ki map
          KiMap.insert({0, funcT * ( CT_gg_DO + CT_gq_DO) + funcL * (CL_gg_DO + CL_gq_DO )});

          // Construct the other channels
          for (int i = 1; i < 13; i++)
            {
              // Distribution for qq channel NS (LO / NLO / NNLO)
              // ------------------------------------------------
              apfel::Distribution eqfqTqi = Bq[0] * ( DistPDFs.at(1) * Tqi.at(1).at(i) + DistPDFs.at(-1) * Tqi.at(-1).at(i) );
              for (int q = 2; q <= 5; q++)
                eqfqTqi += Bq[q-1] * ( DistPDFs.at(q) * Tqi.at(q).at(i) + DistPDFs.at(-q) * Tqi.at(-q).at(i) );
              const apfel::DistributionOperator CT_qq_NS = OTns.MultiplyFirstBy(eqfqTqi);
              const apfel::DistributionOperator CL_qq_NS = OLns.MultiplyFirstBy(eqfqTqi);

              // Distribution for qg channel (NLO / NNLO)
              // ----------------------------------------
              double eqTqi = 0;
              for (int q = 1; q <= 5; q++)
                eqTqi += Bq[q-1] * ( Tqi.at(q).at(i) + Tqi.at(-q).at(i) );
              const apfel::Distribution fgeqTqi = eqTqi * DistPDFs.at(21);
              const apfel::DistributionOperator CT_qg = OTqg.MultiplyFirstBy(fgeqTqi);
              const apfel::DistributionOperator CL_qg = OLqg.MultiplyFirstBy(fgeqTqi);

              // Distribution PS (NNLO)
              // ----------------------
              apfel::Distribution etot_fqTqi = DistPDFs.at(1) * Tqi.at(1).at(i) + DistPDFs.at(-1) * Tqi.at(-1).at(i);
              for (int q = 2; q <= 5; q++)
                etot_fqTqi += DistPDFs.at(q) * Tqi.at(q).at(i) + DistPDFs.at(-q) * Tqi.at(-q).at(i);
              etot_fqTqi *= etot;
              const apfel::DistributionOperator CL_qq_ps = OTps.MultiplyFirstBy(etot_fqTqi);
              const apfel::DistributionOperator CT_qq_ps = OLps.MultiplyFirstBy(etot_fqTqi);

              // Distribution for \bar{q}q channel (NNLO)
              // ----------------------------------------
              apfel::Distribution eqfmqTqi = Bq[0] * ( DistPDFs.at(-1) * Tqi.at(1).at(i) + DistPDFs.at(1) * Tqi.at(-1).at(i) );
              for (int q = 2; q <= 5; q++)
                eqfmqTqi += Bq[q-1] * ( DistPDFs.at(-q) * Tqi.at(q).at(i) + DistPDFs.at(q) * Tqi.at(-q).at(i) );
              const apfel::DistributionOperator CT_qbq = OTqbq.MultiplyFirstBy(eqfmqTqi);
              const apfel::DistributionOperator CL_qbq = OLqbq.MultiplyFirstBy(eqfmqTqi);

              // Distribution (1), (2) and (3) for q'q (NNLO)
              // --------------------------------------------
              apfel::Distribution eq1fq1Tq2i{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution eq1fq2Tq1i{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution eq1eq2fq2Tq1i{*_gx, [] (double const&) -> double { return 0.0; }};
              for (int q1 = -nf; q1 <= nf; q1++)
                {
                  for (int q2 = -nf; q2 <= nf; q2++)
                    {
                      if (q1 == 0 || q2 == 0)
                        continue;

                      if (q1 != q2 && q1 != -q2)
                        {
                          eq1fq1Tq2i += Bq[std::abs(q1)-1] * DistPDFs.at(q1) * Tqi.at(q2).at(i);
                          eq1fq2Tq1i += Bq[std::abs(q2)-1] * DistPDFs.at(q1) * Tqi.at(q2).at(i);
                          eq1eq2fq2Tq1i += (q1*q2 < 0 ? -1 : 1) * sqrt(Bq[std::abs(q1)-1] * Bq[std::abs(q2)-1]) * DistPDFs.at(q1) * Tqi.at(q2).at(i);
                        }
                    }
                }
              const apfel::DistributionOperator CT_qpq_1 = OTqpq1.MultiplyFirstBy(eq1fq1Tq2i);
              const apfel::DistributionOperator CL_qpq_1 = OLqpq1.MultiplyFirstBy(eq1fq1Tq2i);
              const apfel::DistributionOperator CT_qpq_2 = OTqpq2.MultiplyFirstBy(eq1fq2Tq1i);
              const apfel::DistributionOperator CL_qpq_2 = OLqpq2.MultiplyFirstBy(eq1fq2Tq1i);
              const apfel::DistributionOperator CT_qpq_3 = OTqpq3.MultiplyFirstBy(eq1eq2fq2Tq1i);
              const apfel::DistributionOperator CL_qpq_3 = OLqpq3.MultiplyFirstBy(eq1eq2fq2Tq1i);
              KiMap.insert({i, funcT * ( CT_qq_NS + CT_qg + CT_qq_ps + CT_qbq + CT_qpq_1 + CT_qpq_2 + CT_qpq_3) + funcL * (CL_qq_NS + CL_qg + CL_qq_ps + CL_qbq + CL_qpq_1 + CL_qpq_2 + CL_qpq_3 )});
	          }

	      return apfel::Set<apfel::DistributionOperator>{KiMap};
        };

        // Tabulate semi-inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Set<apfel::DistributionOperator>> TabKi{Ki, 100, 1, 10, 3, _Thresholds};

        // Pointers to the tabulated functions
        std::unique_ptr<apfel::TabulateObject<double>> TabIncQIntegrand;
        std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabSemiIncQIntegrand;

        // Keep track of the integration bounds to avoid unneeded
        // computations
        double xl = -1;
        double xu = -1;
        double xc = -1;
        double Ql = -1;
        double Qu = -1;
        double Qc = -1;

        // Integrate inclusive cross sections and store them
        for (int i = 0; i < (int) _bins.size(); i++)
          {
            double Qmin;
            double Qmax;
            if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
              {
                Qmin = std::max(sqrt(_bins[i].xmin * _bins[i].ymin) * Vs, DH.GetKinematics().var1b.first);
                Qmax = sqrt(_bins[i].xmax * _bins[i].ymax) * Vs;
              }
            else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
              {
                Qmin = _bins[i].Qmin;
                Qmax = _bins[i].Qmax;
              }
            else
              throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");

            // If the point does not obey the cut, set FK table to zero and continue
            if (!_cutmask[i])
              {
                _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
                continue;
              }

            // If the integration bounds in x and Q are the same, no need
            // to redo the computation.
            if ((_bins[i].Intx ? xl == _bins[i].xmin && xu == _bins[i].xmax : xc == _bins[i].xav) &&
                (_bins[i].IntQ ? Ql == Qmin && Qu == Qmax : Qc == _bins[i].Qav))
              {
                _FKt.push_back(_FKt.back());
                continue;
              }

            // Tabulate Q integrand for the inclusive cross section
            const std::function<double(double const&)> IncQIntegrand = [=] (double const& Q) -> double
            {
              // Integration bounds in x
              double xbmin;
              double xbmax;
              if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
                {
                  xbmin = std::max(_bins[i].xmin, pow(Q / Vs, 2) / _bins[i].ymax);
                  xbmax = std::min(_bins[i].xmax, pow(Q / Vs, 2) / _bins[i].ymin);
                  if (DH.GetKinematics().PSRed)
                    xbmax = std::min(xbmax, 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                }
              else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
                {
                  xbmin = _bins[i].xmin;
                  xbmax = _bins[i].xmax;
                  if (DH.GetKinematics().PSRed)
                    {
                      xbmin = std::max(xbmin, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.second);
                      xbmax = std::min(std::min(xbmax, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.first), 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                    }
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");
              return (_bins[i].Intx ? TabIncXSecQ.Evaluate(Q).Integrate(xbmin, xbmax) : TabIncXSecQ.Evaluate(Q).Evaluate(_bins[i].xav));
            };

            // Tabulate Q integrand for the semi-inclusive cross section
            const std::function<apfel::Set<apfel::Operator>(double const&)> Nj = [&] (double const& Q) -> apfel::Set<apfel::Operator>
            {
              // Integration bounds in x
              double xbmin;
              double xbmax;
              if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
                {
                  xbmin = std::max(_bins[i].xmin, pow(Q / Vs, 2) / _bins[i].ymax);
                  xbmax = std::min(_bins[i].xmax, pow(Q / Vs, 2) / _bins[i].ymin);
                  if (DH.GetKinematics().PSRed)
                    xbmax = std::min(xbmax, 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                }
              else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
                {
                  xbmin = _bins[i].xmin;
                  xbmax = _bins[i].xmax;
                  if (DH.GetKinematics().PSRed)
                    {
                      xbmin = std::max(xbmin, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.second);
                      xbmax = std::min(std::min(xbmax, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.first), 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                    }
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");

              // Get Ki_map objects at the scale Q
              const std::map<int, apfel::DistributionOperator> Ki_map = TabKi.Evaluate(Q).GetObjects();

              // Compute integral of Ki in x and construct a set
              std::map<int, apfel::Operator> IntKi;
              for (auto const& tms : Ki_map)
                {
                  apfel::Operator cumulant = (_bins[i].Intx ? tms.second.Integrate(xbmin, xbmax) : tms.second.Evaluate(_bins[i].xav));
                  IntKi.insert({tms.first, cumulant});
                };

              // Get evolution operator
              apfel::Set<apfel::Operator> Gammaij = TabGammaij->Evaluate(Q);

              // Intialise container for the FK table
              std::map<int, apfel::Operator> Nj_map;
              for (int j = 0; j < 13; j++)
                Nj_map.insert({j, Zero});

              // Compute the product of Ki and Gammaij and adjust the
              // convolution basis
              for (int j = 0; j < 13; j++)
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) != 0)
                    Nj_map.at(j) += IntKi.at(i) * Gammaij.at(apfel::Gkj.at({i, j}));

              // Return the result
              return apfel::Set<apfel::Operator> {_cmap, Nj_map};
            };

            // Tabulate cross section
            TabIncQIntegrand = std::unique_ptr<apfel::TabulateObject<double>> {new apfel::TabulateObject<double> {IncQIntegrand, 50, 0.9 * Qmin, 1.1 * Qmax, 3, _Thresholds}};
            TabSemiIncQIntegrand = std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>>
                                   (new apfel::TabulateObject<apfel::Set<apfel::Operator>> {Nj, 50, 0.9 * Qmin, 1.1 * Qmax, 3, _Thresholds});

            // Push back multiplicities
            if (_bins[i].IntQ)
              _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Integrate(Qmin, Qmax) / TabIncQIntegrand->Integrate(Qmin, Qmax)});
            else
              _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Evaluate(_bins[i].Qav) / TabIncQIntegrand->Evaluate(_bins[i].Qav)});

            xl = _bins[i].xmin;
            xu = _bins[i].xmax;
            xc = _bins[i].xav;
            Ql = Qmin;
            Qu = Qmax;
            Qc = _bins[i].Qav;
          }
     }
    if (DH.GetProcess() == NangaParbat::DataHandler::Process::SIDIS_nu || DH.GetProcess() == NangaParbat::DataHandler::Process::SIDIS_nubar) 
     {
      // PDF set
        const LHAPDF::PDF* PDFs = LHAPDF::mkPDF(config["pdfset"]["name"].as<std::string>(), config["pdfset"]["member"].as<int>());

        // Target isoscalarity
        const double iso = DH.GetTargetIsoscalarity();

        // Adjust PDFs to account for the isoscalarity
        const std::function<std::map<int, double>(double const&, double const&)> tPDFs = [&] (double const& x, double const& Q) -> std::map<int, double>
        {
          const std::map<int, double> pr = PDFs->xfxQ(x, Q);
          std::map<int, double> tg = pr;
          tg.at(1)  = iso * pr.at(1)  + ( 1 - iso ) * pr.at(2);
          tg.at(2)  = iso * pr.at(2)  + ( 1 - iso ) * pr.at(1);
          tg.at(-1) = iso * pr.at(-1) + ( 1 - iso ) * pr.at(-2);
          tg.at(-2) = iso * pr.at(-2) + ( 1 - iso ) * pr.at(-1);
          return tg;
        };

        // Rotate input PDF set into the QCD evolution basis
        const auto RotPDFs = [=] (double const& x, double const& mu) -> std::map<int, double> { return apfel::PhysToQCDEv(tPDFs(x, mu)); };

        // CKM matrix elements
        std::function<std::vector<double>(double const&)> fCKM = [] (double const&) -> std::vector<double> { return apfel::CKM2; }; 

        // Check if it is nu or nubar 
        const int sign = (DH.GetProcess() == NangaParbat::DataHandler::Process::SIDIS_nu) ? +1 : -1;

        // Initialise inclusive structure functions
        const auto IF2p = BuildStructureFunctions(InitializeF2CCPlusObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fCKM);
        const auto IF2m = BuildStructureFunctions(InitializeF2CCMinusObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fCKM);

        const auto IFLp = BuildStructureFunctions(InitializeFLCCPlusObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fCKM);
        const auto IFLm = BuildStructureFunctions(InitializeFLCCMinusObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fCKM);
        
        const auto IF3p = BuildStructureFunctions(InitializeF3CCPlusObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fCKM);
        const auto IF3m = BuildStructureFunctions(InitializeF3CCMinusObjectsZM(*_gx, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fCKM);

        // Inclusive cross section differential in x and Q as a
        // distribution function of Q
        const std::function<apfel::Distribution(double const&)> IncXSecQ = [&] (double const& Q) -> apfel::Distribution
        {
          const double etaW = pow( (apfel::GFermi * pow(apfel::WMass * Q, 2)) / (4 * M_PI * Alphaem(Q) * ( pow(Q, 2) + pow(apfel::WMass, 2) ) ) , 2) / 2;
          const double fact = ( 4 * M_PI * (pow(Alphaem(Q), 2) / pow(Q, 3)) ) * 4 * etaW;

          // Functions that multiply F2 and FL
          const std::function<double(double const&)> func2 = [=] (double const& x) -> double
          { 
            const double y = pow(Q / Vs, 2) / x ;
            return fact * ( 1 + pow(1 - y, 2) ) / x; 
          };
          const std::function<double(double const&)> funcL = [=] (double const& x) -> double
          { 
            const double y = pow(Q / Vs, 2) / x;
            return - fact * pow(y, 2) / x; 
          };
          const std::function<double(double const&)> func3 = [=] (double const& x) -> double
          { 
            const double y = pow(Q / Vs, 2) / x ;
            return (sign) * fact * x * ( 1 - pow(1 - y, 2) ) / x; 
          };  

          // Return cross section 
          return func2 * ( IF2p.at(0).Evaluate(Q) + sign * IF2m.at(0).Evaluate(Q) ) + funcL * ( IFLp.at(0).Evaluate(Q) + sign * IFLm.at(0).Evaluate(Q) ) + func3 * ( IF3p.at(0).Evaluate(Q) + sign * IF3m.at(0).Evaluate(Q) );
        };

        // Tabulate total inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Distribution> TabIncXSecQ{IncXSecQ, 100, 1, 10, 3, _Thresholds};

        // Path to SIDIS tables 
        std::string SIDISTablePath = SOURCE_DIR + std::string("/") + (config["SIDIS tables"] ? config["SIDIS tables"].as<std::string>() : std::string("tables_ew"));
        
        // -- LO
        const apfel::DoubleOperator O0qq{YAML::LoadFile(SIDISTablePath + "/DoubleIdentity.yaml"), *_gx, *_gz, apfel::DoubleIdentity{}};

        // -- NLO
        // Transverse
        const apfel::DoubleOperator OT1qq{YAML::LoadFile(SIDISTablePath + "/FTC1q2qM.yaml"), *_gx, *_gz, apfel::FTC1q2qM{}};
        const apfel::DoubleOperator OT1gq{YAML::LoadFile(SIDISTablePath + "/FTC1q2gM.yaml"), *_gx, *_gz, apfel::FTC1q2gM{}};
        const apfel::DoubleOperator OT1qg{YAML::LoadFile(SIDISTablePath + "/FTC1g2qM.yaml"), *_gx, *_gz, apfel::FTC1g2qM{}};
        // Longitudinal
        const apfel::DoubleOperator OL1qq{YAML::LoadFile(SIDISTablePath + "/FLC1q2qM.yaml"), *_gx, *_gz, apfel::FLC1q2qM{}};
        const apfel::DoubleOperator OL1gq{YAML::LoadFile(SIDISTablePath + "/FLC1q2gM.yaml"), *_gx, *_gz, apfel::FLC1q2gM{}};
        const apfel::DoubleOperator OL1qg{YAML::LoadFile(SIDISTablePath + "/FLC1g2qM.yaml"), *_gx, *_gz, apfel::FLC1g2qM{}};
        // F3
        const apfel::DoubleOperator O31qq{YAML::LoadFile(SIDISTablePath + "/F3C1q2qM.yaml"), *_gx, *_gz, apfel::F3C1q2qM{}};
        const apfel::DoubleOperator O31gq{YAML::LoadFile(SIDISTablePath + "/F3C1q2gM.yaml"), *_gx, *_gz, apfel::F3C1q2gM{}};
        const apfel::DoubleOperator O31qg{YAML::LoadFile(SIDISTablePath + "/F3C1g2qM.yaml"), *_gx, *_gz, apfel::F3C1g2qM{}};

        // -- NNLO
        // Transverse
        const apfel::DoubleOperator OT2qq_nf3{YAML::LoadFile(SIDISTablePath + "/FTC2q2qM_nf3.yaml"), *_gx, *_gz, apfel::FTC2q2qM{3}};
        const apfel::DoubleOperator OT2qq_nf4{YAML::LoadFile(SIDISTablePath + "/FTC2q2qM_nf4.yaml"), *_gx, *_gz, apfel::FTC2q2qM{4}};
        const apfel::DoubleOperator OT2qq_nf5{YAML::LoadFile(SIDISTablePath + "/FTC2q2qM_nf5.yaml"), *_gx, *_gz, apfel::FTC2q2qM{5}};
        const apfel::DoubleOperator OT2gq{YAML::LoadFile(SIDISTablePath + "/FTC2q2gM.yaml"), *_gx, *_gz, apfel::FTC2q2gM{}};
        const apfel::DoubleOperator OT2qg{YAML::LoadFile(SIDISTablePath + "/FTC2g2qM.yaml"), *_gx, *_gz, apfel::FTC2g2qM{}};
        const apfel::DoubleOperator OT2gg{YAML::LoadFile(SIDISTablePath + "/FTC2g2gM.yaml"), *_gx, *_gz, apfel::FTC2g2gM{}};
        const apfel::DoubleOperator OT2qqFcon2{YAML::LoadFile(SIDISTablePath + "/FTC2q2qMFcon2.yaml"), *_gx, *_gz, apfel::FTC2q2qMFcon2{}};
        const apfel::DoubleOperator OT2qbq{YAML::LoadFile(SIDISTablePath + "/FTC2q2qbM.yaml"), *_gx, *_gz, apfel::FTC2q2qbM{}};
        const apfel::DoubleOperator OT2qqFcon1{YAML::LoadFile(SIDISTablePath + "/FTC2q2qMFcon1.yaml"), *_gx, *_gz, apfel::FTC2q2qMFcon1{}};
        const apfel::DoubleOperator OT2qbqFcon{YAML::LoadFile(SIDISTablePath + "/FTC2q2qbMFcon.yaml"), *_gx, *_gz, apfel::FTC2q2qbMFcon{}};
        const apfel::DoubleOperator OT2qpq1{YAML::LoadFile(SIDISTablePath + "/FTC2q2qpM1.yaml"), *_gx, *_gz, apfel::FTC2q2qpM1{}};
        const apfel::DoubleOperator OT2qpq2{YAML::LoadFile(SIDISTablePath + "/FTC2q2qpM2.yaml"), *_gx, *_gz, apfel::FTC2q2qpM2{}};
        // Longitudinal
        const apfel::DoubleOperator OL2qq_nf3{YAML::LoadFile(SIDISTablePath + "/FLC2q2qM_nf3.yaml"), *_gx, *_gz, apfel::FLC2q2qM{3}};
        const apfel::DoubleOperator OL2qq_nf4{YAML::LoadFile(SIDISTablePath + "/FLC2q2qM_nf4.yaml"), *_gx, *_gz, apfel::FLC2q2qM{4}};
        const apfel::DoubleOperator OL2qq_nf5{YAML::LoadFile(SIDISTablePath + "/FLC2q2qM_nf5.yaml"), *_gx, *_gz, apfel::FLC2q2qM{5}};
        const apfel::DoubleOperator OL2gq{YAML::LoadFile(SIDISTablePath + "/FLC2q2gM.yaml"), *_gx, *_gz, apfel::FLC2q2gM{}};
        const apfel::DoubleOperator OL2qg{YAML::LoadFile(SIDISTablePath + "/FLC2g2qM.yaml"), *_gx, *_gz, apfel::FLC2g2qM{}};
        const apfel::DoubleOperator OL2gg{YAML::LoadFile(SIDISTablePath + "/FLC2g2gM.yaml"), *_gx, *_gz, apfel::FLC2g2gM{}};
        const apfel::DoubleOperator OL2qqFcon2{YAML::LoadFile(SIDISTablePath + "/FLC2q2qMFcon2.yaml"), *_gx, *_gz, apfel::FLC2q2qMFcon2{}};
        const apfel::DoubleOperator OL2qbq{YAML::LoadFile(SIDISTablePath + "/FLC2q2qbM.yaml"), *_gx, *_gz, apfel::FLC2q2qbM{}};
        const apfel::DoubleOperator OL2qqFcon1{YAML::LoadFile(SIDISTablePath + "/FLC2q2qMFcon1.yaml"), *_gx, *_gz, apfel::FLC2q2qMFcon1{}};
        const apfel::DoubleOperator OL2qbqFcon{YAML::LoadFile(SIDISTablePath + "/FLC2q2qbMFcon.yaml"), *_gx, *_gz, apfel::FLC2q2qbMFcon{}};
        const apfel::DoubleOperator OL2qpq1{YAML::LoadFile(SIDISTablePath + "/FLC2q2qpM1.yaml"), *_gx, *_gz, apfel::FLC2q2qpM1{}};
        const apfel::DoubleOperator OL2qpq2{YAML::LoadFile(SIDISTablePath + "/FLC2q2qpM2.yaml"), *_gx, *_gz, apfel::FLC2q2qpM2{}};
        // F3
        const apfel::DoubleOperator O32qq_nf3{YAML::LoadFile(SIDISTablePath + "/F3C2q2qM_nf3.yaml"), *_gx, *_gz, apfel::F3C2q2qM{3}};
        const apfel::DoubleOperator O32qq_nf4{YAML::LoadFile(SIDISTablePath + "/F3C2q2qM_nf4.yaml"), *_gx, *_gz, apfel::F3C2q2qM{4}};
        const apfel::DoubleOperator O32qq_nf5{YAML::LoadFile(SIDISTablePath + "/F3C2q2qM_nf5.yaml"), *_gx, *_gz, apfel::F3C2q2qM{5}};
        const apfel::DoubleOperator O32gq{YAML::LoadFile(SIDISTablePath + "/F3C2q2gM.yaml"), *_gx, *_gz, apfel::F3C2q2gM{}};
        const apfel::DoubleOperator O32qg{YAML::LoadFile(SIDISTablePath + "/F3C2g2qM.yaml"), *_gx, *_gz, apfel::F3C2g2qM{}};
        const apfel::DoubleOperator O32qbq{YAML::LoadFile(SIDISTablePath + "/F3C2q2qbM.yaml"), *_gx, *_gz, apfel::F3C2q2qbM{}};
        const apfel::DoubleOperator O32qqFcon1{YAML::LoadFile(SIDISTablePath + "/F3C2q2qMFcon1.yaml"), *_gx, *_gz, apfel::F3C2q2qMFcon1{}};
        const apfel::DoubleOperator O32qbqFcon{YAML::LoadFile(SIDISTablePath + "/F3C2q2qbMFcon.yaml"), *_gx, *_gz, apfel::F3C2q2qbMFcon{}};
        const apfel::DoubleOperator O32qpq1{YAML::LoadFile(SIDISTablePath + "/F3C2q2qpM1.yaml"), *_gx, *_gz, apfel::F3C2q2qpM1{}};
        const apfel::DoubleOperator O32qpq2{YAML::LoadFile(SIDISTablePath + "/F3C2q2qpM2.yaml"), *_gx, *_gz, apfel::F3C2q2qpM2{}};
        
        const apfel::DoubleOperator OZero{*_gx, *_gz, apfel::DoubleNull{}};

        // Rotation Matrix from evolution to physical basis
        std::map<int, std::map<int, double>> Tqi;
        for (int q = -6; q <= 6; q++)
          {
            if (q == 0)
              continue;
            Tqi.insert({q, std::map<int, double>{}});
            for (int j = 1; j <= 12; j++)
              Tqi[q].insert({j, apfel::RotQCDEvToPhysFull[q+6][j]});
          }

        // Defining the quark hard x-sec
        const std::function<apfel::Set<apfel::DistributionOperator>(double const&)> Ki = [&] (double const& Q) -> apfel::Set<apfel::DistributionOperator>
        {
          // Coupling constant at NLO
          const double as  = Alphas(Q) / apfel::FourPi;
          const double as2  = as * as;

          // Number of active flavours
          const int nf = apfel::NF(Q, _Thresholds);

          // Get charges
          double V2[3][3];
          std::copy(apfel::CKM2.begin(), apfel::CKM2.end(), &V2[0][0]);

          // Overall Q-dependent factor of the cross section
          const double etaW = pow( (apfel::GFermi * pow(apfel::WMass * Q, 2)) / (4 * M_PI * Alphaem(Q) * ( pow(Q, 2) + pow(apfel::WMass, 2) ) ) , 2) / 2;
          const double fact = ( 4 * M_PI * (pow(Alphaem(Q), 2) / pow(Q, 3)) ) * 4 * etaW;

          // Functions that multiply FT, FL and F3
	        const std::function<double(double const&, double const&)> funcT = [=] (double const& x, double const&) -> double
          { 
            const double y = pow(Q / Vs, 2) / x ;
	          return fact * ( 1 + pow(1 - y, 2) ) / x;
          };  
          const std::function<double(double const&, double const&)> func3 = [=] (double const& x, double const&) -> double
          { 
            const double y = pow(Q / Vs, 2) / x ;
	          return (sign) * fact * ( 1 - pow(1 - y, 2) ) / x;
          };

          const std::function<double(double const&, double const&)> funcL = [=] (double const& x, double const&) -> double
	        {
	          const double y = pow(Q / Vs, 2) / x ;
	          return fact * 2 * ( 1 - y ) / x;
	        };

          // Transverse
          apfel::DoubleOperator OTqq      = O0qq;
          apfel::DoubleOperator OTgq      = OZero;
          apfel::DoubleOperator OTqg      = OZero;
          apfel::DoubleOperator OTgg      = OZero;
          apfel::DoubleOperator OTqbq     = OZero;
          apfel::DoubleOperator OTqpq1    = OZero;
          apfel::DoubleOperator OTqpq2    = OZero;
          apfel::DoubleOperator OTqqFcon2 = OZero;
          apfel::DoubleOperator OTqqFcon1 = OZero;
          apfel::DoubleOperator OTqbqFcon = OZero;

          // Longitudinal
          apfel::DoubleOperator OLqq      = OZero;
          apfel::DoubleOperator OLgq      = OZero;
          apfel::DoubleOperator OLqg      = OZero;
          apfel::DoubleOperator OLgg      = OZero;
          apfel::DoubleOperator OLqbq     = OZero;
          apfel::DoubleOperator OLqpq1    = OZero;
          apfel::DoubleOperator OLqpq2    = OZero;
          apfel::DoubleOperator OLqqFcon2 = OZero;
          apfel::DoubleOperator OLqqFcon1 = OZero;
          apfel::DoubleOperator OLqbqFcon = OZero;

          // 3 
          apfel::DoubleOperator O3qq      = O0qq;
          apfel::DoubleOperator O3gq      = OZero;
          apfel::DoubleOperator O3qg      = OZero;
          apfel::DoubleOperator O3qbq     = OZero;
          apfel::DoubleOperator O3qqFcon1 = OZero;
          apfel::DoubleOperator O3qbqFcon = OZero;
          apfel::DoubleOperator O3qpq1    = OZero;
          apfel::DoubleOperator O3qpq2    = OZero;

          // Define operators that multiply all channels
          if (PerturbativeOrder >= 1)
            {
              OTqq += as * OT1qq;
              OTgq += as * OT1gq;
              OTqg += as * OT1qg;

              OLqq += as * OL1qq;
              OLgq += as * OL1gq;
              OLqg += as * OL1qg;

              O3qq += as * O31qq;
              O3gq += as * O31gq;
              O3qg += as * O31qg;
            }
            if (PerturbativeOrder >= 2)
            {
              if (nf == 3)
                {
                  OTqq += as2 * OT2qq_nf3;
                  OLqq += as2 * OL2qq_nf3;
                  O3qq += as2 * O32qq_nf3;
                }
              else if (nf == 4)
                {
                  OTqq += as2 * OT2qq_nf4;
                  OLqq += as2 * OL2qq_nf4;
                  O3qq += as2 * O32qq_nf4;
                }
              else if (nf == 5)
                {
                  OTqq += as2 * OT2qq_nf5;
                  OLqq += as2 * OL2qq_nf5;
                  O3qq += as2 * O32qq_nf5;
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown number of active flavours.");

              OTgq      += as2 * OT2gq;
              OTqg      += as2 * OT2qg;
              OTgg      += as2 * OT2gg;
              OTqbq     += as2 * OT2qbq;
              OTqpq1    += as2 * OT2qpq1;
              OTqpq2    += as2 * OT2qpq2;
              OTqbqFcon += as2 * OT2qbqFcon;
              OTqqFcon1 += as2 * OT2qqFcon1;
              OTqqFcon2 += as2 * OT2qqFcon2;

              OLgq      += as2 * OL2gq;
              OLqg      += as2 * OL2qg;
              OLgg      += as2 * OL2gg;
              OLqbq     += as2 * OL2qbq;
              OLqpq1    += as2 * OL2qpq1;
              OLqpq2    += as2 * OL2qpq2;
              OLqbqFcon += as2 * OL2qbqFcon;
              OLqqFcon1 += as2 * OL2qqFcon1;
              OLqqFcon2 += as2 * OL2qqFcon2;

              O3gq      += as2 * O32gq;
              O3qg      += as2 * O32qg;
              O3qbq     += as2 * O32qbq;
              O3qpq1    += as2 * O32qpq1;
              O3qpq2    += as2 * O32qpq2;
              O3qqFcon1 += as2 * O32qqFcon1;
              O3qbqFcon += as2 * O32qbqFcon;
              
            }
          // Produce a map of distributions out of the PDFs in the physical basis
          // This line envelops the PDFs into a map of apfel::Distribution objects. These
          // objects will then be convoluted.
          const std::map<int, apfel::Distribution> DistPDFs = apfel::DistributionMap(*_gx, tPDFs, Q);

          // Initialise a map of double objects to be used to construct
          // a set
          std::map<int, apfel::DistributionOperator> KiMap;

          // Define vectors to pair the CC contributions for the quarks 
          std::vector<int> U = {2, 4, 6};
          std::vector<int> D = {1, 3, 5};  

          // gq channel (NLO / NNLO)
          // -----------------------
          apfel::Distribution Vqfq{*_gx, [] (double const&) -> double { return 0.0; }};
          apfel::Distribution Vqfq_3{*_gx, [] (double const&) -> double { return 0.0; }};

          for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++)
            {
              int alpha = U[k];
              int beta  = D[j];   
              if (beta > nf || alpha > nf )
		             continue;               
              Vqfq   += V2[k][j] *             ( DistPDFs.at((- sign) * alpha) + DistPDFs.at((- sign) * - beta));
              Vqfq_3 += V2[k][j] *  (- sign) * ( DistPDFs.at((- sign) * alpha) - DistPDFs.at((- sign) * - beta));
            }

          const apfel::DistributionOperator CT_gq_DO = OTgq.MultiplyFirstBy(Vqfq);
          const apfel::DistributionOperator CL_gq_DO = OLgq.MultiplyFirstBy(Vqfq);
          const apfel::DistributionOperator C3_gq_DO = O3gq.MultiplyFirstBy(Vqfq_3);

          // gg channel (NNLO)
          // -----------------
          double SumV2 = 0;
	        for (int i = 0; i < 3; i++)
	        for (int j = 0; j < 3; j++)
	        {
		      const int alpha = U[i];
		      const int beta  = D[j];

		      if (beta > nf || alpha > nf)
		        continue;

          SumV2 += V2[i][j];
          }
          const apfel::Distribution VqfgTgi = SumV2 * DistPDFs.at(21);
          const apfel::DistributionOperator CT_gg_DO = OTgg.MultiplyFirstBy(VqfgTgi);
          const apfel::DistributionOperator CL_gg_DO = OLgg.MultiplyFirstBy(VqfgTgi);

          // Sum the gg and gq channels and insert into Ki map
          KiMap.insert({0,  funcT * ( CT_gg_DO + CT_gq_DO )
                          + funcL * ( CL_gg_DO + CL_gq_DO ) 
                          + func3 * ( C3_gq_DO ) 
                      });

          // Construct the other channels
          for (int i = 1; i < 13; i++) 
            {
              // Distribution for qq channel NS (LO / NLO / NNLO)
              // ------------------------------------------------
              apfel::Distribution VqfqTqi{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution VqfqTqi_3{*_gx, [] (double const&) -> double { return 0.0; }};            
              for (int k = 0; k < 3; k++)
	              for (int j = 0; j < 3; j++)
	              {
                int alpha = U[k];
                int beta  = D[j];   
                if (beta > nf || alpha > nf )
		             continue;   

                VqfqTqi   += V2[k][j] *            ( DistPDFs.at((- sign) * alpha) * Tqi.at((- sign) * beta).at(i) 
                                                      + DistPDFs.at((- sign) * (- beta)) * Tqi.at((- sign) * (- alpha)).at(i) );
                VqfqTqi_3 += V2[k][j] * (- sign) * ( DistPDFs.at((- sign) * alpha) * Tqi.at((- sign) * beta).at(i) 
                                                      - DistPDFs.at((- sign) * (- beta)) * Tqi.at((- sign) * (- alpha)).at(i) );
                }
              const apfel::DistributionOperator CT_qq = OTqq.MultiplyFirstBy(VqfqTqi);
              const apfel::DistributionOperator CL_qq = OLqq.MultiplyFirstBy(VqfqTqi);
              const apfel::DistributionOperator C3_qq = O3qq.MultiplyFirstBy(VqfqTqi_3);

              // Distribution for qg channel (NLO / NNLO)
              // ----------------------------------------
              double VqTqi = 0;
              double VqTqi_3 = 0;
              
              for (int k = 0; k < 3; k++)
	             for (int j = 0; j < 3; j++)
	             {
                int alpha = U[k];
                int beta  = D[j];   
                if (beta > nf || alpha > nf )
		             continue; 
                  VqTqi   += V2[k][j] *            ( Tqi.at((- sign) * beta ).at(i) + Tqi.at( (- sign) * (- alpha) ).at(i) );
                  VqTqi_3 += V2[k][j] * (- sign) * ( Tqi.at((- sign) * beta ).at(i) - Tqi.at( (- sign) * (- alpha) ).at(i) );
               }
              const apfel::Distribution fgVqTqi = VqTqi * DistPDFs.at(21);
              const apfel::Distribution fgVqTqi_3 = VqTqi_3 * DistPDFs.at(21);

              const apfel::DistributionOperator CT_qg = OTqg.MultiplyFirstBy(fgVqTqi);
              const apfel::DistributionOperator CL_qg = OLqg.MultiplyFirstBy(fgVqTqi);
              const apfel::DistributionOperator C3_qg = O3qg.MultiplyFirstBy(fgVqTqi_3);

              // Distribution for \bar{q}q and \bar{q}qFcon channel (NNLO)
              // ---------------------------------------------------------
              apfel::Distribution VqfmqTqi{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution VqfmqTqi_3{*_gx, [] (double const&) -> double { return 0.0; }};
              
              apfel::Distribution VqfmqTqiFcon{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution VqfmqTqiFcon_3{*_gx, [] (double const&) -> double { return 0.0; }};

              for (int k = 0; k < 3; k++)
	              for (int j = 0; j < 3; j++)
	             {
                int alpha = U[k];
                int beta  = D[j];   
                if (beta > nf || alpha > nf )
		             continue; 
                  VqfmqTqi       += V2[k][j] *            ( DistPDFs.at((- sign) * alpha) * Tqi.at(- (- sign) * beta).at(i) + DistPDFs.at(- (- sign) * beta) * Tqi.at((- sign) * alpha).at(i) );
                  VqfmqTqi_3     += V2[k][j] * (- sign) * ( DistPDFs.at((- sign) * alpha) * Tqi.at(- (- sign) * beta).at(i) - DistPDFs.at(- (- sign) * beta) * Tqi.at((- sign) * alpha).at(i) );

                  VqfmqTqiFcon   += V2[k][j] *            ( DistPDFs.at((- sign) * alpha) * Tqi.at(- (- sign) * alpha).at(i) + DistPDFs.at(- (- sign) * beta) * Tqi.at((- sign) * beta).at(i) );
                  VqfmqTqiFcon_3 += V2[k][j] * (- sign) * ( DistPDFs.at((- sign) * alpha) * Tqi.at(- (- sign) * alpha).at(i) - DistPDFs.at(- (- sign) * beta) * Tqi.at((- sign) * beta).at(i) );
               }
              const apfel::DistributionOperator CT_qbq = OTqbq.MultiplyFirstBy(VqfmqTqi);
              const apfel::DistributionOperator CL_qbq = OLqbq.MultiplyFirstBy(VqfmqTqi);
              const apfel::DistributionOperator C3_qbq = O3qbq.MultiplyFirstBy(VqfmqTqi_3);

              const apfel::DistributionOperator CT_qbqFcon = OTqbqFcon.MultiplyFirstBy(VqfmqTqiFcon);
              const apfel::DistributionOperator CL_qbqFcon = OLqbqFcon.MultiplyFirstBy(VqfmqTqiFcon);
              const apfel::DistributionOperator C3_qbqFcon = O3qbqFcon.MultiplyFirstBy(VqfmqTqiFcon_3);

              // Distribution Fcon1 for qq (NNLO)
              // -----------------------------------------  
              apfel::Distribution VqfqTqiFcon1{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution VqfqTqiFcon1_3{*_gx, [] (double const&) -> double { return 0.0; }};           

              for (int k = 0; k < 3; k++)
	              for (int j = 0; j < 3; j++)
	             {
                int alpha = U[k];
                int beta  = D[j];   
                if (beta > nf || alpha > nf )
		             continue; 
                  VqfqTqiFcon1   += V2[k][j] *            ( DistPDFs.at((- sign) * alpha) * Tqi.at((- sign) * alpha).at(i) + DistPDFs.at(- (- sign) * beta) * Tqi.at(- (- sign) * beta).at(i) );
                  VqfqTqiFcon1_3 += V2[k][j] * (- sign) * ( DistPDFs.at((- sign) * alpha) * Tqi.at((- sign) * alpha).at(i) - DistPDFs.at(- (- sign) * beta) * Tqi.at(- (- sign) * beta).at(i) );
                }

              const apfel::DistributionOperator CT_qqFcon1 = OTqqFcon1.MultiplyFirstBy(VqfqTqiFcon1);
              const apfel::DistributionOperator CL_qqFcon1 = OLqqFcon1.MultiplyFirstBy(VqfqTqiFcon1);
              const apfel::DistributionOperator C3_qqFcon1 = O3qqFcon1.MultiplyFirstBy(VqfqTqiFcon1_3);

              // Distribution Fcon2 for qq (NNLO) 
              // --------------------------------  
              apfel::Distribution VqFcon2{*_gx, [] (double const&) -> double { return 0.0; }};

              for (int f=1; f <= nf; f++)
                VqFcon2 += SumV2 * ( DistPDFs.at(- (- sign) * f) * Tqi.at(- (- sign) * f).at(i) + DistPDFs.at((- sign) * f) * Tqi.at((- sign) * f).at(i) );

              const apfel::DistributionOperator CT_qqFcon2 = OTqqFcon2.MultiplyFirstBy(VqFcon2);
              const apfel::DistributionOperator CL_qqFcon2 = OLqqFcon2.MultiplyFirstBy(VqFcon2);

            
              // Distribution (1) and (2) for q'q channel (NNLO)
              // ------------------------------------------------
              apfel::Distribution VqfqpTqi1{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution VqfqpTqi2{*_gx, [] (double const&) -> double { return 0.0; }};

              apfel::Distribution VqfqpTqi1_3{*_gx, [] (double const&) -> double { return 0.0; }};
              apfel::Distribution VqfqpTqi2_3{*_gx, [] (double const&) -> double { return 0.0; }};

              for (int f=1; f <= nf; f++)
              for (int k = 0; k < 3; k++)
	            for (int j = 0; j < 3; j++)
	            {
                int alpha = U[k];
                int beta  = D[j];   
                if (beta > nf || alpha > nf )
		             continue; 
                    VqfqpTqi1   += V2[k][j] *            ( DistPDFs.at((- sign) * alpha) + DistPDFs.at(- (- sign) * beta)) * ( Tqi.at(f).at(i) + Tqi.at(-f).at(i) );
                    VqfqpTqi1_3 += V2[k][j] * (- sign) * ( DistPDFs.at((- sign) * alpha) - DistPDFs.at(- (- sign) * beta)) * ( Tqi.at(f).at(i) + Tqi.at(-f).at(i) );

                    VqfqpTqi2   += V2[k][j] *            ( DistPDFs.at(f) + DistPDFs.at(-f) ) * (Tqi.at( (- sign) * beta ).at(i) + Tqi.at(- (- sign) * alpha).at(i));
                    VqfqpTqi2_3 += V2[k][j] * (- sign) * ( DistPDFs.at(f) + DistPDFs.at(-f) ) * (Tqi.at( (- sign) * beta ).at(i) - Tqi.at(- (- sign) * alpha).at(i));
              }
                
              const apfel::DistributionOperator CT_qpq1 = OTqpq1.MultiplyFirstBy(VqfqpTqi1);
              const apfel::DistributionOperator CL_qpq1 = OLqpq1.MultiplyFirstBy(VqfqpTqi1);
              const apfel::DistributionOperator C3_qpq1 = O3qpq1.MultiplyFirstBy(VqfqpTqi1_3);

              const apfel::DistributionOperator CT_qpq2 = OTqpq2.MultiplyFirstBy(VqfqpTqi2);
              const apfel::DistributionOperator CL_qpq2 = OLqpq2.MultiplyFirstBy(VqfqpTqi2);
              const apfel::DistributionOperator C3_qpq2 = O3qpq2.MultiplyFirstBy(VqfqpTqi2_3);

              KiMap.insert({i,  funcT * ( CT_qq + CT_qg + CT_qbq + CT_qbqFcon + CT_qqFcon1 + CT_qqFcon2 + CT_qpq1 + CT_qpq2 ) 
                              + funcL * ( CL_qq + CL_qg + CL_qbq + CL_qbqFcon + CL_qqFcon1 + CL_qqFcon2 + CL_qpq1 + CL_qpq2 ) 
                              + func3 * ( C3_qq + C3_qg + C3_qbq + C3_qbqFcon + C3_qqFcon1 +              C3_qpq1 + C3_qpq2 ) 
                            });
            }

	      return apfel::Set<apfel::DistributionOperator>{KiMap};
        };

        // Tabulate semi-inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Set<apfel::DistributionOperator>> TabKi{Ki, 100, 1, 10, 3, _Thresholds};

        // Pointers to the tabulated functions
        std::unique_ptr<apfel::TabulateObject<double>> TabIncQIntegrand;
        std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabSemiIncQIntegrand;

        // Keep track of the integration bounds to avoid unneeded
        // computations
        double xl = -1;
        double xu = -1;
        double xc = -1;
        double Ql = -1;
        double Qu = -1;
        double Qc = -1;

        // Integrate inclusive cross sections and store them
        for (int i = 0; i < (int) _bins.size(); i++)
          {
            double Qmin;
            double Qmax;
            if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
              {
                Qmin = std::max(sqrt(_bins[i].xmin * _bins[i].ymin) * Vs, DH.GetKinematics().var1b.first);
                Qmax = sqrt(_bins[i].xmax * _bins[i].ymax) * Vs;
              }
            else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
              {
                Qmin = _bins[i].Qmin;
                Qmax = _bins[i].Qmax;
              }
            else
              throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");

            // If the point does not obey the cut, set FK table to zero and continue
            if (!_cutmask[i])
              {
                _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
                continue;
              }

            // If the integration bounds in x and Q are the same, no need
            // to redo the computation.
            if ((_bins[i].Intx ? xl == _bins[i].xmin && xu == _bins[i].xmax : xc == _bins[i].xav) &&
                (_bins[i].IntQ ? Ql == Qmin && Qu == Qmax : Qc == _bins[i].Qav))
              {
                _FKt.push_back(_FKt.back());
                continue;
              }

            // Tabulate Q integrand for the inclusive cross section
            const std::function<double(double const&)> IncQIntegrand = [=] (double const& Q) -> double
            {
              // Integration bounds in x
              double xbmin;
              double xbmax;
              if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
                {
                  xbmax = std::min(std::min(_bins[i].xmax, pow(Q / Vs, 2) / _bins[i].ymin), 1.0);
                  if (DH.GetKinematics().PSRed)
                    xbmax = std::min(xbmax, 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));

                  xbmin = std::min( xbmax, std::max(_bins[i].xmin, pow(Q / Vs, 2) / _bins[i].ymax));
                }
              else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
                {
                  xbmin = _bins[i].xmin;
                  xbmax = _bins[i].xmax;
                  if (DH.GetKinematics().PSRed)
                  {
                    xbmax = std::min(std::min(std::min( xbmax, 1.0 / ( 1.0 + pow(DH.GetKinematics().pTMin / Q, 2)) ) , 
                                                          pow(Q / Vs, 2) / DH.GetKinematics().etaRange.first), 1.0);
                    xbmin = std::min( xbmax, std::max(xbmin, pow(Q / Vs, 2)/ DH.GetKinematics().etaRange.second ) );    
                  }
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");
              return (_bins[i].Intx ? TabIncXSecQ.Evaluate(Q).Integrate(xbmin, xbmax) : TabIncXSecQ.Evaluate(Q).Evaluate(_bins[i].xav));
            };

            // Tabulate Q integrand for the semi-inclusive cross section
            const std::function<apfel::Set<apfel::Operator>(double const&)> Nj = [&] (double const& Q) -> apfel::Set<apfel::Operator>
            {
              // Integration bounds in x
              double xbmin;
              double xbmax;
              if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
              {
                xbmax = std::min(std::min(_bins[i].xmax, pow(Q / Vs, 2) / _bins[i].ymin), 1.0);
                if (DH.GetKinematics().PSRed)
                  xbmax = std::min(xbmax, 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                    
                xbmin = std::min( xbmax, std::max(_bins[i].xmin, pow(Q / Vs, 2) / _bins[i].ymax));
              }
              else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
              {
                xbmin = _bins[i].xmin;
                xbmax = _bins[i].xmax;
                if (DH.GetKinematics().PSRed)
                  {
                    xbmax = std::min(std::min(std::min( xbmax, 1.0 / ( 1.0 + pow(DH.GetKinematics().pTMin / Q, 2)) ) , 
                                                          pow(Q / Vs, 2) / DH.GetKinematics().etaRange.first), 1.0);
                    xbmin = std::min( xbmax, std::max(xbmin, pow(Q / Vs, 2)/ DH.GetKinematics().etaRange.second ) );   
                  }
              }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");

              // Get Ki_map objects at the scale Q
              const std::map<int, apfel::DistributionOperator> Ki_map = TabKi.Evaluate(Q).GetObjects();

              // Compute integral of Ki in x and construct a set
              std::map<int, apfel::Operator> IntKi;
              for (auto const& tms : Ki_map)
                {
                  apfel::Operator cumulant = (_bins[i].Intx ? tms.second.Integrate(xbmin, xbmax) : tms.second.Evaluate(_bins[i].xav));
                  IntKi.insert({tms.first, cumulant});  
                };

              // Get evolution operator
              apfel::Set<apfel::Operator> Gammaij = TabGammaij->Evaluate(Q);

              // Intialise container for the FK table
              std::map<int, apfel::Operator> Nj_map;
              for (int j = 0; j < 13; j++)
                Nj_map.insert({j, Zero});

              // Compute the product of Ki and Gammaij and adjust the
              // convolution basis
              for (int j = 0; j < 13; j++)
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) != 0)
                    Nj_map.at(j) += IntKi.at(i) * Gammaij.at(apfel::Gkj.at({i, j}));

              // Return the result
              return apfel::Set<apfel::Operator> {_cmap, Nj_map};
            };

            // Tabulate cross section
            TabIncQIntegrand = std::unique_ptr<apfel::TabulateObject<double>> {new apfel::TabulateObject<double> {IncQIntegrand, 100, 0.9 * Qmin, 1.1 * Qmax, 3, _Thresholds}};
            TabSemiIncQIntegrand = std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>>
                                   (new apfel::TabulateObject<apfel::Set<apfel::Operator>> {Nj, 100, 0.9 * Qmin, 1.1 * Qmax, 3, _Thresholds});
            // Push back multiplicities
            if (!DH.GetNormalised())
            {
              if (_bins[i].IntQ)
                _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Integrate(Qmin, Qmax)});
              else
                _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Evaluate(_bins[i].Qav)});
            }
            else 
            {
              if (_bins[i].IntQ)
              _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Integrate(Qmin, Qmax) / TabIncQIntegrand->Integrate(Qmin, Qmax)});
            else
              _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Evaluate(_bins[i].Qav) / TabIncQIntegrand->Evaluate(_bins[i].Qav)});
            }
            

            xl = _bins[i].xmin;
            xu = _bins[i].xmax;
            xc = _bins[i].xav;
            Ql = Qmin;
            Qu = Qmax;
            Qc = _bins[i].Qav;
          }      
     }
    else
      throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Process.");
  }

  //_________________________________________________________________________
  PredictionsHandler::PredictionsHandler(PredictionsHandler                             const& PH,
                                         std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    NangaParbat::ConvolutionTable{},
    _mu0(PH._mu0),
    _Thresholds(PH._Thresholds),
    _gx(PH._gx),
    _gz(PH._gz),
    _obs(PH._obs),
    _bins(PH._bins),
    _qTfact(PH._qTfact),
    _cmap(PH._cmap),
    _ChargeMap(PH._ChargeMap)
  {
    // Set cuts in the mather class
    _cuts = PH._cuts;

    // Compute total cut mask as a product of single masks
    _cutmask = PH._cutmask;
    for (auto const& c : cuts)
      _cutmask *= c->GetMask();

    // Impose new cuts
    _FKt.resize(_bins.size());
    for (int i = 0; i < (int) _bins.size(); i++)
      _FKt[i] = (_cutmask[i] ? PH._FKt[i] : apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
  }

  //_________________________________________________________________________
  void PredictionsHandler::SetInputFFs(std::function<apfel::Set<apfel::Distribution>(double const&)> const& InDistFunc)
  {
    // Construct set of distributions
    _D = apfel::Set<apfel::Distribution> {_ChargeMap * InDistFunc(_mu0)};
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&) const
  {
    // Initialise vector of predictions
    std::vector<double> preds(_bins.size());

    // Compute predictions by convoluting the precomputed kernels with
    // the initial-scale FFs and then perform the integration in
    // z. Finally Divide by the bin width in z.
    for (int id = 0; id < (int) _bins.size(); id++)
      if (_bins[id].Intz)
        preds[id] = (_cutmask[id] ? ((_FKt[id] * _D).Combine() * [] (double const& z) -> double{ return 1 / z; }).Integrate(_bins[id].zmin, _bins[id].zmax)
                     / ( _bins[id].zmax - _bins[id].zmin ) * _qTfact[id] : 0);
      else
        preds[id] = (_cutmask[id] ? (_FKt[id] * _D).Combine().Evaluate(_bins[id].zav) / _bins[id].zav * _qTfact[id] : 0);
    return preds;
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&,
                                                         std::function<double(double const&, double const&, double const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&,
                                                         std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }
}