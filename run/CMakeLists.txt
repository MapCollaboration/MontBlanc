# build exacutable
add_executable(ComputeChi2s ComputeChi2s.cc)
target_link_libraries(ComputeChi2s MontBlanc)

add_executable(Optimize Optimize.cc)
target_link_libraries(Optimize MontBlanc)

add_executable(NNGrid NNGrid.cc)
target_link_libraries(NNGrid MontBlanc)

add_executable(LHAPDFGrid LHAPDFGrid.cc)
target_link_libraries(LHAPDFGrid MontBlanc)

add_executable(Predictions Predictions.cc)
target_link_libraries(Predictions MontBlanc)

add_executable(Predictions_SIDIS Predictions_SIDIS.cc)
target_link_libraries(Predictions_SIDIS MontBlanc)

add_executable(Predictions_SIDIS_V2 Predictions_SIDIS_V2.cc)
target_link_libraries(Predictions_SIDIS_V2 MontBlanc)

add_executable(Reweighting_SIDIS reweighting.cc)
target_link_libraries(Reweighting_SIDIS MontBlanc -lgsl -lgslcblas)

