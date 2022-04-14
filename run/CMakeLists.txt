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
