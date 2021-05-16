# build exacutable
add_executable(ClosureTests ClosureTests.cc)
target_link_libraries(ClosureTests MontBlanc)

add_executable(MCReplicaGeneration MCReplicaGeneration.cc)
target_link_libraries(MCReplicaGeneration MontBlanc)

add_executable(Reweighting Reweighting.cc)
target_link_libraries(Reweighting MontBlanc)
