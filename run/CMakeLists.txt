# build run
file(GLOB runcodes *.cc)
foreach(testsource ${runcodes})
  GET_FILENAME_COMPONENT(filename ${testsource} NAME_WE)
  add_executable(${filename} ${filename}.cc)
  target_link_libraries(${filename} MontBlanc)
  add_test(${filename} ${filename})
endforeach()
