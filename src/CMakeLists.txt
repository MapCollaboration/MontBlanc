add_subdirectory(predictions)
add_subdirectory(parameterisations)
add_subdirectory(chisquare)

option(SHARED "Build shared-libray instead of static-libray" ON)
if(SHARED)
  add_library(MontBlanc SHARED $<TARGET_OBJECTS:predictions> $<TARGET_OBJECTS:parameterisations> $<TARGET_OBJECTS:chisquare>)
else(SHARED)
  add_library(MontBlanc STATIC $<TARGET_OBJECTS:predictions> $<TARGET_OBJECTS:parameterisations> $<TARGET_OBJECTS:chisquare>)
endif(SHARED)

target_link_libraries(MontBlanc ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LIBRARIES} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS} ${APFELXX_LIBRARIES} ${NANGAPARBAT_LIBRARIES} ${LHAPDF_LIBRARIES} ${GSL_LIBRARIES})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/MontBlanc DESTINATION include)
install(TARGETS MontBlanc DESTINATION lib)
