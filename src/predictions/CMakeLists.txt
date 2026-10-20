set(predictions_source
  predictionshandler.cc
  predictionshandler_approx.cc
  predictionshandlerdata.cc
  )

add_library(predictions OBJECT ${predictions_source})
