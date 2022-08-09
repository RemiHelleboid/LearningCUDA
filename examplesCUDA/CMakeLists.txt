message("Configuring CUDA examples ... ")

file(GLOB EXAMPLES *.cu)
foreach(EXAMPLE ${EXAMPLES})
  get_filename_component(EXAMPLENAME ${EXAMPLE} NAME_WE)
  message("\t - ${EXAMPLENAME}")
  add_executable(${EXAMPLENAME} ${EXAMPLE})
endforeach()
