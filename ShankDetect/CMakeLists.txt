set(MODULE_NAME ShankDetect)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  WITH_GENERIC_TESTS
)

# Install reusable core Python package for this module.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Lib/
  DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}/${MODULE_NAME}/Lib
  COMPONENT Runtime
)
