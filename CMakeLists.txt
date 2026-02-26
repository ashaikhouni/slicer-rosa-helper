cmake_minimum_required(VERSION 3.16.3...3.27 FATAL_ERROR)

project(RosaHelper)

# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/<org-or-user>/slicer-rosa-helper")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Ammar Shaikhouni (project lead)")
set(EXTENSION_DESCRIPTION "Load ROSA case folders (Analyze volumes + trajectories) into 3D Slicer with centered volumes and registration transforms.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(RosaHelper)
add_subdirectory(ShankDetect)
add_subdirectory(ExportCenter)

# Install extension-level shared Python libraries used by multiple modules.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CommonLib/
  DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}/CommonLib
  COMPONENT Runtime
)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
