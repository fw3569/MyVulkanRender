cmake_minimum_required(VERSION 4.0)
# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_TOOLCHAIN_FILE "D:/Project/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
project(Test)
add_executable(test src/test.cpp)

find_package(Vulkan REQUIRED)
target_link_libraries(test
PUBLIC
Vulkan::Vulkan
)
target_sources(test
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS
        "${Vulkan_INCLUDE_DIR}"
        FILES
        "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
find_package(glfw3 REQUIRED)
target_link_libraries(test PRIVATE glfw)
find_package(imgui CONFIG REQUIRED)
target_link_libraries(test PRIVATE imgui::imgui)

function(add_slang_shader_target TARGET)
  cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
  set(SHADERS_DIR "${CMAKE_CURRENT_LIST_DIR}/gen_shaders")
  set(SHADERS_PATH "${SHADERS_DIR}/slang.spv")
  # set(SHADERS_PATH "${SHADERS_DIR}/slang.spv" CACHE STRING "" FORCE)
  set(ENTRY_POINTS -entry vertShadowmap -entry fragShadowmap -entry vertMain -entry fragMain -entry vertLighting -entry fragLighting -entry vertBloomDownsample -entry fragBloomDownsample -entry vertBloomUpsample -entry fragBloomUpsample -entry compParticle -entry vertParticle -entry fragParticle)
  target_compile_options(test PRIVATE "-D SHADER_FILE_PATH=\"${SHADERS_PATH}\"")
  add_custom_command(
    OUTPUT "${SHADERS_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADERS_DIR}"
  )
  add_custom_command(
    OUTPUT "${SHADERS_PATH}"
    COMMAND "D:/Project/VulkanSDK/1.4.313.1/Bin/slangc.exe" ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
    WORKING_DIRECTORY "${SHADERS_DIR}"
    DEPENDS "${SHADERS_DIR}" ${SHADER_SOURCES}
    COMMENT "Compiling Slang Shaders"
    VERBATIM
  )
  add_custom_target("${TARGET}" DEPENDS "${SHADERS_DIR}/slang.spv")
endfunction()

add_slang_shader_target(slang_shaders SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/test_shader.slang")
add_dependencies(test slang_shaders)

target_compile_options(test PRIVATE "-D DATA_FILE_PATH=\"${CMAKE_CURRENT_LIST_DIR}/data\"")
target_compile_options(test PRIVATE "-DVULKAN_HPP_NO_STRUCT_CONSTRUCTORS")
set_target_properties(test PROPERTIES CXX_STANDARD 20)
