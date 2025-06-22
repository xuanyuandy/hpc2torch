cmake_minimum_required(VERSION 3.16)

project(MyCUDAProject)
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

option(USE_CUDA "Enable CUDA compilation" OFF)
option(USE_CPU "Enable CPU-only compilation" OFF)

set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i.86)")
    message(STATUS "Target architecture: x86 (enabling AVX2/FMA)")
    set(ARCH_CXX_FLAGS "-mavx2 -mfma -O3")
    
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(ARCH_CXX_FLAGS "${ARCH_CXX_FLAGS} -mavx2")
    else()
        message(WARNING "Compiler does not support AVX2 instructions")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|arm.*)")
    message(STATUS "Target architecture: ARM (enabling NEON)")
    set(ARCH_CXX_FLAGS "-O3 -mcpu=generic+simd")
    
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
        set(ARCH_CXX_FLAGS "${ARCH_CXX_FLAGS} -mfpu=neon")
    else()
        message(WARNING "Compiler does not support NEON instructions")
    endif()
else()
    message(WARNING "Unknown architecture, using generic optimization flags")
    set(ARCH_CXX_FLAGS "-O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_CXX_FLAGS}") 
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB INCLUDE_SOURCE_FILES "include/cpu/**.cpp")
file(GLOB CPP_SOURCE_FILES "src/**/cpu/*.cpp")
list(APPEND CPP_SOURCE_FILES ${INCLUDE_SOURCE_FILES})

if(USE_CUDA)
  if(EXISTS "/usr/local/cuda")
    file(GLOB GPU_CUDA_FILES "src/**/gpu/*.cu")
    file(GLOB GPU_CUDNN_FILES "src/**/gpu/*.cpp")
    list(APPEND CUDA_SOURCE_FILES ${GPU_CUDA_FILES} ${GPU_CUDNN_FILES})
  else()
    file(GLOB_RECURSE GPU_CUDNN_FILES "src/**/gpu/*.cpp")
    file(GLOB_RECURSE GPU_CUDA_FILES "src/**/gpu/*.cu")
    list(APPEND CUDA_SOURCE_FILES ${GPU_CUDA_FILES} ${GPU_CUDNN_FILES})
    set(SRC_CUDA_CU_FILES ${GPU_CUDA_FILES} CACHE INTERNAL "DLCC .cu files")
  endif()
endif()

if(USE_CUDA)
  if(EXISTS "/usr/local/cuda")
    message(STATUS "Detected NVIDIA CUDA environment")
    enable_language(CUDA)
    list(APPEND ALL_SOURCE_FILES ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES})
    
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set_target_properties(my_library PROPERTIES
        CUDA_ARCHITECTURES 80
    )
    target_link_libraries(my_library ${CUDA_LIBRARIES} cudnn cublas)
  else()
    message(STATUS "Detected DLCC environment, configuring CUDA build")
    add_compile_definitions(USE_CUDA=1)
    set(DLCC_PATH /home/qy/Desktop/sdk/sdk/bin/dlcc)
    set(DLCC_CUDA_PATH /home/qy/Desktop/sdk/sdk)
    set(DLCC_ARCH dlgput64)
    set(SRC_CUDA_CU_FILES ${GPU_CUDA_FILES} CACHE INTERNAL "DLCC .cu files")
    set(SRC_CUDA_CPP_FILES ${GPU_CUDNN_FILES} CACHE INTERNAL "DLCC .cpp/.cc files")

    list(APPEND ALL_SOURCE_FILES ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})
    set(GENERATED_CU_OBJS "")
    foreach(cu_file ${SRC_CUDA_CU_FILES})
      get_filename_component(cu_name ${cu_file} NAME_WE)
      get_filename_component(cu_dir ${cu_file} DIRECTORY)
      string(REPLACE "${CMAKE_SOURCE_DIR}/" "" relative_dir ${cu_dir})
      set(obj_path "${CMAKE_CURRENT_BINARY_DIR}/${relative_dir}/${cu_name}.cu.o")
      get_filename_component(obj_dir ${obj_path} DIRECTORY)

      add_custom_command(
        OUTPUT ${obj_path}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${obj_dir}
        COMMAND ${DLCC_PATH} -c ${cu_file} -o ${obj_path}
                --cuda-path=${DLCC_CUDA_PATH}
                --cuda-gpu-arch=${DLCC_ARCH}
                -I${PROJECT_SOURCE_DIR}/include -O2 -std=c++17 -fPIC
        DEPENDS ${cu_file}
        COMMENT "Compiling ${cu_file} with dlcc"
        VERBATIM
      )
      list(APPEND GENERATED_CU_OBJS ${obj_path})
    endforeach()

    add_custom_target(cuda_objs ALL DEPENDS ${GENERATED_CU_OBJS})

    add_library(my_library SHARED ${ALL_SOURCE_FILES})
    add_dependencies(my_library cuda_objs)
    target_sources(my_library PRIVATE ${GENERATED_CU_OBJS})
    include_directories(${DLCC_CUDA_PATH}/include)
    target_include_directories(my_library PRIVATE ${DLCC_CUDA_PATH}/include)
    link_directories(${DLCC_CUDA_PATH}/lib)

    target_link_libraries(my_library
      ${DLCC_CUDA_PATH}/lib/libcurt.so
      ${DLCC_CUDA_PATH}/lib/libcublas.so
      ${DLCC_CUDA_PATH}/lib/libcudnn.so
    )
  endif()
elseif(USE_CPU)
    message(STATUS "CPU-only build enabled.")
    enable_language(CXX)
    list(APPEND ALL_SOURCE_FILES ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES})
else()
    message(FATAL_ERROR "No valid compilation mode specified. Please enable USE_CUDA, or USE_CPU.")
endif()


target_compile_features(my_library PUBLIC cxx_std_11)
target_link_libraries(my_library PRIVATE ${Python3_LIBRARIES})
set_target_properties(my_library PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
