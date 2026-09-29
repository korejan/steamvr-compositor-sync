cmake_minimum_required(VERSION 3.25...4.4)

project(steamvr_compositor_sync_layer
    VERSION 0.1.0
    DESCRIPTION "Vulkan layer that stops SteamVR's vrcompositor from recycling GPU work still in flight"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STEAMVR_COMPOSITOR_SYNC_BUILD_TESTS "Build the isolation tests" ON)
option(STEAMVR_COMPOSITOR_SYNC_SANITIZE "Build with AddressSanitizer, LeakSanitizer and UBSan (testing only)" OFF)
option(STEAMVR_COMPOSITOR_SYNC_INSTALL_OVERRIDE "Install the override that enables the layer for vrcompositor" ON)
set(STEAMVR_COMPOSITOR_SYNC_VRCOMPOSITOR_PATH "" CACHE STRING
    "vrcompositor executable(s) to enable the layer for (empty: detect SteamVR)")
option(STEAMVR_COMPOSITOR_SYNC_RELOCATABLE
    "Point the layer manifest at the library by a relative path (prebuilt packages)" OFF)
set(STEAMVR_COMPOSITOR_SYNC_VERSION_LABEL "" CACHE STRING "Pre-release label appended to the version, e.g. rc.1")

if(STEAMVR_COMPOSITOR_SYNC_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

include(GNUInstallDirs)
include(cmake/SteamVR.cmake)
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

set(layer_name "VK_LAYER_STEAMVR_compositor_sync")
set(library "VkLayer_steamvr_compositor_sync")
# Manifest api_version: the Vulkan headers the layer is built against.
set(api_version "${Vulkan_VERSION}")
if(NOT api_version)
    set(api_version "1.3.239")
endif()
set(version "${PROJECT_VERSION}")
if(STEAMVR_COMPOSITOR_SYNC_VERSION_LABEL)
    string(APPEND version "-${STEAMVR_COMPOSITOR_SYNC_VERSION_LABEL}")
endif()
set(warnings -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
set(layer_name_definition STEAMVR_COMPOSITOR_SYNC_LAYER_NAME="${layer_name}")

# Writes manifest/<name>.json.in to <dir>/<name>.json. The template takes
# @layer_name@ and @api_version@ from this file, @library_path@ and @app_keys@
# from the arguments of those names; generator expressions are allowed.
function(steamvr_compositor_sync_manifest name dir)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "library_path;app_keys" "")
    set(library_path "${arg_library_path}")
    set(app_keys "${arg_app_keys}")
    file(READ manifest/${name}.json.in content)
    string(CONFIGURE "${content}" content @ONLY)
    file(GENERATE OUTPUT ${dir}/${name}.json CONTENT "${content}")
endfunction()

# --- the layer -------------------------------------------------------------

add_library(${library} SHARED
    src/layer.cpp
    src/tracker.cpp)
target_include_directories(${library} PRIVATE src)
target_link_libraries(${library} PRIVATE Vulkan::Headers Threads::Threads)
target_compile_definitions(${library} PRIVATE
    ${layer_name_definition}
    STEAMVR_COMPOSITOR_SYNC_VERSION="${version}")
target_compile_options(${library} PRIVATE ${warnings})
set_target_properties(${library} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_link_options(${library} PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map)
set_property(TARGET ${library} APPEND PROPERTY
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map)

# vrcompositor runs with the Steam runtime's libraries, whose libstdc++ can be
# older than the one we build against: carry our own, and refuse to link with
# unresolved symbols.
if(NOT STEAMVR_COMPOSITOR_SYNC_SANITIZE)
    target_link_options(${library} PRIVATE -static-libstdc++ -static-libgcc -Wl,-z,defs)
endif()

install(TARGETS ${library} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Explicit layer manifest: nothing loads the layer unless it is enabled by
# name.
set(install_manifests ${CMAKE_BINARY_DIR}/install-manifest)
set(library_path "${CMAKE_INSTALL_FULL_LIBDIR}/lib${library}.so")
if(STEAMVR_COMPOSITOR_SYNC_RELOCATABLE)
    # The loader resolves a relative library_path against the manifest's directory.
    file(RELATIVE_PATH library_path "${CMAKE_INSTALL_FULL_DATADIR}/vulkan/explicit_layer.d" "${library_path}")
endif()
steamvr_compositor_sync_manifest(${library} ${install_manifests} library_path "${library_path}")
install(FILES ${install_manifests}/${library}.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/vulkan/explicit_layer.d)

# vrcompositor inherits none of SteamVR's environment, so the layer is enabled
# for it by an override meta layer whose app_keys name its executable: by
# default, every SteamVR runtime registered in openvrpaths.vrpath.
if(STEAMVR_COMPOSITOR_SYNC_INSTALL_OVERRIDE)
    if(STEAMVR_COMPOSITOR_SYNC_VRCOMPOSITOR_PATH)
        set(vrcompositor_paths "")
        foreach(path IN LISTS STEAMVR_COMPOSITOR_SYNC_VRCOMPOSITOR_PATH)
            if(NOT EXISTS "${path}")
                message(WARNING "${path} does not exist")
            endif()
            file(REAL_PATH "${path}" path)
            list(APPEND vrcompositor_paths "${path}")
        endforeach()
    else()
        steamvr_compositor_sync_find_vrcompositor(vrcompositor_paths)
    endif()

    if(vrcompositor_paths)
        list(JOIN vrcompositor_paths ", " listed)
        message(STATUS "Enabling the layer for: ${listed}")
        steamvr_compositor_sync_json_strings(app_keys ${vrcompositor_paths})
        steamvr_compositor_sync_manifest(${library}_override ${install_manifests} app_keys "${app_keys}")
        set(override_dir "${CMAKE_INSTALL_DATADIR}/vulkan/implicit_layer.d")
        install(FILES ${install_manifests}/${library}_override.json DESTINATION ${override_dir})
        install(CODE "
            set(STEAMVR_COMPOSITOR_SYNC_APP_KEYS [[${vrcompositor_paths}]])
            set(STEAMVR_COMPOSITOR_SYNC_OWN_OVERRIDE
                \"\${CMAKE_INSTALL_PREFIX}/${override_dir}/${library}_override.json\")
            include([[${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckLoaderConflicts.cmake]])")
    else()
        message(WARNING "SteamVR not found: installing without the override, so the layer will not load "
                        "into vrcompositor. Run SteamVR once and configure again, or set "
                        "STEAMVR_COMPOSITOR_SYNC_VRCOMPOSITOR_PATH.")
    endif()
endif()

# --- isolation tests -------------------------------------------------------

if(STEAMVR_COMPOSITOR_SYNC_BUILD_TESTS)
    enable_testing()

    # The same manifests pointing at the build tree, so the tests load the
    # layer without installing anything.
    set(test_layer_dir ${CMAKE_BINARY_DIR}/test-layer)
    set(test_override_dir ${CMAKE_BINARY_DIR}/test-override)
    steamvr_compositor_sync_manifest(${library} ${test_layer_dir} library_path "$<TARGET_FILE:${library}>")
    steamvr_compositor_sync_manifest(${library}_override ${test_override_dir} app_keys "\"$<TARGET_FILE:sync_test>\"")

    # A test program linked against the loader that runs with the build tree's layer.
    function(steamvr_compositor_sync_test_program name)
        add_executable(${name} tests/${name}.cpp)
        target_link_libraries(${name} PRIVATE Vulkan::Vulkan Threads::Threads)
        target_compile_definitions(${name} PRIVATE ${layer_name_definition} ${ARGN})
        target_compile_options(${name} PRIVATE ${warnings})
        add_dependencies(${name} ${library})
    endfunction()

    set(test_env "VK_ADD_LAYER_PATH=${test_layer_dir}")
    set(forced "STEAMVR_COMPOSITOR_SYNC_FORCE=1")

    steamvr_compositor_sync_test_program(sync_test)
    target_include_directories(sync_test PRIVATE tests)

    # With the layer forced on: the misuse sequences must produce no
    # validation errors, and the layer must wait/swap as designed.
    add_test(NAME layer_fixes_misuse COMMAND sync_test --expect-clean)
    set_tests_properties(layer_fixes_misuse PROPERTIES
        ENVIRONMENT "${test_env};${forced};STEAMVR_COMPOSITOR_SYNC_VERBOSE=1"
        SKIP_RETURN_CODE 77)

    # Without the layer the same sequences must trip validation: proves the
    # test detects the bug it guards against.
    add_test(NAME misuse_detected_without_layer COMMAND sync_test --no-layer --expect-errors)
    set_tests_properties(misuse_detected_without_layer PROPERTIES
        ENVIRONMENT "${test_env}"
        SKIP_RETURN_CODE 77)

    # Not enabled by the application: the override's app_keys must load it.
    add_test(NAME layer_loaded_by_app_keys COMMAND sync_test --no-layer --expect-clean)
    set_tests_properties(layer_loaded_by_app_keys PROPERTIES
        ENVIRONMENT "${test_env};VK_IMPLICIT_LAYER_PATH=${test_override_dir};${forced}"
        SKIP_RETURN_CODE 77)

    # Layer loaded but the process is not vrcompositor: it must stay inert.
    add_test(NAME layer_inert_outside_vrcompositor COMMAND sync_test --expect-errors)
    set_tests_properties(layer_inert_outside_vrcompositor PROPERTIES
        ENVIRONMENT "${test_env}"
        SKIP_RETURN_CODE 77)

    # Per-call overhead of the layer; run by hand (timings do not pass or fail).
    steamvr_compositor_sync_test_program(bench STEAMVR_COMPOSITOR_SYNC_TEST_LAYER_DIR="${test_layer_dir}")

    # The steady-state paths must not allocate once warmed up. alloc_test
    # replaces malloc, as the sanitizers do, so it is left out of their builds.
    if(NOT STEAMVR_COMPOSITOR_SYNC_SANITIZE)
        steamvr_compositor_sync_test_program(alloc_test)
        add_test(NAME layer_hot_paths_do_not_allocate COMMAND alloc_test)
        set_tests_properties(layer_hot_paths_do_not_allocate PROPERTIES
            ENVIRONMENT "${test_env};${forced}"
            SKIP_RETURN_CODE 77)
    endif()
endif()
