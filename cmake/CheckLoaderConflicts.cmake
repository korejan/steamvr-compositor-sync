# Run by `cmake --install`: warns about Vulkan loader configuration that keeps
# the installed override from applying to vrcompositor, or that the override
# displaces. Expects STEAMVR_COMPOSITOR_SYNC_APP_KEYS (the vrcompositor paths)
# and STEAMVR_COMPOSITOR_SYNC_OWN_OVERRIDE (the manifest being installed).
# Mirrors the loader's rules (loader.c remove_all_non_valid_override_layers,
# settings.c get_loader_settings).

# A staged install (packaging) says nothing about the target system.
if(NOT "$ENV{DESTDIR}" STREQUAL "")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/SteamVR.cmake")

steamvr_compositor_sync_user_dir(config_home XDG_CONFIG_HOME .config)
steamvr_compositor_sync_user_dir(data_home XDG_DATA_HOME .local/share)
set(config_dirs "$ENV{XDG_CONFIG_DIRS}")
if(config_dirs STREQUAL "")
    set(config_dirs "/etc/xdg")
endif()
set(data_dirs "$ENV{XDG_DATA_DIRS}")
if(data_dirs STREQUAL "")
    set(data_dirs "/usr/local/share:/usr/share")
endif()
string(REPLACE ":" ";" config_dirs "${config_dirs}")
string(REPLACE ":" ";" data_dirs "${data_dirs}")

# True if the JSON string array at <json> <path...> names a vrcompositor path.
function(steamvr_compositor_sync_names_vrcompositor out json)
    set(${out} FALSE PARENT_SCOPE)
    string(JSON count ERROR_VARIABLE error LENGTH "${json}" ${ARGN})
    if(error OR count EQUAL 0)
        return()
    endif()
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
        string(JSON key GET "${json}" ${ARGN} ${i})
        if(key IN_LIST STEAMVR_COMPOSITOR_SYNC_APP_KEYS)
            set(${out} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# Override layers: the loader applies one per application, preferring one
# whose app_keys name it over a global one (no or empty app_keys).
file(REAL_PATH "${STEAMVR_COMPOSITOR_SYNC_OWN_OVERRIDE}" own_override)
foreach(root IN ITEMS "${config_home}" ${config_dirs} "/etc" "${data_home}" ${data_dirs})
    file(GLOB manifests "${root}/vulkan/implicit_layer.d/*.json")
    foreach(manifest IN LISTS manifests)
        file(REAL_PATH "${manifest}" resolved)
        if(resolved STREQUAL own_override)
            continue()
        endif()
        file(READ "${manifest}" json)
        string(JSON name ERROR_VARIABLE error GET "${json}" layer name)
        if(error OR NOT name STREQUAL "VK_LAYER_LUNARG_override")
            continue()
        endif()
        string(JSON count ERROR_VARIABLE error LENGTH "${json}" layer app_keys)
        if(error OR count EQUAL 0)
            message(WARNING "${manifest} is a global override layer (vkconfig's \"all applications\"). "
                            "The loader prefers an override naming the application, so vrcompositor gets "
                            "this project's override instead and no longer the layers that one enables.")
            continue()
        endif()
        steamvr_compositor_sync_names_vrcompositor(conflicts "${json}" layer app_keys)
        if(conflicts)
            message(WARNING "${manifest} is another override layer naming vrcompositor. The loader uses "
                            "only one of them (whichever it finds first), so this layer may not load. "
                            "Remove one of the two.")
        endif()
    endforeach()
endforeach()

# Loader settings (newer vkconfig): the loader reads the first file it finds,
# and in it the entry naming the application, else the first without
# app_keys. An entry that configures layers without keeping the other ones
# ("unordered_layer_location") makes the loader ignore every override.
set(settings_file "")
foreach(root IN ITEMS "${config_home}" "${data_home}" ${config_dirs} "/etc" ${data_dirs})
    if(EXISTS "${root}/vulkan/loader_settings.d/vk_loader_settings.json")
        set(settings_file "${root}/vulkan/loader_settings.d/vk_loader_settings.json")
        break()
    endif()
endforeach()
if(settings_file)
    file(READ "${settings_file}" json)
    string(JSON count ERROR_VARIABLE error LENGTH "${json}" settings_array)
    set(entries "")
    if(error)
        set(entries "settings")
    elseif(count GREATER 0)
        math(EXPR last "${count} - 1")
        foreach(i RANGE ${last})
            list(APPEND entries "settings_array:${i}")
        endforeach()
    endif()

    set(selected "")
    set(global "")
    foreach(entry IN LISTS entries)
        string(REPLACE ":" ";" entry "${entry}")
        string(JSON type ERROR_VARIABLE error TYPE "${json}" ${entry} app_keys)
        if(error)
            if(global STREQUAL "")
                set(global "${entry}")
            endif()
            continue()
        endif()
        steamvr_compositor_sync_names_vrcompositor(named "${json}" ${entry} app_keys)
        if(named)
            set(selected "${entry}")
            break()
        endif()
    endforeach()
    if(selected STREQUAL "")
        set(selected "${global}")
    endif()

    if(NOT selected STREQUAL "")
        string(JSON layers ERROR_VARIABLE error GET "${json}" ${selected} layers)
        if(NOT error AND NOT layers MATCHES "unordered_layer_location")
            message(WARNING "${settings_file} configures the layers for vrcompositor without keeping the "
                            "other layers (no \"unordered_layer_location\" entry), so the loader ignores "
                            "this layer's override and the layer will not load.")
        endif()
    endif()
endif()
