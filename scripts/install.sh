#!/usr/bin/env bash
# Installs the prebuilt layer for the current user and enables it for SteamVR's
# vrcompositor, which it finds through ~/.config/openvr/openvrpaths.vrpath.
#
#   ./install.sh [--prefix DIR] [--vrcompositor PATH]... [--dry-run] [--force]
#   ./install.sh --uninstall [--prefix DIR] [--dry-run]
#
# It only ever writes or removes its own three files, and never replaces a file
# that is not its own. When other Vulkan loader configuration would conflict,
# it changes nothing unless given --force.

# POSIX sh up to here, so another shell stops with a clear message.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "error: run this with bash: ./install.sh" >&2
    exit 1
fi
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
    echo "error: bash 4.4 or newer is required" >&2
    exit 1
fi
set -euo pipefail

readonly layer_name="VK_LAYER_STEAMVR_compositor_sync"
readonly library_file="libVkLayer_steamvr_compositor_sync.so"
readonly manifest_file="VkLayer_steamvr_compositor_sync.json"
readonly override_file="VkLayer_steamvr_compositor_sync_override.json"

here=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prefix="$HOME/.local"
uninstall=false
dry_run=false
force=false
vrcompositors=()

usage() {
    sed -n '2,6s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix | --vrcompositor)
            [[ $# -ge 2 && -n "$2" ]] || usage 2
            if [[ "$1" == --prefix ]]; then prefix=$2; else vrcompositors+=("$2"); fi
            shift 2
            ;;
        --uninstall) uninstall=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --force) force=true; shift ;;
        -h | --help) usage ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

if [[ $EUID -eq 0 ]]; then
    echo "error: run this as the user who runs SteamVR, not as root" >&2
    exit 1
fi
if [[ "$prefix" != /* ]]; then
    prefix="$PWD/$prefix"
fi

library="$prefix/lib/$library_file"
manifest="$prefix/share/vulkan/explicit_layer.d/$manifest_file"
override="$prefix/share/vulkan/implicit_layer.d/$override_file"

# Why an existing file at one of the three paths is not ours to replace or
# remove (nothing if it is: a regular file naming this layer).
foreign() {
    if [[ -L "$1" ]]; then
        echo "it is a symbolic link"
    elif [[ ! -f "$1" ]]; then
        echo "it is not a regular file"
    elif ! grep -qaF "$layer_name" "$1"; then
        echo "it does not belong to $layer_name"
    fi
}

# --- uninstall: remove this layer's files, and only them --------------------

if $uninstall; then
    for file in "$library" "$manifest" "$override"; do
        [[ -e "$file" || -L "$file" ]] || continue
        reason=$(foreign "$file")
        if [[ -n "$reason" ]]; then
            echo "skipped $file: $reason" >&2
            continue
        fi
        if $dry_run; then echo "would remove $file"; else rm -f "$file" && echo "removed $file"; fi
    done
    exit 0
fi

if [[ ! -f "$here/lib/$library_file" ]]; then
    echo "error: run install.sh from the extracted release archive (or build from source with CMake)" >&2
    exit 1
fi

# --- find vrcompositor: given, or those of the registered SteamVR runtimes ---

config_home=${XDG_CONFIG_HOME:-$HOME/.config}
data_home=${XDG_DATA_HOME:-$HOME/.local/share}
if [[ ${#vrcompositors[@]} -eq 0 ]]; then
    registry="$config_home/openvr/openvrpaths.vrpath"
    runtimes=()
    if [[ -f "$registry" ]]; then
        # The strings of the "runtime" array, however the JSON is laid out.
        mapfile -t runtimes < <(tr -d '\n' <"$registry" | grep -o '"runtime"[[:space:]]*:[[:space:]]*\[[^]]*\]' |
            grep -o '"[^"]*"' | tail -n +2 | tr -d '"')
    fi
    runtimes+=("$data_home/Steam/steamapps/common/SteamVR")
    for runtime in "${runtimes[@]}"; do
        if [[ -x "$runtime/bin/linux64/vrcompositor" ]]; then
            vrcompositors+=("$runtime/bin/linux64/vrcompositor")
        fi
    done
fi

# The loader compares app_keys with the resolved path of the running executable.
declare -A seen=()
app_keys=()
for path in "${vrcompositors[@]}"; do
    if [[ "$path" == *[[:cntrl:]]* ]]; then
        echo "warning: skipped a path containing control characters" >&2
        continue
    fi
    if [[ ! -e "$path" ]]; then
        echo "warning: $path does not exist" >&2
        continue
    fi
    path=$(readlink -f "$path")
    if [[ "$(basename "$path")" != vrcompositor ]]; then
        echo "warning: $path is not vrcompositor; the layer only activates in vrcompositor" >&2
    fi
    if [[ -z "${seen[$path]:-}" ]]; then
        seen[$path]=1
        app_keys+=("$path")
    fi
done
if [[ ${#app_keys[@]} -eq 0 ]]; then
    echo "error: SteamVR not found. Run SteamVR once, or pass --vrcompositor PATH." >&2
    exit 1
fi

# --- check before changing anything --------------------------------------------

# Never replace a file that is not ours, not even with --force.
for file in "$library" "$manifest" "$override"; do
    [[ -e "$file" || -L "$file" ]] || continue
    reason=$(foreign "$file")
    if [[ -n "$reason" ]]; then
        echo "error: nothing changed: $file already exists and $reason" >&2
        exit 1
    fi
done

problems=()
IFS=: read -ra config_dirs <<<"${XDG_CONFIG_DIRS:-/etc/xdg}"
IFS=: read -ra data_dirs <<<"${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"

searched=false
for root in "$data_home" "${data_dirs[@]}"; do
    if [[ "$(readlink -m "$root")" == "$(readlink -m "$prefix/share")" ]]; then
        searched=true
    fi
done
if ! $searched; then
    problems+=("the Vulkan loader does not search $prefix/share, so the layer would not load from there")
fi

# The loader applies one override layer per application, preferring one that
# names it over a global one.
for root in "$config_home" "${config_dirs[@]}" /etc "$data_home" "${data_dirs[@]}"; do
    for other in "$root"/vulkan/implicit_layer.d/*.json; do
        [[ -f "$other" && "$(readlink -m "$other")" != "$(readlink -m "$override")" ]] || continue
        if [[ "$(basename "$other")" == "$override_file" ]]; then
            if [[ "$other" == */share/vulkan/implicit_layer.d/* ]]; then
                problems+=("an earlier install of this layer is at $other; remove it first with" \
                    "--uninstall --prefix ${other%/share/vulkan/implicit_layer.d/*}")
            else
                problems+=("an earlier install of this layer is at $other; remove it first")
            fi
            continue
        fi
        grep -q '"VK_LAYER_LUNARG_override"' "$other" || continue
        if ! grep -q '"app_keys"' "$other"; then
            problems+=("$other is a global override layer (e.g. from vkconfig); vrcompositor would get this" \
                "layer's override instead, and no longer the layers that one enables")
        fi
        for path in "${app_keys[@]}"; do
            if grep -qF "\"$path\"" "$other"; then
                problems+=("$other also names vrcompositor; the loader would use only one of the two")
                break
            fi
        done
    done
done

# The loader reads the first settings file it finds; one without
# unordered_layer_location replaces every other layer source.
for root in "$config_home" "$data_home" "${config_dirs[@]}" /etc "${data_dirs[@]}"; do
    settings="$root/vulkan/loader_settings.d/vk_loader_settings.json"
    [[ -f "$settings" ]] || continue
    if ! grep -q 'unordered_layer_location' "$settings"; then
        problems+=("$settings (e.g. from vkconfig) may keep the layer from loading: it has no" \
            "unordered_layer_location entry")
    fi
    break
done

if [[ ${#problems[@]} -gt 0 ]]; then
    for problem in "${problems[@]}"; do
        echo "warning: $problem" >&2
    done
    if ! $force; then
        echo "error: nothing changed. Resolve the above, or pass --force to install anyway." >&2
        exit 1
    fi
fi

# --- install ------------------------------------------------------------------

if $dry_run; then
    echo "would install $library"
    echo "would install $manifest"
    echo "would install $override, enabling the layer for:"
    printf '    %s\n' "${app_keys[@]}"
    exit 0
fi

# Each file is written aside and renamed into place, so nothing (a starting
# vrcompositor included) ever sees a partial file, and a running vrcompositor
# keeps the library it has mapped.
partials=()
trap 'rm -f -- "${partials[@]}"' EXIT
trap 'exit 1' HUP INT TERM
place() {
    mkdir -p -- "$(dirname -- "$2")"
    local partial
    partial=$(mktemp -- "$2.XXXXXX")
    partials+=("$partial")
    cat >"$partial"
    chmod "$1" -- "$partial"
    mv -f -- "$partial" "$2"
}

place 755 "$library" <"$here/lib/$library_file"
place 644 "$manifest" <"$here/share/vulkan/explicit_layer.d/$manifest_file"

api_version=$(sed -n 's/.*"api_version": *"\([^"]*\)".*/\1/p' "$manifest")
json_keys=""
for path in "${app_keys[@]}"; do
    path=${path//\\/\\\\}
    json_keys+="${json_keys:+, }\"${path//\"/\\\"}\""
done

# Last, so a failure before this point leaves nothing that loads.
place 644 "$override" <<EOF
{
    "file_format_version": "1.2.0",
    "layer": {
        "name": "VK_LAYER_LUNARG_override",
        "type": "GLOBAL",
        "api_version": "$api_version",
        "implementation_version": "1",
        "description": "Loads $layer_name into vrcompositor only.",
        "component_layers": [
            "$layer_name"
        ],
        "app_keys": [
            $json_keys
        ],
        "disable_environment": {
            "STEAMVR_COMPOSITOR_SYNC_DISABLE": "1"
        }
    }
}
EOF

echo "installed $library"
echo "installed $manifest"
echo "installed $override, enabling the layer for:"
printf '    %s\n' "${app_keys[@]}"
echo
echo "Restart SteamVR; ~/.steam/steam/logs/vrcompositor-linux.txt should then show:"
echo "    [steamvr-compositor-sync] ... active in vrcompositor"
undo=(--uninstall)
if [[ "$prefix" != "$HOME/.local" ]]; then
    undo+=(--prefix "$prefix")
fi
echo "To undo: $(printf '%q ' "$here/install.sh" "${undo[@]}")"
