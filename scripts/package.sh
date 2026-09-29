#!/usr/bin/env bash
# Builds the prebuilt release archive: the layer, its manifest (pointing at the
# library by a relative path) and install.sh, which enables the layer for the
# user's vrcompositor.
#
#   scripts/package.sh [--label rc.1] [--output DIR]
#
# Built in the Steam Runtime 3 (sniper) SDK, the layer runs on any distribution
# with that glibc or newer. The compiler comes from CC / CXX.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
label=""
output="$root/dist"

usage() {
    echo "usage: $0 [--label LABEL] [--output DIR]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || usage
    case "$1" in
        --label) label=$2 ;;  # may be empty: a final release
        --output) output=$2 ;;
        *) usage ;;
    esac
    shift 2
done

if [[ -n "$label" && ! "$label" =~ ^[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*$ ]]; then
    echo "invalid pre-release label: $label" >&2
    exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cmake -S "$root" -B "$work/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DSTEAMVR_COMPOSITOR_SYNC_BUILD_TESTS=OFF \
    -DSTEAMVR_COMPOSITOR_SYNC_INSTALL_OVERRIDE=OFF \
    -DSTEAMVR_COMPOSITOR_SYNC_RELOCATABLE=ON \
    -DSTEAMVR_COMPOSITOR_SYNC_VERSION_LABEL="$label"
cmake --build "$work/build"

version=$(sed -n 's/^CMAKE_PROJECT_VERSION:STATIC=//p' "$work/build/CMakeCache.txt")
version+=${label:+-$label}
name="steamvr-compositor-sync-$version-linux-$(uname -m)"
stage="$work/stage/$name"

cmake --install "$work/build" --prefix "$stage"
install -m 755 "$root/scripts/install.sh" "$stage/install.sh"
install -m 644 "$root/README.md" "$root/LICENSE" "$stage/"

# The layer must carry its own C++ runtime (vrcompositor runs with the Steam
# runtime's old libstdc++) and depend on nothing but glibc.
# Captured first: grep -q ending a pipeline early fails it under pipefail.
library="$stage/lib/libVkLayer_steamvr_compositor_sync.so"
needed=$(readelf -d "$library" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | sort | tr '\n' ' ')
symbols=$(objdump -T "$library")
glibc=$(grep -oE 'GLIBC_[0-9.]+' <<<"$symbols" | sort -Vu | tail -1)
if grep -qE 'GLIBCXX|CXXABI' <<<"$symbols"; then
    echo "error: the layer links libstdc++ dynamically" >&2
    exit 1
fi
for dependency in $needed; do
    case "$dependency" in
        libc.so.6 | libm.so.6 | ld-linux-*.so.*) ;;
        *) echo "error: unexpected dependency $dependency" >&2; exit 1 ;;
    esac
done
echo "layer needs: $needed(newest symbol $glibc)"

# Reproducible archive: fixed owners, order and timestamps.
epoch=${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct 2>/dev/null || date +%s)}
mkdir -p "$output"
archive="$output/$name.tar.gz"
tar -C "$work/stage" --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" -cf - "$name" | gzip -9n >"$archive"
(cd "$output" && sha256sum "$name.tar.gz" >"$name.tar.gz.sha256")
echo "wrote $archive"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    sdk=$(sed -n 's/^BUILD_ID="\?\([^"]*\)"\?$/\1/p' /etc/os-release 2>/dev/null || true)
    {
        echo "version=$version"
        echo "archive=$name.tar.gz"
        echo "glibc=${glibc#GLIBC_}"
        echo "sdk=$sdk"
    } >>"$GITHUB_OUTPUT"
fi
