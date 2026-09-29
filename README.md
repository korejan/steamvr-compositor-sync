# steamvr-compositor-sync

A Vulkan layer that stops SteamVR's `vrcompositor` on Linux from reusing GPU
work that is still executing, which freezes the headset
([ValveSoftware/SteamVR-for-Linux#952](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/952)).

## The problem

vrcompositor recycles command buffers and resets descriptor pools while the
GPU may still be running work that uses them. The validation layer reports it
continuously (`VUID-vkBeginCommandBuffer-commandBuffer-00049`,
`VUID-vkQueueSubmit-pCommandBuffers-00071`,
`VUID-vkResetDescriptorPool-descriptorPool-00313`).

Usually the GPU has finished anyway. When it is behind (startup, a headset
connecting, level loads, heavy load), it runs half-overwritten command buffers
or stale descriptors: the GPU faults (`NVRM: Xid 32/13/31` on NVIDIA),
vrcompositor loses its device and the headset freezes. It affects every
SteamVR driver: Steam Link, ALVR and native headsets alike.

## Install

Run SteamVR once first, so the installer can find it.

**Prebuilt** (x86_64, any distribution with glibc 2.25 or newer, SteamOS
included): download the archive from
[Releases](https://github.com/korejan/steamvr-compositor-sync/releases), then

```bash
tar -xzf steamvr-compositor-sync-*-linux-x86_64.tar.gz
cd steamvr-compositor-sync-*-linux-x86_64
./install.sh
```

It only ever writes (and `--uninstall` only removes) its own three files, and
changes nothing when other Vulkan configuration would conflict (see below)
unless given `--force`. `--dry-run` shows what it would do; `--help` lists
the options.

**From source** (CMake ≥ 3.25, a C++20 compiler, Vulkan headers and loader):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build
cmake --install build
```

Either way, this installs the layer (`VK_LAYER_STEAMVR_compositor_sync`) and a
small override manifest that enables it for vrcompositor only:

- `~/.local/lib/libVkLayer_steamvr_compositor_sync.so`
- `~/.local/share/vulkan/explicit_layer.d/VkLayer_steamvr_compositor_sync.json`
- `~/.local/share/vulkan/implicit_layer.d/VkLayer_steamvr_compositor_sync_override.json`

No launch options are needed (vrcompositor does not receive SteamVR's launch
options). Restart SteamVR, then check
`~/.steam/steam/logs/vrcompositor-linux.txt` for:

```
[steamvr-compositor-sync] v0.1.0 active in vrcompositor: ...
```

Afterwards, a summary line (at most every 10 s, only when something changed)
counts the reuse the layer prevented.

### Configuration

The installer (and the build) finds vrcompositor through
`~/.config/openvr/openvrpaths.vrpath`, where SteamVR registers its location.
After moving SteamVR to another library, install again; `install.sh
--vrcompositor PATH` names the executable directly. CMake options:

| option | effect |
|---|---|
| `STEAMVR_COMPOSITOR_SYNC_VRCOMPOSITOR_PATH` | vrcompositor executable(s) to enable the layer for, instead of detecting them |
| `STEAMVR_COMPOSITOR_SYNC_INSTALL_OVERRIDE=OFF` | install the layer without the override |
| `STEAMVR_COMPOSITOR_SYNC_RELOCATABLE=ON` | the layer manifest finds the library by a relative path (prebuilt packages) |
| `STEAMVR_COMPOSITOR_SYNC_VERSION_LABEL` | pre-release label appended to the version, e.g. `rc.1` |

Environment variables, for running vrcompositor or the tests by hand
(vrcompositor started by SteamVR does not see them):

| variable | effect |
|---|---|
| `STEAMVR_COMPOSITOR_SYNC_DISABLE=1` | do not apply the override |
| `STEAMVR_COMPOSITOR_SYNC_FORCE=1` | activate in any process (testing) |
| `STEAMVR_COMPOSITOR_SYNC_TIMEOUT_MS` | bound on each wait, default 1000, at most 60000 |
| `STEAMVR_COMPOSITOR_SYNC_MAX_RETIRED_POOLS` | busy pools kept per application pool before resets wait, default 16 |
| `STEAMVR_COMPOSITOR_SYNC_VERBOSE=1` | log every wait |

### Conflicts with other Vulkan configuration

The loader applies only one override layer per application. `install.sh`
refuses to install (and `cmake --install` warns) when something else
interferes:

- another override naming vrcompositor: only one of the two applies;
- a global override (vkconfig's "all applications"): vrcompositor gets this
  one instead, so no longer the global one's layers;
- a loader settings file (`vk_loader_settings.json`, newer vkconfig) covering
  vrcompositor without keeping other layers (`unordered_layer_location`): the
  layer does not load.

### Uninstall

```bash
rm ~/.local/lib/libVkLayer_steamvr_compositor_sync.so \
   ~/.local/share/vulkan/explicit_layer.d/VkLayer_steamvr_compositor_sync.json \
   ~/.local/share/vulkan/implicit_layer.d/VkLayer_steamvr_compositor_sync_override.json
```

Deleting just the override manifest turns the layer off.

## How it works

- Every `vkQueueSubmit` / `vkQueueSubmit2` also signals a per-queue timeline
  semaphore of the layer's own, giving each submission a completion point. The
  signal joins the submission's last batch when it can, and goes in a batch of
  its own otherwise.
- Beginning, resetting or freeing a command buffer (or resetting or destroying
  its pool), and resubmitting one still pending without `SIMULTANEOUS_USE`,
  wait for its last submission to complete. Secondary command buffers are
  covered through the primaries that run them.
- Descriptor pools are virtualized: resetting a pool the GPU still uses swaps
  in a fresh one without waiting, and the busy one is reused once idle.
  Destroying a busy pool is deferred until it is idle.
- Every wait is bounded (1 s by default); on timeout the call proceeds, so the
  layer can never deadlock vrcompositor.
- The layer is loaded into vrcompositor only (by the override's `app_keys`)
  and stays inactive in any other process.

The cost is negligible: the hot paths do not allocate or hold a lock across
driver calls. At 120 Hz the layer adds about 3 µs per frame on an RTX 3090 and
16 µs on a ROG Ally, under 0.2% of the frame budget.

## Development

Tests (need `VK_LAYER_KHRONOS_validation` and a Vulkan device):

```bash
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

They reproduce each misuse pattern on the GPU, with the validation layer below
the layer under test told to skip every invalid call, so the driver never runs
the misuse:

| test | checks |
|---|---|
| `layer_fixes_misuse` | with the layer: no validation errors, the right calls wait, pool resets and destroys do not; includes a multi-threaded stress scenario |
| `misuse_detected_without_layer` | without it, validation catches the same misuse |
| `layer_loaded_by_app_keys` | an override manifest alone loads the layer |
| `layer_inert_outside_vrcompositor` | the layer stays inactive in other processes |
| `layer_hot_paths_do_not_allocate` | no allocations by the layer once warmed up |

`build/bench` measures the CPU cost the layer adds to each call it intercepts,
alternating rounds without and with the layer (`--iterations`, `--rounds`,
`--binds`); `--hz N` paces frames like a compositor and reports each frame's
cost and late frames. Configure with `-DCMAKE_BUILD_TYPE=Release` to measure
the optimized layer.

Sanitizers: `-DSTEAMVR_COMPOSITOR_SYNC_SANITIZE=ON` builds with
AddressSanitizer, LeakSanitizer and UBSan; run with
`VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1`, or memory held by unloaded
libraries shows up as leaks. ThreadSanitizer (`-fsanitize=thread` in the
compiler and linker flags) crashes inside NVIDIA's driver; use RADV or lavapipe
with `sync_test --no-validation --expect-clean`.

## Limitations

- Completion points are per queue, so waiting for a command buffer also waits
  for earlier work on its queue.
- Descriptor pools created with custom allocation callbacks, and
  `vkFreeDescriptorSets` or descriptor updates on sets the GPU is still using,
  are passed through unchanged; vrcompositor uses none of these.
- Tested with SteamVR 2.18.1 on NVIDIA (RTX 3090, nvidia-open 615.71.09) and
  AMD (ROG Ally, RADV / Mesa 26.2.3), and on lavapipe.
