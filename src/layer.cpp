// SPDX-License-Identifier: MIT
//
// Vulkan layer entry points: loader negotiation, instance/device chaining and
// the intercepted device functions, which forward to steamvr_compositor_sync::Tracker.
//
// In any process other than vrcompositor (unless STEAMVR_COMPOSITOR_SYNC_FORCE=1) the layer
// hands out the next layer's function pointers, so it costs nothing there.
#include "tracker.hpp"

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#define STEAMVR_COMPOSITOR_SYNC_EXPORT extern "C" __attribute__((visibility("default")))

namespace steamvr_compositor_sync {
namespace {

constexpr const char* kLayerName = STEAMVR_COMPOSITOR_SYNC_LAYER_NAME;

using DispatchKey = void*;

template <typename Handle>
DispatchKey KeyOf(Handle handle) {
    return *reinterpret_cast<DispatchKey*>(handle);
}

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2 = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
};

struct DeviceData {
    DeviceDispatch vk;
    // Null when the layer is inactive for this device.
    std::unique_ptr<Tracker> tracker;
};

std::shared_mutex g_mapMutex;
std::unordered_map<DispatchKey, std::unique_ptr<InstanceData>> g_instances;
std::unordered_map<DispatchKey, std::unique_ptr<DeviceData>> g_devices;

InstanceData* InstanceOf(DispatchKey key) {
    std::shared_lock lock(g_mapMutex);
    const auto it = g_instances.find(key);
    return it == g_instances.end() ? nullptr : it->second.get();
}

// Lock-free lookup for the first few devices, which every intercepted device
// call goes through. Written under g_mapMutex; a reader never races with its
// own device's creation or destruction, which the application orders.
struct FastDevice {
    std::atomic<DispatchKey> key{nullptr};
    std::atomic<DeviceData*> data{nullptr};
};
std::array<FastDevice, 4> g_fastDevices;

void AddFastDeviceLocked(DispatchKey key, DeviceData* data) {
    for (FastDevice& slot : g_fastDevices) {
        if (slot.key.load(std::memory_order_relaxed) == nullptr) {
            slot.data.store(data, std::memory_order_relaxed);
            slot.key.store(key, std::memory_order_release);
            return;
        }
    }
}

void RemoveFastDeviceLocked(DispatchKey key) {
    for (FastDevice& slot : g_fastDevices) {
        if (slot.key.load(std::memory_order_relaxed) == key) {
            slot.key.store(nullptr, std::memory_order_relaxed);
            slot.data.store(nullptr, std::memory_order_relaxed);
        }
    }
}

DeviceData* DeviceOf(DispatchKey key) {
    for (const FastDevice& slot : g_fastDevices) {
        if (slot.key.load(std::memory_order_acquire) == key) {
            return slot.data.load(std::memory_order_relaxed);
        }
    }

    std::shared_lock lock(g_mapMutex);
    const auto it = g_devices.find(key);
    return it == g_devices.end() ? nullptr : it->second.get();
}

template <typename T>
T* FindLayerLink(const void* pNext, VkStructureType type) {
    for (auto* next = static_cast<const VkBaseInStructure*>(pNext); next != nullptr; next = next->pNext) {
        if (next->sType == type && reinterpret_cast<const T*>(next)->function == VK_LAYER_LINK_INFO) {
            return const_cast<T*>(reinterpret_cast<const T*>(next));
        }
    }
    return nullptr;
}

// --- timeline semaphore feature --------------------------------------------

// The tracker needs timeline semaphores. vrcompositor enables them itself;
// otherwise the feature is enabled when the device supports it.
class TimelineFeature {
  public:
    TimelineFeature(const InstanceData& instance, VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* original)
        : m_createInfo(*original) {
        VkBool32* flag = nullptr;
        for (auto* next = static_cast<const VkBaseInStructure*>(original->pNext); next != nullptr; next = next->pNext) {
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
                flag =
                    &const_cast<VkPhysicalDeviceVulkan12Features*>(reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(next))
                         ->timelineSemaphore;
            } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
                flag = &const_cast<VkPhysicalDeviceTimelineSemaphoreFeatures*>(
                            reinterpret_cast<const VkPhysicalDeviceTimelineSemaphoreFeatures*>(next))
                            ->timelineSemaphore;
            }
        }

        if (flag != nullptr && *flag) {
            m_enabled = true;
            return;
        }

        if (!Supported(instance, physicalDevice)) {
            return;
        }

        if (flag != nullptr) {
            // Flip the application's own struct for the duration of vkCreateDevice.
            m_patched = flag;
            *flag = VK_TRUE;
            m_enabled = true;
            return;
        }

        m_feature = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .pNext = const_cast<void*>(original->pNext),
            .timelineSemaphore = VK_TRUE,
        };
        m_createInfo.pNext = &m_feature;

        if (EffectiveApiVersion(instance, physicalDevice) < VK_API_VERSION_1_2) {
            m_extensions.assign(original->ppEnabledExtensionNames,
                                original->ppEnabledExtensionNames + original->enabledExtensionCount);
            const bool present = std::any_of(m_extensions.begin(), m_extensions.end(), [](const char* e) {
                return std::strcmp(e, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0;
            });
            if (!present) {
                m_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            }
            m_createInfo.enabledExtensionCount = static_cast<uint32_t>(m_extensions.size());
            m_createInfo.ppEnabledExtensionNames = m_extensions.data();
        }

        m_enabled = true;
    }

    ~TimelineFeature() {
        if (m_patched != nullptr) {
            *m_patched = VK_FALSE;
        }
    }

    TimelineFeature(const TimelineFeature&) = delete;
    TimelineFeature& operator=(const TimelineFeature&) = delete;

    bool Enabled() const { return m_enabled; }
    const VkDeviceCreateInfo* CreateInfo() const { return &m_createInfo; }

  private:
    static uint32_t EffectiveApiVersion(const InstanceData& instance, VkPhysicalDevice physicalDevice) {
        VkPhysicalDeviceProperties properties = {};
        instance.GetPhysicalDeviceProperties(physicalDevice, &properties);
        return std::min(instance.apiVersion, properties.apiVersion);
    }

    static bool Supported(const InstanceData& instance, VkPhysicalDevice physicalDevice) {
        if (instance.GetPhysicalDeviceFeatures2 == nullptr || EffectiveApiVersion(instance, physicalDevice) < VK_API_VERSION_1_1) {
            return false;
        }

        VkPhysicalDeviceTimelineSemaphoreFeatures timeline = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .pNext = nullptr,
            .timelineSemaphore = VK_FALSE,
        };
        VkPhysicalDeviceFeatures2 features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &timeline,
            .features = {},
        };

        instance.GetPhysicalDeviceFeatures2(physicalDevice, &features);
        return timeline.timelineSemaphore == VK_TRUE;
    }

    VkDeviceCreateInfo m_createInfo;
    VkPhysicalDeviceTimelineSemaphoreFeatures m_feature = {};
    std::vector<const char*> m_extensions;
    VkBool32* m_patched = nullptr;
    bool m_enabled = false;
};

template <typename Pfn>
Pfn Resolve(PFN_vkGetDeviceProcAddr gdpa, VkDevice device, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (PFN_vkVoidFunction function = gdpa(device, name)) {
            return reinterpret_cast<Pfn>(function);
        }
    }
    return nullptr;
}

DeviceDispatch LoadDeviceDispatch(PFN_vkGetDeviceProcAddr gdpa, VkDevice device) {
    return DeviceDispatch{
        .GetDeviceProcAddr = gdpa,
        .DestroyDevice = Resolve<PFN_vkDestroyDevice>(gdpa, device, {"vkDestroyDevice"}),
        .DeviceWaitIdle = Resolve<PFN_vkDeviceWaitIdle>(gdpa, device, {"vkDeviceWaitIdle"}),
        .QueueSubmit = Resolve<PFN_vkQueueSubmit>(gdpa, device, {"vkQueueSubmit"}),
        .QueueSubmit2 = Resolve<PFN_vkQueueSubmit2>(gdpa, device, {"vkQueueSubmit2"}),
        .QueueSubmit2KHR = Resolve<PFN_vkQueueSubmit2>(gdpa, device, {"vkQueueSubmit2KHR"}),
        .CreateSemaphore = Resolve<PFN_vkCreateSemaphore>(gdpa, device, {"vkCreateSemaphore"}),
        .DestroySemaphore = Resolve<PFN_vkDestroySemaphore>(gdpa, device, {"vkDestroySemaphore"}),
        .WaitSemaphores = Resolve<PFN_vkWaitSemaphores>(gdpa, device, {"vkWaitSemaphores", "vkWaitSemaphoresKHR"}),
        .GetSemaphoreCounterValue =
            Resolve<PFN_vkGetSemaphoreCounterValue>(gdpa, device, {"vkGetSemaphoreCounterValue", "vkGetSemaphoreCounterValueKHR"}),
        .AllocateCommandBuffers = Resolve<PFN_vkAllocateCommandBuffers>(gdpa, device, {"vkAllocateCommandBuffers"}),
        .FreeCommandBuffers = Resolve<PFN_vkFreeCommandBuffers>(gdpa, device, {"vkFreeCommandBuffers"}),
        .BeginCommandBuffer = Resolve<PFN_vkBeginCommandBuffer>(gdpa, device, {"vkBeginCommandBuffer"}),
        .ResetCommandBuffer = Resolve<PFN_vkResetCommandBuffer>(gdpa, device, {"vkResetCommandBuffer"}),
        .ResetCommandPool = Resolve<PFN_vkResetCommandPool>(gdpa, device, {"vkResetCommandPool"}),
        .DestroyCommandPool = Resolve<PFN_vkDestroyCommandPool>(gdpa, device, {"vkDestroyCommandPool"}),
        .CmdExecuteCommands = Resolve<PFN_vkCmdExecuteCommands>(gdpa, device, {"vkCmdExecuteCommands"}),
        .CreateDescriptorPool = Resolve<PFN_vkCreateDescriptorPool>(gdpa, device, {"vkCreateDescriptorPool"}),
        .DestroyDescriptorPool = Resolve<PFN_vkDestroyDescriptorPool>(gdpa, device, {"vkDestroyDescriptorPool"}),
        .ResetDescriptorPool = Resolve<PFN_vkResetDescriptorPool>(gdpa, device, {"vkResetDescriptorPool"}),
        .AllocateDescriptorSets = Resolve<PFN_vkAllocateDescriptorSets>(gdpa, device, {"vkAllocateDescriptorSets"}),
        .FreeDescriptorSets = Resolve<PFN_vkFreeDescriptorSets>(gdpa, device, {"vkFreeDescriptorSets"}),
        .CmdBindDescriptorSets = Resolve<PFN_vkCmdBindDescriptorSets>(gdpa, device, {"vkCmdBindDescriptorSets"}),
        .CmdBindDescriptorSets2 = Resolve<PFN_vkCmdBindDescriptorSets2>(gdpa, device, {"vkCmdBindDescriptorSets2"}),
        .CmdBindDescriptorSets2KHR = Resolve<PFN_vkCmdBindDescriptorSets2>(gdpa, device, {"vkCmdBindDescriptorSets2KHR"}),
    };
}

bool HasRequiredFunctions(const DeviceDispatch& vk) {
    return vk.DeviceWaitIdle && vk.QueueSubmit && vk.CreateSemaphore && vk.DestroySemaphore && vk.WaitSemaphores &&
           vk.GetSemaphoreCounterValue && vk.AllocateCommandBuffers && vk.FreeCommandBuffers && vk.BeginCommandBuffer &&
           vk.ResetCommandBuffer && vk.ResetCommandPool && vk.DestroyCommandPool && vk.CmdExecuteCommands &&
           vk.CreateDescriptorPool && vk.DestroyDescriptorPool && vk.ResetDescriptorPool && vk.AllocateDescriptorSets &&
           vk.FreeDescriptorSets && vk.CmdBindDescriptorSets;
}

// --- intercepted device functions ------------------------------------------

void AppendCommandBuffers(std::vector<VkCommandBuffer>& out, const VkSubmitInfo& submit) {
    out.insert(out.end(), submit.pCommandBuffers, submit.pCommandBuffers + submit.commandBufferCount);
}

void AppendCommandBuffers(std::vector<VkCommandBuffer>& out, const VkSubmitInfo2& submit) {
    for (const VkCommandBufferSubmitInfo& info : std::span(submit.pCommandBufferInfos, submit.commandBufferInfoCount)) {
        out.push_back(info.commandBuffer);
    }
}

// --- joining the completion signal to the application's submission ---------
//
// The signal joins the last batch when that batch runs command buffers: its
// semaphore waits then precede work the signal has to wait for anyway, so the
// value completes exactly when a separate submission's would. Batches with an
// extension structure not handled here get a separate submission instead.

const VkSubmitInfo* JoinSignal(std::span<const VkSubmitInfo> submits, const Tracker::Signal& signal) {
    const VkSubmitInfo& last = submits.back();
    if (last.commandBufferCount == 0) {
        return nullptr;
    }

    const VkTimelineSemaphoreSubmitInfo* timeline = nullptr;
    const VkProtectedSubmitInfo* protectedInfo = nullptr;
    const VkPerformanceQuerySubmitInfoKHR* performanceQuery = nullptr;
    for (auto* next = static_cast<const VkBaseInStructure*>(last.pNext); next != nullptr; next = next->pNext) {
        switch (next->sType) {
            case VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO:
                timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
                break;
            case VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO:
                protectedInfo = reinterpret_cast<const VkProtectedSubmitInfo*>(next);
                break;
            case VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR:
                performanceQuery = reinterpret_cast<const VkPerformanceQuerySubmitInfoKHR*>(next);
                break;
            default:
                return nullptr;
        }
    }

    const uint32_t signalCount = last.signalSemaphoreCount;
    if (timeline != nullptr && timeline->signalSemaphoreValueCount != 0 && timeline->signalSemaphoreValueCount != signalCount) {
        return nullptr;
    }

    thread_local struct {
        std::vector<VkSubmitInfo> batches;
        std::vector<VkSemaphore> semaphores;
        std::vector<uint64_t> values;
        VkTimelineSemaphoreSubmitInfo timeline;
        VkProtectedSubmitInfo protectedInfo;
        VkPerformanceQuerySubmitInfoKHR performanceQuery;
    } joined;

    joined.semaphores.assign(last.pSignalSemaphores, last.pSignalSemaphores + signalCount);
    joined.semaphores.push_back(signal.semaphore);
    if (timeline != nullptr && timeline->signalSemaphoreValueCount != 0) {
        joined.values.assign(timeline->pSignalSemaphoreValues, timeline->pSignalSemaphoreValues + signalCount);
    } else {
        joined.values.assign(signalCount, 0);  // binary semaphores: ignored
    }
    joined.values.push_back(signal.value);

    const void* chain = nullptr;
    if (performanceQuery != nullptr) {
        joined.performanceQuery = *performanceQuery;
        joined.performanceQuery.pNext = chain;
        chain = &joined.performanceQuery;
    }

    if (protectedInfo != nullptr) {
        joined.protectedInfo = *protectedInfo;
        joined.protectedInfo.pNext = chain;
        chain = &joined.protectedInfo;
    }

    joined.timeline = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = chain,
        .waitSemaphoreValueCount = timeline != nullptr ? timeline->waitSemaphoreValueCount : 0,
        .pWaitSemaphoreValues = timeline != nullptr ? timeline->pWaitSemaphoreValues : nullptr,
        .signalSemaphoreValueCount = static_cast<uint32_t>(joined.values.size()),
        .pSignalSemaphoreValues = joined.values.data(),
    };

    joined.batches.assign(submits.begin(), submits.end());
    VkSubmitInfo& batch = joined.batches.back();
    batch.pNext = &joined.timeline;
    batch.signalSemaphoreCount = static_cast<uint32_t>(joined.semaphores.size());
    batch.pSignalSemaphores = joined.semaphores.data();
    return joined.batches.data();
}

const VkSubmitInfo2* JoinSignal(std::span<const VkSubmitInfo2> submits, const Tracker::Signal& signal) {
    const VkSubmitInfo2& last = submits.back();
    if (last.commandBufferInfoCount == 0) {
        return nullptr;
    }

    // None of these refers to the signal list.
    for (auto* next = static_cast<const VkBaseInStructure*>(last.pNext); next != nullptr; next = next->pNext) {
        switch (next->sType) {
            case VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR:
            case VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT:
            case VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV:
                break;
            default:
                return nullptr;
        }
    }

    thread_local struct {
        std::vector<VkSubmitInfo2> batches;
        std::vector<VkSemaphoreSubmitInfo> signals;
    } joined;

    joined.signals.assign(last.pSignalSemaphoreInfos, last.pSignalSemaphoreInfos + last.signalSemaphoreInfoCount);
    joined.signals.push_back(VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = signal.semaphore,
        .value = signal.value,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    });

    joined.batches.assign(submits.begin(), submits.end());
    VkSubmitInfo2& batch = joined.batches.back();
    batch.signalSemaphoreInfoCount = static_cast<uint32_t>(joined.signals.size());
    batch.pSignalSemaphoreInfos = joined.signals.data();
    return joined.batches.data();
}

template <typename SubmitInfo, typename Next>
VkResult TrackedSubmit(VkQueue queue, uint32_t submitCount, const SubmitInfo* submits, VkFence fence, Next DeviceDispatch::* next) {
    DeviceData* d = DeviceOf(KeyOf(queue));
    thread_local std::vector<VkCommandBuffer> commandBuffers;
    commandBuffers.clear();
    for (const SubmitInfo& submit : std::span(submits, submitCount)) {
        AppendCommandBuffers(commandBuffers, submit);
    }

    const auto submitNext = [&](const SubmitInfo* batches) { return (d->vk.*next)(queue, submitCount, batches, fence); };
    if (commandBuffers.empty()) {
        return submitNext(submits);
    }

    d->tracker->BeforeSubmit(commandBuffers);
    const std::optional<Tracker::Signal> signal = d->tracker->ReserveSignal(queue);
    if (!signal) {
        return submitNext(submits);
    }

    VkResult result = VK_SUCCESS;
    bool signalled = false;
    if (const SubmitInfo* joined = JoinSignal(std::span(submits, submitCount), *signal)) {
        result = submitNext(joined);
        signalled = result == VK_SUCCESS;
    } else {
        result = submitNext(submits);
        signalled = result == VK_SUCCESS && d->tracker->SubmitSignal(queue, *signal) == VK_SUCCESS;
    }

    d->tracker->FinishSubmit(*signal, signalled, commandBuffers);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* submits, VkFence fence) {
    return TrackedSubmit(queue, submitCount, submits, fence, &DeviceDispatch::QueueSubmit);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* submits, VkFence fence) {
    return TrackedSubmit(queue, submitCount, submits, fence, &DeviceDispatch::QueueSubmit2);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2KHR(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* submits, VkFence fence) {
    return TrackedSubmit(queue, submitCount, submits, fence, &DeviceDispatch::QueueSubmit2KHR);
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* allocateInfo,
                                                      VkCommandBuffer* commandBuffers) {
    DeviceData* d = DeviceOf(KeyOf(device));
    const VkResult result = d->vk.AllocateCommandBuffers(device, allocateInfo, commandBuffers);
    if (result == VK_SUCCESS) {
        d->tracker->OnAllocateCommandBuffers(allocateInfo->commandPool,
                                             std::span(commandBuffers, allocateInfo->commandBufferCount));
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
                                              const VkCommandBuffer* commandBuffers) {
    DeviceData* d = DeviceOf(KeyOf(device));
    d->tracker->BeforeFreeCommandBuffers(pool, std::span(commandBuffers, count));
    d->vk.FreeCommandBuffers(device, pool, count, commandBuffers);
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* beginInfo) {
    DeviceData* d = DeviceOf(KeyOf(commandBuffer));
    d->tracker->BeforeBeginCommandBuffer(commandBuffer, beginInfo ? beginInfo->flags : 0);
    return d->vk.BeginCommandBuffer(commandBuffer, beginInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    DeviceData* d = DeviceOf(KeyOf(commandBuffer));
    d->tracker->BeforeResetCommandBuffer(commandBuffer);
    return d->vk.ResetCommandBuffer(commandBuffer, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) {
    DeviceData* d = DeviceOf(KeyOf(device));
    d->tracker->BeforeResetCommandPool(pool);
    return d->vk.ResetCommandPool(device, pool, flags);
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* allocator) {
    DeviceData* d = DeviceOf(KeyOf(device));
    if (pool != VK_NULL_HANDLE) {
        d->tracker->BeforeDestroyCommandPool(pool);
    }

    d->vk.DestroyCommandPool(device, pool, allocator);
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t count, const VkCommandBuffer* secondaries) {
    DeviceData* d = DeviceOf(KeyOf(commandBuffer));
    d->tracker->OnExecuteCommands(commandBuffer, std::span(secondaries, count));
    d->vk.CmdExecuteCommands(commandBuffer, count, secondaries);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint bindPoint,
                                                 VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount,
                                                 const VkDescriptorSet* sets, uint32_t dynamicOffsetCount,
                                                 const uint32_t* dynamicOffsets) {
    DeviceData* d = DeviceOf(KeyOf(commandBuffer));
    d->tracker->OnBindDescriptorSets(commandBuffer, std::span(sets, setCount));
    d->vk.CmdBindDescriptorSets(commandBuffer, bindPoint, layout, firstSet, setCount, sets, dynamicOffsetCount, dynamicOffsets);
}

void TrackedBindDescriptorSets2(VkCommandBuffer commandBuffer, const VkBindDescriptorSetsInfo* info,
                                PFN_vkCmdBindDescriptorSets2 DeviceDispatch::* next) {
    DeviceData* d = DeviceOf(KeyOf(commandBuffer));
    d->tracker->OnBindDescriptorSets(commandBuffer, std::span(info->pDescriptorSets, info->descriptorSetCount));
    (d->vk.*next)(commandBuffer, info);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets2(VkCommandBuffer commandBuffer, const VkBindDescriptorSetsInfo* info) {
    TrackedBindDescriptorSets2(commandBuffer, info, &DeviceDispatch::CmdBindDescriptorSets2);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets2KHR(VkCommandBuffer commandBuffer, const VkBindDescriptorSetsInfo* info) {
    TrackedBindDescriptorSets2(commandBuffer, info, &DeviceDispatch::CmdBindDescriptorSets2KHR);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* createInfo,
                                                    const VkAllocationCallbacks* allocator, VkDescriptorPool* pool) {
    return DeviceOf(KeyOf(device))->tracker->CreateDescriptorPool(createInfo, allocator, pool);
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* allocator) {
    DeviceData* d = DeviceOf(KeyOf(device));
    if (pool == VK_NULL_HANDLE) {
        d->vk.DestroyDescriptorPool(device, pool, allocator);
        return;
    }

    d->tracker->DestroyDescriptorPool(pool, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags) {
    return DeviceOf(KeyOf(device))->tracker->ResetDescriptorPool(pool, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* allocateInfo,
                                                      VkDescriptorSet* sets) {
    return DeviceOf(KeyOf(device))->tracker->AllocateDescriptorSets(allocateInfo, sets);
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(VkDevice device, VkDescriptorPool pool, uint32_t count,
                                                  const VkDescriptorSet* sets) {
    return DeviceOf(KeyOf(device))->tracker->FreeDescriptorSets(pool, std::span(sets, count));
}

// --- device and instance lifetime ------------------------------------------

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name);

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks* allocator) {
    if (device == VK_NULL_HANDLE) {
        return;
    }

    std::unique_ptr<DeviceData> data;
    {
        std::unique_lock lock(g_mapMutex);
        const auto it = g_devices.find(KeyOf(device));
        if (it == g_devices.end()) {
            return;
        }

        data = std::move(it->second);
        g_devices.erase(it);
        RemoveFastDeviceLocked(KeyOf(device));
    }

    if (data->tracker) {
        data->tracker->Shutdown();
    }

    data->vk.DestroyDevice(device, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* createInfo,
                                            const VkAllocationCallbacks* allocator, VkDevice* device) {
    auto* link = FindLayerLink<VkLayerDeviceCreateInfo>(createInfo->pNext, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO);
    InstanceData* instance = InstanceOf(KeyOf(physicalDevice));
    if (link == nullptr || link->u.pLayerInfo == nullptr || instance == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto nextCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance->instance, "vkCreateDevice"));
    if (nextCreateDevice == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const Config& config = GetConfig();
    std::optional<TimelineFeature> timeline;
    if (config.active) {
        timeline.emplace(*instance, physicalDevice, createInfo);
    }

    const VkResult result = nextCreateDevice(physicalDevice, timeline ? timeline->CreateInfo() : createInfo, allocator, device);
    if (result != VK_SUCCESS) {
        return result;
    }

    auto data = std::make_unique<DeviceData>();
    data->vk = LoadDeviceDispatch(gdpa, *device);
    if (config.active) {
        if (!timeline->Enabled()) {
            Log("timeline semaphores unavailable on this device; layer inactive");
        } else if (!HasRequiredFunctions(data->vk)) {
            Log("required device functions missing; layer inactive");
        } else {
            data->tracker = std::make_unique<Tracker>(*device, data->vk);
            Log("v%s active in %s: command buffers and descriptor pools are recycled only after "
                "the GPU finished with them (wait timeout %llu ms)",
                STEAMVR_COMPOSITOR_SYNC_VERSION, config.process.c_str(),
                static_cast<unsigned long long>(config.waitTimeoutNs / 1'000'000));
        }
    }

    std::unique_lock lock(g_mapMutex);
    AddFastDeviceLocked(KeyOf(*device), data.get());
    g_devices[KeyOf(*device)] = std::move(data);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char* layerName,
                                                                  uint32_t* count, VkExtensionProperties* properties) {
    if (layerName != nullptr && std::strcmp(layerName, kLayerName) == 0) {
        *count = 0;
        return VK_SUCCESS;
    }

    InstanceData* instance = InstanceOf(KeyOf(physicalDevice));
    if (instance == nullptr || instance->EnumerateDeviceExtensionProperties == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return instance->EnumerateDeviceExtensionProperties(physicalDevice, layerName, count, properties);
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks* allocator) {
    if (instance == VK_NULL_HANDLE) {
        return;
    }

    std::unique_ptr<InstanceData> data;
    {
        std::unique_lock lock(g_mapMutex);
        const auto it = g_instances.find(KeyOf(instance));
        if (it == g_instances.end()) {
            return;
        }

        data = std::move(it->second);
        g_instances.erase(it);
    }

    data->DestroyInstance(instance, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* createInfo, const VkAllocationCallbacks* allocator,
                                              VkInstance* instance) {
    auto* link = FindLayerLink<VkLayerInstanceCreateInfo>(createInfo->pNext, VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO);
    if (link == nullptr || link->u.pLayerInfo == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto nextCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (nextCreateInstance == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const VkResult result = nextCreateInstance(createInfo, allocator, instance);
    if (result != VK_SUCCESS) {
        return result;
    }

    const uint32_t apiVersion = (createInfo->pApplicationInfo != nullptr && createInfo->pApplicationInfo->apiVersion != 0)
                                    ? createInfo->pApplicationInfo->apiVersion
                                    : VK_API_VERSION_1_0;
    auto resolve = [&](const char* name) { return gipa(*instance, name); };
    auto data = std::make_unique<InstanceData>(InstanceData{
        .instance = *instance,
        .apiVersion = apiVersion,
        .GetInstanceProcAddr = gipa,
        .DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(resolve("vkDestroyInstance")),
        .EnumerateDeviceExtensionProperties =
            reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(resolve("vkEnumerateDeviceExtensionProperties")),
        .GetPhysicalDeviceFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(resolve("vkGetPhysicalDeviceFeatures2")),
        .GetPhysicalDeviceProperties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(resolve("vkGetPhysicalDeviceProperties")),
    });

    std::unique_lock lock(g_mapMutex);
    g_instances[KeyOf(*instance)] = std::move(data);
    return VK_SUCCESS;
}

// --- proc addr -------------------------------------------------------------

template <typename Hook, typename Next>
PFN_vkVoidFunction IfPresent(Hook hook, Next next) {
    return next != nullptr ? reinterpret_cast<PFN_vkVoidFunction>(hook) : nullptr;
}

PFN_vkVoidFunction TrackedDeviceFunction(const DeviceData& d, std::string_view name) {
    const DeviceDispatch& vk = d.vk;

    if (name == "vkQueueSubmit") return IfPresent(&QueueSubmit, vk.QueueSubmit);
    if (name == "vkQueueSubmit2") return IfPresent(&QueueSubmit2, vk.QueueSubmit2);
    if (name == "vkQueueSubmit2KHR") return IfPresent(&QueueSubmit2KHR, vk.QueueSubmit2KHR);
    if (name == "vkAllocateCommandBuffers") return IfPresent(&AllocateCommandBuffers, vk.AllocateCommandBuffers);
    if (name == "vkFreeCommandBuffers") return IfPresent(&FreeCommandBuffers, vk.FreeCommandBuffers);
    if (name == "vkBeginCommandBuffer") return IfPresent(&BeginCommandBuffer, vk.BeginCommandBuffer);
    if (name == "vkResetCommandBuffer") return IfPresent(&ResetCommandBuffer, vk.ResetCommandBuffer);
    if (name == "vkResetCommandPool") return IfPresent(&ResetCommandPool, vk.ResetCommandPool);
    if (name == "vkDestroyCommandPool") return IfPresent(&DestroyCommandPool, vk.DestroyCommandPool);
    if (name == "vkCmdExecuteCommands") return IfPresent(&CmdExecuteCommands, vk.CmdExecuteCommands);
    if (name == "vkCmdBindDescriptorSets") return IfPresent(&CmdBindDescriptorSets, vk.CmdBindDescriptorSets);
    if (name == "vkCmdBindDescriptorSets2") return IfPresent(&CmdBindDescriptorSets2, vk.CmdBindDescriptorSets2);
    if (name == "vkCmdBindDescriptorSets2KHR") {
        return IfPresent(&CmdBindDescriptorSets2KHR, vk.CmdBindDescriptorSets2KHR);
    }

    if (name == "vkCreateDescriptorPool") return IfPresent(&CreateDescriptorPool, vk.CreateDescriptorPool);
    if (name == "vkDestroyDescriptorPool") return IfPresent(&DestroyDescriptorPool, vk.DestroyDescriptorPool);
    if (name == "vkResetDescriptorPool") return IfPresent(&ResetDescriptorPool, vk.ResetDescriptorPool);
    if (name == "vkAllocateDescriptorSets") return IfPresent(&AllocateDescriptorSets, vk.AllocateDescriptorSets);
    if (name == "vkFreeDescriptorSets") return IfPresent(&FreeDescriptorSets, vk.FreeDescriptorSets);

    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name) {
    if (name == nullptr || device == VK_NULL_HANDLE) {
        return nullptr;
    }

    const std::string_view function(name);
    if (function == "vkGetDeviceProcAddr") {
        return reinterpret_cast<PFN_vkVoidFunction>(&GetDeviceProcAddr);
    }

    DeviceData* d = DeviceOf(KeyOf(device));
    if (d == nullptr) {
        return nullptr;
    }

    if (function == "vkDestroyDevice") {
        return reinterpret_cast<PFN_vkVoidFunction>(&DestroyDevice);
    }

    if (d->tracker) {
        if (PFN_vkVoidFunction hook = TrackedDeviceFunction(*d, function)) {
            return hook;
        }
    }

    return d->vk.GetDeviceProcAddr(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char* name) {
    if (name == nullptr) {
        return nullptr;
    }

    const std::string_view function(name);

    if (function == "vkGetInstanceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(&GetInstanceProcAddr);
    if (function == "vkCreateInstance") return reinterpret_cast<PFN_vkVoidFunction>(&CreateInstance);
    if (function == "vkDestroyInstance") return reinterpret_cast<PFN_vkVoidFunction>(&DestroyInstance);
    if (function == "vkCreateDevice") return reinterpret_cast<PFN_vkVoidFunction>(&CreateDevice);
    if (function == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(&GetDeviceProcAddr);
    if (function == "vkEnumerateDeviceExtensionProperties") {
        return reinterpret_cast<PFN_vkVoidFunction>(&EnumerateDeviceExtensionProperties);
    }

    if (instance == VK_NULL_HANDLE) {
        return nullptr;
    }

    InstanceData* data = InstanceOf(KeyOf(instance));
    return data == nullptr ? nullptr : data->GetInstanceProcAddr(instance, name);
}

}  // namespace
}  // namespace steamvr_compositor_sync

// --- exported loader interface ---------------------------------------------

STEAMVR_COMPOSITOR_SYNC_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* version) {
    if (version == nullptr || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (version->loaderLayerInterfaceVersion > 2) {
        version->loaderLayerInterfaceVersion = 2;
    }

    if (version->loaderLayerInterfaceVersion >= 2) {
        version->pfnGetInstanceProcAddr = &steamvr_compositor_sync::GetInstanceProcAddr;
        version->pfnGetDeviceProcAddr = &steamvr_compositor_sync::GetDeviceProcAddr;
        version->pfnGetPhysicalDeviceProcAddr = nullptr;
    }

    return VK_SUCCESS;
}

STEAMVR_COMPOSITOR_SYNC_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance,
                                                                                              const char* name) {
    return steamvr_compositor_sync::GetInstanceProcAddr(instance, name);
}

STEAMVR_COMPOSITOR_SYNC_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* name) {
    return steamvr_compositor_sync::GetDeviceProcAddr(device, name);
}
