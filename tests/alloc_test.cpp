// SPDX-License-Identifier: MIT
//
// Checks that the layer's steady-state paths do not allocate. The test
// replaces malloc and friends, and counts the calls whose caller lies in the
// layer library (which carries its own libstdc++, so its operator new counts
// too). Two phases are measured after a warm-up:
//
//   frames    a vrcompositor-like loop: reset a descriptor pool the GPU may
//             still use, allocate sets (binding one, freeing the other),
//             re-record a command buffer that may still be pending, submit
//   deferred  submissions while a destroyed descriptor pool is still in use,
//             so every submission re-checks the deferred destruction
//
//   alloc_test    with STEAMVR_COMPOSITOR_SYNC_FORCE=1 and the layer on VK_ADD_LAYER_PATH
#include <vulkan/vulkan.h>

#include <link.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

extern "C" {
void* __libc_malloc(size_t size);
void* __libc_calloc(size_t count, size_t size);
void* __libc_realloc(void* pointer, size_t size);
void* __libc_memalign(size_t alignment, size_t size);
}

namespace {

constexpr const int kSkip = 77;
constexpr const int kWarmupFrames = 200;
constexpr const int kFrames = 2000;
constexpr const int kCommandBuffers = 3;
constexpr const int kCommandPoolResetEvery = 100;
constexpr const int kDeferredWarmup = 2;
constexpr const int kDeferredSubmits = 200;

std::atomic<uintptr_t> g_layerBegin{0};
std::atomic<uintptr_t> g_layerEnd{0};
std::atomic<bool> g_counting{false};
std::atomic<uint64_t> g_layerAllocations{0};

void Count(void* caller) {
    const auto address = reinterpret_cast<uintptr_t>(caller);
    if (g_counting.load(std::memory_order_relaxed) && address >= g_layerBegin.load(std::memory_order_relaxed) &&
        address < g_layerEnd.load(std::memory_order_relaxed)) {
        g_layerAllocations.fetch_add(1, std::memory_order_relaxed);
    }
}

// Finds the address range of the layer's loaded segments.
bool FindLayer() {
    auto visit = [](dl_phdr_info* info, size_t, void*) -> int {
        if (info->dlpi_name == nullptr || std::strstr(info->dlpi_name, "libVkLayer_steamvr_compositor_sync") == nullptr) {
            return 0;
        }

        uintptr_t begin = UINTPTR_MAX;
        uintptr_t end = 0;
        for (const ElfW(Phdr) & header : std::span(info->dlpi_phdr, info->dlpi_phnum)) {
            if (header.p_type == PT_LOAD) {
                begin = std::min<uintptr_t>(begin, info->dlpi_addr + header.p_vaddr);
                end = std::max<uintptr_t>(end, info->dlpi_addr + header.p_vaddr + header.p_memsz);
            }
        }

        g_layerBegin = begin;
        g_layerEnd = end;
        return 1;
    };

    return dl_iterate_phdr(visit, nullptr) != 0;
}

#define CHECK(expr)                                                                                               \
    do {                                                                                                          \
        const VkResult result_ = (expr);                                                                          \
        if (result_ != VK_SUCCESS) {                                                                              \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #expr, static_cast<int>(result_)); \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

}  // namespace

extern "C" {
void* malloc(size_t size) {
    Count(__builtin_return_address(0));
    return __libc_malloc(size);
}
void* calloc(size_t count, size_t size) {
    Count(__builtin_return_address(0));
    return __libc_calloc(count, size);
}
void* realloc(void* pointer, size_t size) {
    Count(__builtin_return_address(0));
    return __libc_realloc(pointer, size);
}
void* aligned_alloc(size_t alignment, size_t size) {
    Count(__builtin_return_address(0));
    return __libc_memalign(alignment, size);
}
int posix_memalign(void** pointer, size_t alignment, size_t size) {
    Count(__builtin_return_address(0));
    *pointer = __libc_memalign(alignment, size);
    return *pointer != nullptr ? 0 : ENOMEM;
}
}

int main() {
    const VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "alloc_test",
        .applicationVersion = 1,
        .pEngineName = nullptr,
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_2,
    };
    const char* layerName = STEAMVR_COMPOSITOR_SYNC_LAYER_NAME;
    const VkInstanceCreateInfo instanceInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 1,
        .ppEnabledLayerNames = &layerName,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "no Vulkan instance with %s (is VK_ADD_LAYER_PATH set?)\n", layerName);
        return kSkip;
    }

    uint32_t deviceCount = 1;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    vkEnumeratePhysicalDevices(instance, &deviceCount, &physicalDevice);
    if (deviceCount == 0) {
        return kSkip;
    }

    VkPhysicalDeviceVulkan12Features features12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = nullptr,
        .timelineSemaphore = VK_TRUE,
    };
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queueInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const VkDeviceCreateInfo deviceInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features12,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
        .pEnabledFeatures = nullptr,
    };
    VkDevice device = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, 0, 0, &queue);

    if (!FindLayer()) {
        std::fprintf(stderr, "layer library not loaded\n");
        return 1;
    }

    const VkCommandPoolCreateInfo commandPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = 0,
    };
    VkCommandPool commandPool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool));

    const VkCommandBufferAllocateInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = kCommandBuffers + kDeferredSubmits + 1,
    };
    VkCommandBuffer allCommandBuffers[kCommandBuffers + kDeferredSubmits + 1] = {};
    CHECK(vkAllocateCommandBuffers(device, &commandBufferInfo, allCommandBuffers));

    const std::span<VkCommandBuffer> commandBuffers(allCommandBuffers, kCommandBuffers);
    const std::span<VkCommandBuffer> deferredCommandBuffers(allCommandBuffers + kCommandBuffers, kDeferredSubmits);
    const VkCommandBuffer gatedCommandBuffer = allCommandBuffers[kCommandBuffers + kDeferredSubmits];

    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout));

    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &setLayout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    const VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4,
    };
    const VkDescriptorPoolCreateInfo descriptorPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 4,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    const VkSemaphoreTypeCreateInfo timelineType = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo gateInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timelineType,
        .flags = 0,
    };
    VkSemaphore gate = VK_NULL_HANDLE;
    CHECK(vkCreateSemaphore(device, &gateInfo, nullptr, &gate));

    auto allocateSet = [&](VkDescriptorPool pool) {
        const VkDescriptorSetAllocateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &setLayout,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        CHECK(vkAllocateDescriptorSets(device, &info, &set));
        return set;
    };

    auto record = [&](VkCommandBuffer commandBuffer, VkDescriptorSet set) {
        CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        if (set != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        }
        CHECK(vkEndCommandBuffer(commandBuffer));
    };

    // Submits one command buffer, optionally held until the host signals `gate`.
    auto submit = [&](const VkCommandBuffer& commandBuffer, bool gated = false) {
        const uint64_t gateValue = 1;
        const VkPipelineStageFlags gateStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkTimelineSemaphoreSubmitInfo gateValues = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &gateValue,
            .signalSemaphoreValueCount = 0,
            .pSignalSemaphoreValues = nullptr,
        };
        const VkSubmitInfo info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = gated ? &gateValues : nullptr,
            .waitSemaphoreCount = gated ? 1u : 0u,
            .pWaitSemaphores = &gate,
            .pWaitDstStageMask = &gateStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        CHECK(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
    };

    // A valid first submission: an active layer allocates its queue timeline
    // here. Without an active layer the misuse below would reach the GPU.
    g_counting = true;
    record(commandBuffers[0], VK_NULL_HANDLE);
    submit(commandBuffers[0]);
    CHECK(vkQueueWaitIdle(queue));
    if (g_layerAllocations.exchange(0) == 0) {
        std::fprintf(stderr, "the layer is not active (is STEAMVR_COMPOSITOR_SYNC_FORCE=1 set?)\n");
        return 1;
    }

    auto frame = [&](int index) {
        if (index % kCommandPoolResetEvery == kCommandPoolResetEvery - 1) {
            CHECK(vkResetCommandPool(device, commandPool, 0));
        }

        CHECK(vkResetDescriptorPool(device, descriptorPool, 0));
        const VkDescriptorSet set = allocateSet(descriptorPool);
        const VkDescriptorSet unused = allocateSet(descriptorPool);
        CHECK(vkFreeDescriptorSets(device, descriptorPool, 1, &unused));

        const VkCommandBuffer commandBuffer = commandBuffers[index % kCommandBuffers];
        record(commandBuffer, set);
        submit(commandBuffer);
    };

    for (int i = 0; i < kWarmupFrames; ++i) {
        frame(i);
    }

    const uint64_t warmupAllocations = g_layerAllocations.exchange(0);
    for (int i = kWarmupFrames; i < kWarmupFrames + kFrames; ++i) {
        frame(i);
    }

    const uint64_t frameAllocations = g_layerAllocations.exchange(0);

    // Keep a destroyed pool in use behind the gate while more work is submitted.
    const VkDescriptorPool busyPool = [&] {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &pool));
        return pool;
    }();
    record(gatedCommandBuffer, allocateSet(busyPool));
    submit(gatedCommandBuffer, true);
    vkDestroyDescriptorPool(device, busyPool, nullptr);

    for (VkCommandBuffer commandBuffer : deferredCommandBuffers) {
        record(commandBuffer, VK_NULL_HANDLE);
    }

    for (int i = 0; i < kDeferredWarmup; ++i) {
        submit(deferredCommandBuffers[i]);
    }

    g_layerAllocations = 0;
    for (int i = kDeferredWarmup; i < kDeferredSubmits; ++i) {
        submit(deferredCommandBuffers[i]);
    }

    g_counting = false;
    const uint64_t deferredAllocations = g_layerAllocations.load();

    const VkSemaphoreSignalInfo open = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext = nullptr,
        .semaphore = gate,
        .value = 1,
    };
    CHECK(vkSignalSemaphore(device, &open));

    vkDeviceWaitIdle(device);
    vkDestroySemaphore(device, gate, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::printf(
        "layer allocations: %llu during %d warm-up frames, %llu over %d frames, "
        "%llu over %d submissions with a deferred pool\n",
        static_cast<unsigned long long>(warmupAllocations), kWarmupFrames, static_cast<unsigned long long>(frameAllocations),
        kFrames, static_cast<unsigned long long>(deferredAllocations), kDeferredSubmits - kDeferredWarmup);
    const bool pass = frameAllocations == 0 && deferredAllocations == 0;
    std::printf("%s: expected none after warm-up\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
