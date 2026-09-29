// SPDX-License-Identifier: MIT
//
// Isolation test for the vrcompositor sync layer.
//
// Reproduces, on the real GPU, each misuse pattern vrcompositor exhibits
// (ValveSoftware/SteamVR-for-Linux#952): resetting, re-beginning, freeing and
// resubmitting command buffers whose submission is still pending, resetting
// and destroying descriptor pools whose sets are still in use. Work is kept
// pending deterministically by making each submission wait on a host timeline
// semaphore that a helper thread signals 200 ms later.
//
// VK_LAYER_KHRONOS_validation sits below the layer under test and reports
// every misuse that reaches the driver. The debug callback returns VK_TRUE,
// which makes validation skip the offending call, so the driver never
// executes a misuse even when the test runs without the fix.
//
//   sync_test --expect-clean             layer active: no validation errors,
//                                        and the waits happen where expected
//   sync_test --no-layer --expect-errors validation must catch the misuse
//   sync_test --expect-errors            layer loaded but inactive (the test
//                                        process is not vrcompositor)
//   sync_test --no-validation --expect-clean
//                                        smoke test where validation is not
//                                        available (e.g. inside the Steam
//                                        runtime): only checks the layer's
//                                        waits and swaps
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cstdint>

#include "fill_comp_spv.h"

namespace {

constexpr const int kSkip = 77;
constexpr const auto kSignalDelay = std::chrono::milliseconds(200);
// A call the layer holds until the GPU is done must block at least this long.
constexpr const auto kMinBlocked = std::chrono::milliseconds(150);
// A call the layer satisfies without waiting (pool swap) must be this quick.
constexpr const auto kMaxUnblocked = std::chrono::milliseconds(50);

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

std::mutex g_errorMutex;
std::map<std::string, int> g_errors;
std::atomic<int> g_errorCount{0};

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) == 0) {
        return VK_FALSE;
    }

    const std::string id = data->pMessageIdName ? data->pMessageIdName : "(unnamed)";
    {
        std::lock_guard lock(g_errorMutex);
        g_errors[id]++;
    }

    g_errorCount++;
    std::fprintf(stderr, "  validation error: %s\n", id.c_str());
    // Skip the call: the driver must never see the misuse.
    return VK_TRUE;
}

using Clock = std::chrono::steady_clock;

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

bool LayerAvailable(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());

    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, name) == 0) {
            return true;
        }
    }
    return false;
}

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkQueue otherQueue = VK_NULL_HANDLE;  // second queue of the family, or `queue`
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

#define CHECK(expr)                                                                                               \
    do {                                                                                                          \
        const VkResult result_ = (expr);                                                                          \
        if (result_ != VK_SUCCESS) {                                                                              \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #expr, static_cast<int>(result_)); \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

int CreateContext(Context& ctx, bool withLayer, bool withValidation) {
    if (withValidation && !LayerAvailable(kValidationLayer)) {
        std::fprintf(stderr, "%s not installed; skipping\n", kValidationLayer);
        return kSkip;
    }

    if (withLayer && !LayerAvailable(STEAMVR_COMPOSITOR_SYNC_LAYER_NAME)) {
        std::fprintf(stderr, "%s not found (is VK_ADD_LAYER_PATH set?)\n", STEAMVR_COMPOSITOR_SYNC_LAYER_NAME);
        return 1;
    }

    // First listed = closest to the application: the layer under test calls
    // down into validation.
    std::vector<const char*> layers;
    if (withLayer) {
        layers.push_back(STEAMVR_COMPOSITOR_SYNC_LAYER_NAME);
    }

    if (withValidation) {
        layers.push_back(kValidationLayer);
    }

    const char* extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

    const VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "steamvr-compositor-sync test",
        .applicationVersion = 1,
        .pEngineName = nullptr,
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkDebugUtilsMessengerCreateInfoEXT messengerInfo = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
        .pfnUserCallback = DebugCallback,
        .pUserData = nullptr,
    };
    const VkInstanceCreateInfo instanceInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &messengerInfo,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = extensions,
    };
    CHECK(vkCreateInstance(&instanceInfo, nullptr, &ctx.instance));
    auto createMessenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(ctx.instance, "vkCreateDebugUtilsMessengerEXT"));
    CHECK(createMessenger(ctx.instance, &messengerInfo, nullptr, &ctx.messenger));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
    if (devices.empty()) {
        std::fprintf(stderr, "no Vulkan device; skipping\n");
        return kSkip;
    }

    ctx.physicalDevice = devices[0];
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx.physicalDevice = candidate;
            break;
        }
    }

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &properties);
    std::printf("device: %s\n", properties.deviceName);

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &familyCount, families.data());

    bool found = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queueFamily = i;
            found = true;
            break;
        }
    }

    const uint32_t queueCount = found ? std::min(families[ctx.queueFamily].queueCount, 2u) : 0;
    if (!found) {
        std::fprintf(stderr, "no compute queue; skipping\n");
        return kSkip;
    }

    const float priorities[] = {1.0f, 1.0f};
    const VkDeviceQueueCreateInfo queueInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = ctx.queueFamily,
        .queueCount = queueCount,
        .pQueuePriorities = priorities,
    };
    VkPhysicalDeviceVulkan13Features features13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = nullptr,
        .synchronization2 = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features features12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &features13,
        .timelineSemaphore = VK_TRUE,
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
    CHECK(vkCreateDevice(ctx.physicalDevice, &deviceInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, queueCount - 1, &ctx.otherQueue);

    const VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx.queueFamily,
    };
    CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool));

    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = 4096,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    CHECK(vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &ctx.buffer));

    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(ctx.device, ctx.buffer, &requirements);

    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memoryProperties);

    uint32_t memoryType = 0;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if (requirements.memoryTypeBits & (1u << i)) {
            memoryType = i;
            break;
        }
    }

    const VkMemoryAllocateInfo allocateInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memoryType,
    };
    CHECK(vkAllocateMemory(ctx.device, &allocateInfo, nullptr, &ctx.memory));
    CHECK(vkBindBufferMemory(ctx.device, ctx.buffer, ctx.memory, 0));

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
    CHECK(vkCreateDescriptorSetLayout(ctx.device, &setLayoutInfo, nullptr, &ctx.setLayout));

    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx.setLayout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    CHECK(vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &ctx.pipelineLayout));

    const VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = sizeof(kFillShaderSpirv),
        .pCode = kFillShaderSpirv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(ctx.device, &shaderInfo, nullptr, &shader));

    const VkComputePipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
        .layout = ctx.pipelineLayout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &ctx.pipeline));
    vkDestroyShaderModule(ctx.device, shader, nullptr);
    return 0;
}

void DestroyContext(Context& ctx) {
    if (ctx.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx.device);
        vkDestroyPipeline(ctx.device, ctx.pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, ctx.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, ctx.setLayout, nullptr);
        vkDestroyBuffer(ctx.device, ctx.buffer, nullptr);
        vkFreeMemory(ctx.device, ctx.memory, nullptr);
        vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        vkDestroyDevice(ctx.device, nullptr);
    }

    if (ctx.messenger != VK_NULL_HANDLE) {
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(ctx.instance, "vkDestroyDebugUtilsMessengerEXT"));
        destroyMessenger(ctx.instance, ctx.messenger, nullptr);
    }

    if (ctx.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx.instance, nullptr);
    }
}

// --- helpers ---------------------------------------------------------------

// A timeline semaphore the submissions wait on; signalled from a thread.
class HostGate {
  public:
    explicit HostGate(const Context& ctx) : m_device(ctx.device) {
        const VkSemaphoreTypeCreateInfo typeInfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0,
        };
        const VkSemaphoreCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &typeInfo,
            .flags = 0,
        };
        CHECK(vkCreateSemaphore(m_device, &info, nullptr, &m_semaphore));
    }
    ~HostGate() {
        Open();
        vkDestroySemaphore(m_device, m_semaphore, nullptr);
    }
    HostGate(const HostGate&) = delete;
    HostGate& operator=(const HostGate&) = delete;

    VkSemaphore Semaphore() const { return m_semaphore; }

    void OpenLater() {
        m_thread = std::thread([this] {
            std::this_thread::sleep_for(kSignalDelay);
            Signal();
        });
    }

    // Joins the timer thread (or signals immediately if none), then waits
    // for the queue so nothing is pending when the scenario ends.
    void Open() {
        if (m_thread.joinable()) {
            m_thread.join();
        } else if (!m_signalled) {
            Signal();
        }
    }

  private:
    void Signal() {
        const VkSemaphoreSignalInfo info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            .pNext = nullptr,
            .semaphore = m_semaphore,
            .value = 1,
        };
        vkSignalSemaphore(m_device, &info);
        m_signalled = true;
    }

    VkDevice m_device;
    VkSemaphore m_semaphore = VK_NULL_HANDLE;
    std::thread m_thread;
    std::atomic<bool> m_signalled{false};
};

VkCommandBuffer AllocateCommandBuffer(const Context& ctx, VkCommandPool pool,
                                      VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
    const VkCommandBufferAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = level,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(ctx.device, &info, &commandBuffer));
    return commandBuffer;
}

void Begin(VkCommandBuffer commandBuffer, VkCommandBufferUsageFlags flags = 0) {
    const VkCommandBufferBeginInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = flags,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(commandBuffer, &info);
}

void RecordFill(const Context& ctx, VkCommandBuffer commandBuffer, VkDescriptorSet set = VK_NULL_HANDLE,
                VkCommandBufferUsageFlags flags = 0) {
    Begin(commandBuffer, flags);

    if (set != VK_NULL_HANDLE) {
        // A dispatch that reads the set: only then does validation (like the
        // driver) consider the set in use by the submission.
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
    } else {
        vkCmdFillBuffer(commandBuffer, ctx.buffer, 0, VK_WHOLE_SIZE, 0x5a5a5a5a);
    }

    vkEndCommandBuffer(commandBuffer);
}

// Submits the command buffers; with a gate, they stay pending until it opens.
VkResult Submit(VkQueue queue, std::span<const VkCommandBuffer> commandBuffers, const HostGate* gate = nullptr) {
    const uint64_t waitValue = 1;
    const VkSemaphore waitSemaphore = gate ? gate->Semaphore() : VK_NULL_HANDLE;
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkTimelineSemaphoreSubmitInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = gate ? 1u : 0u,
        .pWaitSemaphoreValues = gate ? &waitValue : nullptr,
        .signalSemaphoreValueCount = 0,
        .pSignalSemaphoreValues = nullptr,
    };
    const VkSubmitInfo info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timelineInfo,
        .waitSemaphoreCount = gate ? 1u : 0u,
        .pWaitSemaphores = gate ? &waitSemaphore : nullptr,
        .pWaitDstStageMask = gate ? &waitStage : nullptr,
        .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
        .pCommandBuffers = commandBuffers.data(),
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    return vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE);
}

VkResult Submit(const Context& ctx, VkCommandBuffer commandBuffer, const HostGate* gate = nullptr) {
    return Submit(ctx.queue, std::span(&commandBuffer, 1), gate);
}

VkResult Submit2(const Context& ctx, VkCommandBuffer commandBuffer, const HostGate& gate) {
    const VkSemaphoreSubmitInfo wait = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = gate.Semaphore(),
        .value = 1,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    };
    const VkCommandBufferSubmitInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = commandBuffer,
        .deviceMask = 0,
    };
    const VkSubmitInfo2 info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferInfo,
        .signalSemaphoreInfoCount = 0,
        .pSignalSemaphoreInfos = nullptr,
    };
    return vkQueueSubmit2(ctx.queue, 1, &info, VK_NULL_HANDLE);
}

VkDescriptorPool CreateDescriptorPool(const Context& ctx) {
    const VkDescriptorPoolSize size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4,
    };
    const VkDescriptorPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 4,
        .poolSizeCount = 1,
        .pPoolSizes = &size,
    };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorPool(ctx.device, &info, nullptr, &pool));
    return pool;
}

VkDescriptorSet AllocateSet(const Context& ctx, VkDescriptorPool pool) {
    const VkDescriptorSetAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx.setLayout,
    };
    VkDescriptorSet set = VK_NULL_HANDLE;
    CHECK(vkAllocateDescriptorSets(ctx.device, &info, &set));

    const VkDescriptorBufferInfo bufferInfo = {
        .buffer = ctx.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &bufferInfo,
        .pTexelBufferView = nullptr,
    };
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return set;
}

// --- scenarios -------------------------------------------------------------

enum class Timing { Blocks, Immediate, Unchecked };

struct Scenario {
    const char* name;
    Timing expected;
    // Sets up pending work, performs the misuse, and returns how long the
    // misusing call took.
    std::function<Clock::duration(Context&)> run;
    // The misuse is too racy for validation to skip reliably, so without the
    // fix it reaches the GPU (and faults it): run only with --expect-clean.
    bool onlyWithFix = false;
};

template <typename F>
Clock::duration Timed(F&& call) {
    const auto start = Clock::now();
    call();
    return Clock::now() - start;
}

// Times `misuse` while `gate` opens after kSignalDelay, then drains the queue.
template <typename F>
Clock::duration TimeWhilePending(const Context& ctx, HostGate& gate, F&& misuse) {
    gate.OpenLater();
    const Clock::duration elapsed = Timed(misuse);
    gate.Open();
    vkDeviceWaitIdle(ctx.device);
    return elapsed;
}

std::vector<Scenario> Scenarios() {
    return {
        {"reset + begin a pending command buffer", Timing::Blocks,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);
             RecordFill(ctx, commandBuffer);
             CHECK(Submit(ctx, commandBuffer, &gate));
             const auto elapsed = TimeWhilePending(ctx, gate, [&] {
                 vkResetCommandBuffer(commandBuffer, 0);
                 RecordFill(ctx, commandBuffer);
             });
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"begin (implicit reset) a pending command buffer", Timing::Blocks,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);
             RecordFill(ctx, commandBuffer);
             CHECK(Submit(ctx, commandBuffer, &gate));
             const auto elapsed = TimeWhilePending(ctx, gate, [&] { RecordFill(ctx, commandBuffer); });
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"resubmit a pending command buffer", Timing::Blocks,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);
             RecordFill(ctx, commandBuffer);
             CHECK(Submit(ctx, commandBuffer, &gate));
             const auto elapsed = TimeWhilePending(ctx, gate, [&] { Submit(ctx, commandBuffer); });
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"vkQueueSubmit2, then reset the pending command buffer", Timing::Blocks,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);
             RecordFill(ctx, commandBuffer);
             CHECK(Submit2(ctx, commandBuffer, gate));
             const auto elapsed = TimeWhilePending(ctx, gate, [&] { vkResetCommandBuffer(commandBuffer, 0); });
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"submit ending in a batch without command buffers, then reset", Timing::Blocks,
         [](Context& ctx) {
             // The signal cannot join the last batch: it is submitted separately.
             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);
             RecordFill(ctx, commandBuffer);
             const uint64_t waitValue = 1;
             const VkSemaphore waitSemaphore = gate.Semaphore();
             const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
             const VkTimelineSemaphoreSubmitInfo timelineInfo = {
                 .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                 .pNext = nullptr,
                 .waitSemaphoreValueCount = 1,
                 .pWaitSemaphoreValues = &waitValue,
                 .signalSemaphoreValueCount = 0,
                 .pSignalSemaphoreValues = nullptr,
             };
             const VkSubmitInfo batches[] = {
                 {
                     .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .pNext = &timelineInfo,
                     .waitSemaphoreCount = 1,
                     .pWaitSemaphores = &waitSemaphore,
                     .pWaitDstStageMask = &waitStage,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &commandBuffer,
                     .signalSemaphoreCount = 0,
                     .pSignalSemaphores = nullptr,
                 },
                 {
                     .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .pNext = nullptr,
                     .waitSemaphoreCount = 0,
                     .pWaitSemaphores = nullptr,
                     .pWaitDstStageMask = nullptr,
                     .commandBufferCount = 0,
                     .pCommandBuffers = nullptr,
                     .signalSemaphoreCount = 0,
                     .pSignalSemaphores = nullptr,
                 },
             };
             CHECK(vkQueueSubmit(ctx.queue, 2, batches, VK_NULL_HANDLE));
             const auto elapsed = TimeWhilePending(ctx, gate, [&] { vkResetCommandBuffer(commandBuffer, 0); });
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"submit signalling the application's own semaphores, then reset", Timing::Blocks,
         [](Context& ctx) {
             // The signal joins a batch that already signals a binary and a
             // timeline semaphore: their values must stay paired.
             HostGate gate(ctx);
             const VkSemaphoreTypeCreateInfo timelineType = {
                 .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                 .pNext = nullptr,
                 .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                 .initialValue = 0,
             };
             const VkSemaphoreCreateInfo binaryInfo = {
                 .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                 .pNext = nullptr,
                 .flags = 0,
             };
             const VkSemaphoreCreateInfo timelineInfo = {
                 .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                 .pNext = &timelineType,
                 .flags = 0,
             };
             VkSemaphore signals[2] = {};
             CHECK(vkCreateSemaphore(ctx.device, &binaryInfo, nullptr, &signals[0]));
             CHECK(vkCreateSemaphore(ctx.device, &timelineInfo, nullptr, &signals[1]));

             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);

             RecordFill(ctx, commandBuffer);

             const uint64_t waitValue = 1;
             const uint64_t signalValues[] = {0, 7};
             const VkSemaphore waitSemaphore = gate.Semaphore();
             const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
             const VkTimelineSemaphoreSubmitInfo values = {
                 .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                 .pNext = nullptr,
                 .waitSemaphoreValueCount = 1,
                 .pWaitSemaphoreValues = &waitValue,
                 .signalSemaphoreValueCount = 2,
                 .pSignalSemaphoreValues = signalValues,
             };
             const VkSubmitInfo submit = {
                 .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                 .pNext = &values,
                 .waitSemaphoreCount = 1,
                 .pWaitSemaphores = &waitSemaphore,
                 .pWaitDstStageMask = &waitStage,
                 .commandBufferCount = 1,
                 .pCommandBuffers = &commandBuffer,
                 .signalSemaphoreCount = 2,
                 .pSignalSemaphores = signals,
             };
             CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));

             const auto elapsed = TimeWhilePending(ctx, gate, [&] { vkResetCommandBuffer(commandBuffer, 0); });

             // The application's own signals arrived with their values.
             uint64_t reached = 0;
             vkGetSemaphoreCounterValue(ctx.device, signals[1], &reached);
             if (reached != 7) {
                 std::fprintf(stderr, "  application timeline semaphore at %llu, expected 7\n",
                              static_cast<unsigned long long>(reached));
                 g_errorCount++;
             }

             // Consume the binary signal before destroying it.
             const VkSubmitInfo consume = {
                 .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                 .pNext = nullptr,
                 .waitSemaphoreCount = 1,
                 .pWaitSemaphores = &signals[0],
                 .pWaitDstStageMask = &waitStage,
                 .commandBufferCount = 0,
                 .pCommandBuffers = nullptr,
                 .signalSemaphoreCount = 0,
                 .pSignalSemaphores = nullptr,
             };
             CHECK(vkQueueSubmit(ctx.queue, 1, &consume, VK_NULL_HANDLE));
             vkQueueWaitIdle(ctx.queue);
             vkDestroySemaphore(ctx.device, signals[0], nullptr);
             vkDestroySemaphore(ctx.device, signals[1], nullptr);
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"resubmit alongside a pending simultaneous-use buffer, then reset it", Timing::Blocks,
         [](Context& ctx) {
             // `simultaneous` is pending on the other queue, which the resubmit
             // on the first queue does not cover: it must stay tracked, so
             // resetting it afterwards still waits for its gate.
             HostGate normalGate(ctx);
             HostGate simultaneousGate(ctx);
             const VkCommandBuffer normal = AllocateCommandBuffer(ctx, ctx.commandPool);
             const VkCommandBuffer simultaneous = AllocateCommandBuffer(ctx, ctx.commandPool);

             RecordFill(ctx, normal);
             RecordFill(ctx, simultaneous, VK_NULL_HANDLE, VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

             CHECK(Submit(ctx, normal, &normalGate));
             CHECK(Submit(ctx.otherQueue, std::span(&simultaneous, 1), &simultaneousGate));

             normalGate.OpenLater();
             const VkCommandBuffer both[] = {normal, simultaneous};
             Submit(ctx.queue, both);

             normalGate.Open();
             const auto elapsed = TimeWhilePending(ctx, simultaneousGate, [&] { vkResetCommandBuffer(simultaneous, 0); });
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 2, both);
             return elapsed;
         }},

        {"reset a command pool with a pending command buffer", Timing::Blocks,
         [](Context& ctx) {
             const VkCommandPoolCreateInfo info = {
                 .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                 .pNext = nullptr,
                 .flags = 0,
                 .queueFamilyIndex = ctx.queueFamily,
             };
             VkCommandPool pool = VK_NULL_HANDLE;
             CHECK(vkCreateCommandPool(ctx.device, &info, nullptr, &pool));

             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, pool);

             RecordFill(ctx, commandBuffer);
             CHECK(Submit(ctx, commandBuffer, &gate));

             const auto elapsed = TimeWhilePending(ctx, gate, [&] { vkResetCommandPool(ctx.device, pool, 0); });
             vkDestroyCommandPool(ctx.device, pool, nullptr);
             return elapsed;
         }},

        {"free a pending command buffer", Timing::Blocks,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);

             RecordFill(ctx, commandBuffer);
             CHECK(Submit(ctx, commandBuffer, &gate));

             const auto elapsed =
                 TimeWhilePending(ctx, gate, [&] { vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer); });
             return elapsed;
         }},

        {"reset a secondary command buffer run by a pending primary", Timing::Blocks,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkCommandBuffer secondary = AllocateCommandBuffer(ctx, ctx.commandPool, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
             const VkCommandBufferInheritanceInfo inheritance = {
                 .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
                 .pNext = nullptr,
                 .renderPass = VK_NULL_HANDLE,
                 .subpass = 0,
                 .framebuffer = VK_NULL_HANDLE,
                 .occlusionQueryEnable = VK_FALSE,
                 .queryFlags = 0,
                 .pipelineStatistics = 0,
             };
             const VkCommandBufferBeginInfo beginInfo = {
                 .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                 .pNext = nullptr,
                 .flags = 0,
                 .pInheritanceInfo = &inheritance,
             };
             vkBeginCommandBuffer(secondary, &beginInfo);
             vkCmdFillBuffer(secondary, ctx.buffer, 0, VK_WHOLE_SIZE, 1);
             vkEndCommandBuffer(secondary);

             const VkCommandBuffer primary = AllocateCommandBuffer(ctx, ctx.commandPool);
             Begin(primary);
             vkCmdExecuteCommands(primary, 1, &secondary);
             vkEndCommandBuffer(primary);

             CHECK(Submit(ctx, primary, &gate));
             const auto elapsed = TimeWhilePending(ctx, gate, [&] { vkResetCommandBuffer(secondary, 0); });
             const VkCommandBuffer both[] = {primary, secondary};
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 2, both);
             return elapsed;
         }},

        {"reset a descriptor pool whose set is in use (swap, no wait)", Timing::Immediate,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkDescriptorPool pool = CreateDescriptorPool(ctx);
             const VkDescriptorSet first = AllocateSet(ctx, pool);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);

             RecordFill(ctx, commandBuffer, first);
             CHECK(Submit(ctx, commandBuffer, &gate));

             const auto elapsed = Timed([&] { vkResetDescriptorPool(ctx.device, pool, 0); });

             // The application keeps using its pool as if it had been reset.
             const VkDescriptorSet second = AllocateSet(ctx, pool);
             const VkCommandBuffer next = AllocateCommandBuffer(ctx, ctx.commandPool);

             RecordFill(ctx, next, second);
             CHECK(Submit(ctx, next));

             gate.Open();

             vkQueueWaitIdle(ctx.queue);
             // Resetting again once idle recycles the retired pool.
             vkResetDescriptorPool(ctx.device, pool, 0);
             vkDestroyDescriptorPool(ctx.device, pool, nullptr);
             const VkCommandBuffer both[] = {commandBuffer, next};
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 2, both);
             return elapsed;
         }},

        {"destroy a descriptor pool whose set is in use (deferred)", Timing::Immediate,
         [](Context& ctx) {
             HostGate gate(ctx);
             const VkDescriptorPool pool = CreateDescriptorPool(ctx);
             const VkDescriptorSet set = AllocateSet(ctx, pool);
             const VkCommandBuffer commandBuffer = AllocateCommandBuffer(ctx, ctx.commandPool);

             RecordFill(ctx, commandBuffer, set);
             CHECK(Submit(ctx, commandBuffer, &gate));

             const auto elapsed = Timed([&] { vkDestroyDescriptorPool(ctx.device, pool, nullptr); });

             gate.Open();
             vkQueueWaitIdle(ctx.queue);
             // The next submission lets the layer destroy the deferred pool.
             RecordFill(ctx, commandBuffer);
             CHECK(Submit(ctx, commandBuffer));
             vkQueueWaitIdle(ctx.queue);
             vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
             return elapsed;
         }},

        {"threads recycling in-flight work on shared queues (stress)", Timing::Unchecked,
         [](Context& ctx) {
             // vrcompositor's pattern from several threads at once: every
             // iteration resets the descriptor pool and re-records a command
             // buffer while the previous submissions may still be running.
             constexpr const int kThreads = 4;
             constexpr const int kIterations = 300;
             constexpr const int kRecreateEvery = 50;
             std::mutex queueMutex;
             auto worker = [&](int index) {
                 const VkCommandPoolCreateInfo info = {
                     .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                     .pNext = nullptr,
                     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                     .queueFamilyIndex = ctx.queueFamily,
                 };
                 VkCommandPool commandPool = VK_NULL_HANDLE;
                 CHECK(vkCreateCommandPool(ctx.device, &info, nullptr, &commandPool));

                 const VkCommandBuffer commandBuffers[] = {AllocateCommandBuffer(ctx, commandPool),
                                                           AllocateCommandBuffer(ctx, commandPool)};

                 const VkQueue queue = index % 2 == 0 ? ctx.queue : ctx.otherQueue;
                 VkDescriptorPool descriptorPool = CreateDescriptorPool(ctx);
                 for (int i = 0; i < kIterations; ++i) {
                     if (i % kRecreateEvery == kRecreateEvery - 1) {
                         vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
                         descriptorPool = CreateDescriptorPool(ctx);
                     } else {
                         vkResetDescriptorPool(ctx.device, descriptorPool, 0);
                     }

                     const VkCommandBuffer commandBuffer = commandBuffers[i % 2];
                     RecordFill(ctx, commandBuffer, AllocateSet(ctx, descriptorPool));
                     std::lock_guard lock(queueMutex);
                     Submit(queue, std::span(&commandBuffer, 1));
                 }

                 {
                     std::lock_guard lock(queueMutex);
                     vkQueueWaitIdle(queue);
                 }
                 vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
                 vkDestroyCommandPool(ctx.device, commandPool, nullptr);
             };
             const auto elapsed = Timed([&] {
                 std::vector<std::thread> threads;
                 for (int index = 0; index < kThreads; ++index) {
                     threads.emplace_back(worker, index);
                 }

                 for (std::thread& thread : threads) {
                     thread.join();
                 }
             });
             vkDeviceWaitIdle(ctx.device);
             return elapsed;
         },
         true},

        {"allocate and free a descriptor set (control)", Timing::Immediate,
         [](Context& ctx) {
             const VkDescriptorPool pool = CreateDescriptorPool(ctx);
             const VkDescriptorSet idle = AllocateSet(ctx, pool);
             const auto elapsed = Timed([&] { vkFreeDescriptorSets(ctx.device, pool, 1, &idle); });
             vkDestroyDescriptorPool(ctx.device, pool, nullptr);
             return elapsed;
         }},
    };
}

}  // namespace

int main(int argc, char** argv) {
    bool withLayer = true;
    bool withValidation = true;
    bool expectClean = false;
    bool expectErrors = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--no-layer") {
            withLayer = false;
        } else if (arg == "--no-validation") {
            withValidation = false;
        } else if (arg == "--expect-clean") {
            expectClean = true;
        } else if (arg == "--expect-errors") {
            expectErrors = true;
        } else {
            std::fprintf(stderr, "usage: %s [--no-layer] [--no-validation] (--expect-clean | --expect-errors)\n", argv[0]);
            return 2;
        }
    }

    if (!withValidation && expectErrors) {
        std::fprintf(stderr, "--expect-errors needs the validation layer\n");
        return 2;
    }

    if (expectClean == expectErrors) {
        std::fprintf(stderr, "pass exactly one of --expect-clean / --expect-errors\n");
        return 2;
    }

    Context ctx;
    if (const int status = CreateContext(ctx, withLayer, withValidation); status != 0) {
        DestroyContext(ctx);
        return status;
    }

    bool timingOk = true;
    for (const Scenario& scenario : Scenarios()) {
        if (scenario.onlyWithFix && !expectClean) {
            std::printf("[skip] %s\n", scenario.name);
            continue;
        }

        const int errorsBefore = g_errorCount.load();
        const Clock::duration elapsed = scenario.run(ctx);
        const int errors = g_errorCount.load() - errorsBefore;

        bool ok = true;
        if (expectClean) {
            ok = errors == 0;
            if (scenario.expected == Timing::Blocks && elapsed < kMinBlocked) {
                ok = false;
            }

            if (scenario.expected == Timing::Immediate && elapsed > kMaxUnblocked) {
                ok = false;
            }

            timingOk = timingOk && ok;
        }

        std::printf("[%s] %-62s %8.2f ms, %d validation error(s)\n", ok ? " ok " : "FAIL", scenario.name, Ms(elapsed), errors);
    }

    DestroyContext(ctx);

    std::printf("\nvalidation errors: %d\n", g_errorCount.load());
    for (const auto& [id, count] : g_errors) {
        std::printf("  %4d  %s\n", count, id.c_str());
    }

    if (expectClean) {
        const bool pass = g_errorCount.load() == 0 && timingOk;
        std::printf("%s: expected no validation errors and correct waits\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    const bool pass = g_errorCount.load() > 0;
    std::printf("%s: expected validation to catch the misuse\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
