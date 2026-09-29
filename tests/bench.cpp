// SPDX-License-Identifier: MIT
//
// Measures the CPU cost the layer adds to the calls it intercepts. Rounds
// without and with the layer alternate, and each call's cost is the median of
// the round averages. The work is valid: every frame waits for its previous
// use to finish (outside the timed calls), so this runs safely with or without
// the layer, on any driver. Both columns include the clock reads around each
// call (tens of ns), which cancel out in the overhead.
//
// With --hz, frames are paced at that rate, as a compositor's are, and it also
// reports each frame's total cost of the intercepted calls and the frames that
// finished past their deadline.
//
//   bench [--iterations N] [--rounds N] [--binds N] [--hz N]
//
// The layer is taken from the build tree unless VK_ADD_LAYER_PATH is set.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr const uint32_t kFrames = 3;

#define CHECK(expr)                                                                                               \
    do {                                                                                                          \
        const VkResult result_ = (expr);                                                                          \
        if (result_ != VK_SUCCESS) {                                                                              \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #expr, static_cast<int>(result_)); \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

enum Call : size_t {
    kResetDescriptorPool,
    kAllocateDescriptorSets,
    kBeginCommandBuffer,
    kBindDescriptorSets,
    kQueueSubmit,
    kQueueSubmit2,
    kCallCount,
};

constexpr const std::array<const char*, kCallCount> kCallNames = {
    "vkResetDescriptorPool", "vkAllocateDescriptorSets", "vkBeginCommandBuffer", "vkCmdBindDescriptorSets",
    "vkQueueSubmit",         "vkQueueSubmit2",
};

struct Options {
    uint32_t iterations = 20000;
    uint32_t rounds = 5;
    uint32_t binds = 8;
    uint32_t hz = 0;  // unpaced unless --hz sets a rate
};

struct Round {
    // Average microseconds per call, per call type; negative when not measured.
    std::array<double, kCallCount> costs = {};
    // Microseconds spent in the intercepted calls, per frame.
    std::vector<double> frames;
    // Frames whose work ended past their deadline (paced rounds only).
    uint32_t missed = 0;
};

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    bool submit2 = false;
    std::string name;
};

Device CreateDevice(bool withLayer) {
    Device d;
    const char* layer = STEAMVR_COMPOSITOR_SYNC_LAYER_NAME;
    const VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "steamvr-compositor-sync bench",
        .applicationVersion = 1,
        .pEngineName = nullptr,
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instanceInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = withLayer ? 1u : 0u,
        .ppEnabledLayerNames = &layer,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };
    if (vkCreateInstance(&instanceInfo, nullptr, &d.instance) != VK_SUCCESS) {
        std::fprintf(stderr, "cannot create an instance%s\n", withLayer ? " with the layer" : "");
        std::exit(1);
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(d.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(d.instance, &count, devices.data());
    if (devices.empty()) {
        std::fprintf(stderr, "no Vulkan device\n");
        std::exit(1);
    }

    d.physicalDevice = devices[0];
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            d.physicalDevice = candidate;
            break;
        }
    }

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(d.physicalDevice, &properties);
    d.name = properties.deviceName;
    d.submit2 = properties.apiVersion >= VK_API_VERSION_1_3;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(d.physicalDevice, &familyCount, families.data());
    const auto family = std::find_if(families.begin(), families.end(),
                                     [](const VkQueueFamilyProperties& f) { return (f.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; });
    if (family == families.end()) {
        std::fprintf(stderr, "no compute queue\n");
        std::exit(1);
    }

    d.queueFamily = static_cast<uint32_t>(family - families.begin());

    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queueInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = d.queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkPhysicalDeviceVulkan13Features features13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = nullptr,
        .synchronization2 = VK_TRUE,
    };
    const VkDeviceCreateInfo deviceInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = d.submit2 ? &features13 : nullptr,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
        .pEnabledFeatures = nullptr,
    };
    CHECK(vkCreateDevice(d.physicalDevice, &deviceInfo, nullptr, &d.device));

    vkGetDeviceQueue(d.device, d.queueFamily, 0, &d.queue);
    return d;
}

void DestroyDevice(Device& d) {
    vkDestroyDevice(d.device, nullptr);
    vkDestroyInstance(d.instance, nullptr);
}

// One round: `iterations` frames, each resetting its descriptor pool,
// allocating and binding a set, re-recording its command buffer and
// submitting it (alternately with vkQueueSubmit and vkQueueSubmit2).
Round RunRound(const Device& d, const Options& options) {
    const VkDevice device = d.device;

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

    const VkCommandPoolCreateInfo commandPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = d.queueFamily,
    };
    VkCommandPool commandPool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool));

    const VkCommandBufferAllocateInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = kFrames,
    };
    std::array<VkCommandBuffer, kFrames> commandBuffers = {};
    CHECK(vkAllocateCommandBuffers(device, &commandBufferInfo, commandBuffers.data()));

    const VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };
    const VkDescriptorPoolCreateInfo descriptorPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    const VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    std::array<VkDescriptorPool, kFrames> descriptorPools = {};
    std::array<VkFence, kFrames> fences = {};
    for (uint32_t i = 0; i < kFrames; ++i) {
        CHECK(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPools[i]));
        CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fences[i]));
    }

    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    Round round;
    round.frames.reserve(options.iterations);
    std::array<Clock::duration, kCallCount> total = {};
    std::array<uint64_t, kCallCount> calls = {};
    Clock::duration frameCost = {};

    auto timed = [&](Call call, auto&& function) {
        const auto start = Clock::now();
        function();
        const Clock::duration elapsed = Clock::now() - start;
        total[call] += elapsed;
        ++calls[call];
        frameCost += elapsed;
    };

    const uint32_t warmup = std::max(options.iterations / 10, 100u);
    const auto period =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(options.hz == 0 ? 0.0 : 1.0 / options.hz));
    const Clock::time_point epoch = Clock::now();

    for (uint32_t i = 0; i < warmup + options.iterations; ++i) {
        if (i == warmup) {
            total = {};
            calls = {};
        }

        frameCost = {};
        const uint32_t frame = i % kFrames;
        CHECK(vkWaitForFences(device, 1, &fences[frame], VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(device, 1, &fences[frame]));

        const VkDescriptorPool pool = descriptorPools[frame];
        const VkCommandBuffer commandBuffer = commandBuffers[frame];
        timed(kResetDescriptorPool, [&] { CHECK(vkResetDescriptorPool(device, pool, 0)); });
        const VkDescriptorSetAllocateInfo setInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &setLayout,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        timed(kAllocateDescriptorSets, [&] { CHECK(vkAllocateDescriptorSets(device, &setInfo, &set)); });

        timed(kBeginCommandBuffer, [&] { CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo)); });
        for (uint32_t b = 0; b < options.binds; ++b) {
            timed(kBindDescriptorSets, [&] {
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
            });
        }

        CHECK(vkEndCommandBuffer(commandBuffer));

        if (d.submit2 && i % 2 == 1) {
            const VkCommandBufferSubmitInfo commandBufferSubmit = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = commandBuffer,
                .deviceMask = 0,
            };
            const VkSubmitInfo2 submit = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .pNext = nullptr,
                .flags = 0,
                .waitSemaphoreInfoCount = 0,
                .pWaitSemaphoreInfos = nullptr,
                .commandBufferInfoCount = 1,
                .pCommandBufferInfos = &commandBufferSubmit,
                .signalSemaphoreInfoCount = 0,
                .pSignalSemaphoreInfos = nullptr,
            };
            timed(kQueueSubmit2, [&] { CHECK(vkQueueSubmit2(d.queue, 1, &submit, fences[frame])); });
        } else {
            const VkSubmitInfo submit = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &commandBuffer,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr,
            };
            timed(kQueueSubmit, [&] { CHECK(vkQueueSubmit(d.queue, 1, &submit, fences[frame])); });
        }

        const bool measured = i >= warmup;
        if (measured) {
            round.frames.push_back(std::chrono::duration<double, std::micro>(frameCost).count());
        }

        if (options.hz != 0) {
            const Clock::time_point deadline = epoch + period * (i + 1);
            if (measured && Clock::now() > deadline) {
                ++round.missed;
            }
            std::this_thread::sleep_until(deadline);
        }
    }

    CHECK(vkDeviceWaitIdle(device));
    for (uint32_t i = 0; i < kFrames; ++i) {
        vkDestroyFence(device, fences[i], nullptr);
        vkDestroyDescriptorPool(device, descriptorPools[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

    for (size_t call = 0; call < kCallCount; ++call) {
        round.costs[call] = calls[call] == 0
                                ? -1.0
                                : std::chrono::duration<double, std::micro>(total[call]).count() / static_cast<double>(calls[call]);
    }
    return round;
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

// The value below which `fraction` of the sorted values fall.
double Percentile(const std::vector<double>& sorted, double fraction) {
    return sorted[static_cast<size_t>(fraction * static_cast<double>(sorted.size() - 1) + 0.5)];
}

bool ParseCount(const char* text, uint32_t& out) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (*end != '\0' || value == 0 || value > 10'000'000) {
        return false;
    }

    out = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool iterationsSet = false;
    bool roundsSet = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        uint32_t* target = arg == "--iterations" ? &options.iterations
                           : arg == "--rounds"   ? &options.rounds
                           : arg == "--binds"    ? &options.binds
                           : arg == "--hz"       ? &options.hz
                                                 : nullptr;
        if (target == nullptr || i + 1 == argc || !ParseCount(argv[++i], *target)) {
            std::fprintf(stderr, "usage: %s [--iterations N] [--rounds N] [--binds N] [--hz N]\n", argv[0]);
            return 2;
        }

        iterationsSet |= target == &options.iterations;
        roundsSet |= target == &options.rounds;
    }

    // Paced rounds take real time: default to 10 s of frames, 3 rounds.
    if (options.hz != 0) {
        options.iterations = iterationsSet ? options.iterations : options.hz * 10;
        options.rounds = roundsSet ? options.rounds : 3;
    }

    // Before the first loader call: the layer from the build tree, forced on
    // although this process is not vrcompositor.
    setenv("VK_ADD_LAYER_PATH", STEAMVR_COMPOSITOR_SYNC_TEST_LAYER_DIR, 0);
    setenv("STEAMVR_COMPOSITOR_SYNC_FORCE", "1", 1);

    // Indexed [without, with layer].
    std::array<std::array<std::vector<double>, kCallCount>, 2> samples;
    std::array<std::vector<double>, 2> frames;
    std::array<uint32_t, 2> missed = {};
    std::string deviceName;

    for (uint32_t round = 0; round < options.rounds; ++round) {
        for (const bool withLayer : {false, true}) {
            Device d = CreateDevice(withLayer);
            deviceName = d.name;
            const Round result = RunRound(d, options);
            DestroyDevice(d);

            for (size_t call = 0; call < kCallCount; ++call) {
                if (result.costs[call] >= 0.0) {
                    samples[withLayer][call].push_back(result.costs[call]);
                }
            }

            frames[withLayer].insert(frames[withLayer].end(), result.frames.begin(), result.frames.end());
            missed[withLayer] += result.missed;
        }
    }

    std::printf("device: %s\n", deviceName.c_str());
    if (options.hz != 0) {
        std::printf("paced at %u Hz (%.2f ms per frame); ", options.hz, 1000.0 / options.hz);
    }
    std::printf("%u frames per round, %u binds per frame; median of %u rounds (µs per call)\n\n", options.iterations, options.binds,
                options.rounds);

    std::printf("%-26s %14s %11s %10s\n", "call", "without layer", "with layer", "overhead");
    for (size_t call = 0; call < kCallCount; ++call) {
        if (samples[0][call].empty()) {
            continue;
        }

        const double without = Median(samples[0][call]);
        const double with = Median(samples[1][call]);
        std::printf("%-26s %14.3f %11.3f %+10.3f\n", kCallNames[call], without, with, with - without);
    }

    if (options.hz != 0) {
        const double budget = 1e6 / options.hz;
        std::printf("\n%-26s %8s %8s %8s %10s %13s\n", "per frame, all rounds (µs)", "median", "p99", "max", "of budget",
                    "late frames");
        for (const bool withLayer : {false, true}) {
            std::vector<double>& values = frames[withLayer];
            std::sort(values.begin(), values.end());
            const double median = Percentile(values, 0.5);
            std::printf("%-26s %8.2f %8.2f %8.2f %9.3f%% %6u / %zu\n", withLayer ? "with layer" : "without layer", median,
                        Percentile(values, 0.99), values.back(), 100.0 * median / budget, missed[withLayer], values.size());
        }
    }
    return 0;
}
