// SPDX-License-Identifier: MIT
//
// GPU-lifetime tracking for command buffers and descriptor pools.
//
// SteamVR's vrcompositor recycles command buffers and resets descriptor pools
// while the GPU may still execute work that uses them
// (ValveSoftware/SteamVR-for-Linux#952). The tracker gives every submission a
// completion point of its own, a value on a per-queue timeline semaphore, and
// holds or redirects each recycling call until that point is reached.
#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace steamvr_compositor_sync {

struct Config {
    // Tracking is enabled in this process (vrcompositor, or forced).
    bool active = false;
    std::string process;
    // Bound on every wait, so a submission that depends on work the caller
    // has not submitted yet cannot deadlock it.
    uint64_t waitTimeoutNs = 1'000'000'000;
    // Busy real pools per application pool before resets wait instead.
    size_t maxRetiredPools = 16;
    bool verbose = false;
};

const Config& GetConfig();
void Log(const char* format, ...) __attribute__((format(printf, 1, 2)));

// Next-in-chain functions the tracker calls itself.
struct DeviceDispatch {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
    PFN_vkDestroyDevice DestroyDevice = nullptr;
    PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;

    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkQueueSubmit2 QueueSubmit2 = nullptr;
    PFN_vkQueueSubmit2 QueueSubmit2KHR = nullptr;

    PFN_vkCreateSemaphore CreateSemaphore = nullptr;
    PFN_vkDestroySemaphore DestroySemaphore = nullptr;
    PFN_vkWaitSemaphores WaitSemaphores = nullptr;
    PFN_vkGetSemaphoreCounterValue GetSemaphoreCounterValue = nullptr;

    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers FreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkResetCommandBuffer ResetCommandBuffer = nullptr;
    PFN_vkResetCommandPool ResetCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkCmdExecuteCommands CmdExecuteCommands = nullptr;

    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
    PFN_vkResetDescriptorPool ResetDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets FreeDescriptorSets = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdBindDescriptorSets2 CmdBindDescriptorSets2 = nullptr;
    PFN_vkCmdBindDescriptorSets2 CmdBindDescriptorSets2KHR = nullptr;
};

// Keeps up to N elements inline and moves all of them to the heap beyond
// that, so short lists never allocate.
template <typename T, size_t N>
class SmallVector {
    static_assert(std::is_trivially_copyable_v<T>);

  public:
    const T* data() const { return m_heap.empty() ? m_inline.data() : m_heap.data(); }
    const T* begin() const { return data(); }
    const T* end() const { return data() + m_size; }
    T* begin() { return const_cast<T*>(std::as_const(*this).begin()); }
    T* end() { return const_cast<T*>(std::as_const(*this).end()); }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

    void push_back(const T& value) {
        if (m_heap.empty() && m_size < N) {
            m_inline[m_size++] = value;
            return;
        }

        if (m_heap.empty()) {
            m_heap.assign(m_inline.begin(), m_inline.end());
        }
        m_heap.push_back(value);
        ++m_size;
    }

    void clear() {
        m_heap.clear();
        m_size = 0;
    }

  private:
    std::array<T, N> m_inline{};
    std::vector<T> m_heap;  // every element, once there are more than N
    size_t m_size = 0;
};

// The steady-state paths (recording, submitting, resetting and reusing
// descriptor pools) do not allocate: short lists are inline, other lists keep
// their capacity when cleared, and descriptor set ownership lives in a pool
// resource that recycles its nodes.
class Tracker {
  public:
    struct QueueTimeline {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint64_t submitted = 0;
        // Highest value seen completed, so most checks skip the driver.
        mutable std::atomic<uint64_t> completed{0};
    };

    // A submission's completion point: `value` on `semaphore` completes once
    // everything submitted to the queue up to it has.
    struct Signal {
        QueueTimeline* timeline = nullptr;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint64_t value = 0;
    };

    Tracker(VkDevice device, const DeviceDispatch& vk);
    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;

    // Waits for the GPU and releases everything the tracker created. Call
    // before the device is destroyed.
    void Shutdown();

    void OnAllocateCommandBuffers(VkCommandPool pool, std::span<const VkCommandBuffer> commandBuffers);
    void BeforeFreeCommandBuffers(VkCommandPool pool, std::span<const VkCommandBuffer> commandBuffers);
    void BeforeBeginCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferUsageFlags flags);
    void BeforeResetCommandBuffer(VkCommandBuffer commandBuffer);
    void BeforeResetCommandPool(VkCommandPool pool);
    void BeforeDestroyCommandPool(VkCommandPool pool);
    void OnExecuteCommands(VkCommandBuffer primary, std::span<const VkCommandBuffer> secondaries);
    void OnBindDescriptorSets(VkCommandBuffer commandBuffer, std::span<const VkDescriptorSet> sets);

    // A submission, for which the caller holds the queue throughout:
    // BeforeSubmit, ReserveSignal, the application's batches with the signal
    // joined to the last one (or followed by SubmitSignal), FinishSubmit.

    // Waits for command buffers resubmitted while still pending without
    // SIMULTANEOUS_USE.
    void BeforeSubmit(std::span<const VkCommandBuffer> commandBuffers);
    // The queue's next completion point; nullopt if the queue is untracked.
    std::optional<Signal> ReserveSignal(VkQueue queue);
    // Queues the signal as a submission of its own.
    VkResult SubmitSignal(VkQueue queue, const Signal& signal);
    // Records the signal on the command buffers and the descriptor pools they
    // bound, or gives the value back if it was not queued.
    void FinishSubmit(const Signal& signal, bool signalled, std::span<const VkCommandBuffer> commandBuffers);

    // Application pool handles in, real pools underneath.
    VkResult CreateDescriptorPool(const VkDescriptorPoolCreateInfo* createInfo, const VkAllocationCallbacks* allocator,
                                  VkDescriptorPool* pool);
    void DestroyDescriptorPool(VkDescriptorPool pool, const VkAllocationCallbacks* allocator);
    VkResult ResetDescriptorPool(VkDescriptorPool pool, VkDescriptorPoolResetFlags flags);
    VkResult AllocateDescriptorSets(const VkDescriptorSetAllocateInfo* allocateInfo, VkDescriptorSet* sets);
    VkResult FreeDescriptorSets(VkDescriptorPool pool, std::span<const VkDescriptorSet> sets);

  private:
    // Complete once `timeline` reaches `value`; at most one entry per queue.
    struct Use {
        const QueueTimeline* timeline = nullptr;
        uint64_t value = 0;
    };
    static constexpr const size_t kInlineQueues = 4;
    using Uses = SmallVector<Use, kInlineQueues>;

    struct CommandBufferState {
        bool simultaneousUse = false;
        Uses pending;
        std::vector<VkDescriptorPool> descriptorPools;  // real pools bound while recording
        std::vector<VkCommandBuffer> secondaries;
    };

    struct RealPool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        Uses uses;
        std::vector<VkDescriptorSet> sets;
    };

    // One application descriptor pool. Its handle is the first real pool's, so
    // calls the layer does not intercept still reach a valid pool.
    struct LogicalPool {
        VkDescriptorPool appHandle = VK_NULL_HANDLE;
        RealPool* current = nullptr;
        std::vector<std::unique_ptr<RealPool>> pools;
        std::vector<RealPool*> retired;                     // reset by the application, still in use
        std::vector<RealPool*> spare;                       // retired, idle and reset
        std::vector<std::pair<RealPool*, Uses>> recycling;  // RecycleRetired's scratch list
        bool canSwap = false;
        VkDescriptorPoolCreateFlags flags = 0;
        uint32_t maxSets = 0;
        std::vector<VkDescriptorPoolSize> sizes;
        std::optional<uint32_t> maxInlineUniformBlockBindings;
    };

    // A real pool the application destroyed while it was still in use.
    struct DeferredPool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        Uses uses;
    };

    struct Stats {
        std::atomic<uint64_t> commandBufferWaits{0};
        std::atomic<uint64_t> submitWaits{0};
        std::atomic<uint64_t> poolWaits{0};
        std::atomic<uint64_t> timeouts{0};
        std::atomic<uint64_t> poolSwaps{0};
        std::atomic<uint64_t> deferredPoolDestroys{0};
        std::atomic<uint64_t> waitNsTotal{0};
        std::atomic<uint64_t> waitNsMax{0};
    };

    static void AddUse(Uses& uses, Use use);

    // These must not hold m_mutex: they call the driver or log. A logical
    // pool's own operations are externally synchronized by the application,
    // which keeps it alive and its real pools exclusive across the unlocked
    // parts.
    bool IsComplete(const Uses& uses) const;
    // Bounded; false on timeout.
    bool Wait(const Uses& uses, std::atomic<uint64_t>& counter, const char* what);
    void WaitIfPending(const Uses& uses, std::atomic<uint64_t>& counter, const char* what);
    void WaitForCommandBuffers(std::span<const VkCommandBuffer> commandBuffers, const char* what);
    std::span<const VkCommandBuffer> CommandBuffersOf(VkCommandPool pool);
    RealPool* TakeSpare(LogicalPool& logical);
    void RecycleRetired(LogicalPool& logical);
    // Destroys the idle pools and defers the rest, emptying `pools`; returns
    // how many were deferred.
    size_t DestroyIdle(std::vector<DeferredPool>& pools);
    void MaybeLogStats(bool force);

    // The rest requires m_mutex.
    QueueTimeline* TimelineLocked(VkQueue queue);
    Uses PendingLocked(std::span<const VkCommandBuffer> commandBuffers, bool skipSimultaneousUse) const;
    void ClearRecordingLocked(std::span<const VkCommandBuffer> commandBuffers);
    void MarkSubmittedLocked(VkCommandBuffer commandBuffer, const Use& use, int depth);
    RealPool* AddRealPoolLocked(LogicalPool& logical, VkDescriptorPool handle);
    void ForgetSetsLocked(RealPool& real);

    const VkDevice m_device;
    const DeviceDispatch m_vk;

    // Guards everything below, including the node pool behind m_setOwners.
    std::mutex m_mutex;
    std::pmr::unsynchronized_pool_resource m_nodePool;
    std::unordered_map<VkQueue, std::unique_ptr<QueueTimeline>> m_timelines;
    std::unordered_map<VkCommandBuffer, CommandBufferState> m_commandBuffers;
    std::unordered_map<VkCommandPool, std::vector<VkCommandBuffer>> m_commandPools;
    std::unordered_map<VkDescriptorPool, std::unique_ptr<LogicalPool>> m_logicalPools;
    std::unordered_map<VkDescriptorPool, RealPool*> m_realPools;
    std::pmr::unordered_map<VkDescriptorSet, RealPool*> m_setOwners{&m_nodePool};
    std::vector<DeferredPool> m_deferredPools;

    Stats m_stats;
    std::atomic<uint64_t> m_lastStatsLogNs{0};
    std::atomic<uint64_t> m_lastStatsSignature{0};
    std::atomic<bool> m_signalFailureLogged{false};
    std::atomic<bool> m_allocatorPoolLogged{false};
};

}  // namespace steamvr_compositor_sync
