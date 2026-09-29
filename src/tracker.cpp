// SPDX-License-Identifier: MIT
#include "tracker.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string_view>
#include <utility>

#include <unistd.h>

namespace steamvr_compositor_sync {

namespace {

constexpr const uint64_t kStatsLogIntervalNs = 10'000'000'000ull;
constexpr const int kMaxSecondaryDepth = 4;

bool EnvIs(const char* name, const char* value) {
    const char* env = std::getenv(name);
    return env != nullptr && std::strcmp(env, value) == 0;
}

uint64_t EnvU64(const char* name, uint64_t fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long long value = std::strtoull(env, &end, 10);
    return *end == '\0' ? value : fallback;
}

std::string ProcessName() {
    char path[PATH_MAX] = {};
    const ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (length <= 0) {
        return {};
    }

    const std::string_view full(path, static_cast<size_t>(length));
    return std::string(full.substr(full.rfind('/') + 1));
}

Config LoadConfig() {
    Config config = {
        .active = false,
        .process = ProcessName(),
        .waitTimeoutNs = std::min<uint64_t>(EnvU64("STEAMVR_COMPOSITOR_SYNC_TIMEOUT_MS", 1000), 60'000) * 1'000'000ull,
        .maxRetiredPools = static_cast<size_t>(EnvU64("STEAMVR_COMPOSITOR_SYNC_MAX_RETIRED_POOLS", 16)),
        .verbose = EnvIs("STEAMVR_COMPOSITOR_SYNC_VERBOSE", "1"),
    };
    config.active = config.process == "vrcompositor" || EnvIs("STEAMVR_COMPOSITOR_SYNC_FORCE", "1");
    return config;
}

uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void ClearRecording(auto& state) {
    state.descriptorPools.clear();
    state.secondaries.clear();
}

void StoreMax(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (value > current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

}  // namespace

const Config& GetConfig() {
    static const Config config = LoadConfig();
    return config;
}

void Log(const char* format, ...) {
    char message[1024];

    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);

    std::fprintf(stderr, "[steamvr-compositor-sync] %s\n", message);
    std::fflush(stderr);
}

Tracker::Tracker(VkDevice device, const DeviceDispatch& vk) : m_device(device), m_vk(vk) {}

void Tracker::AddUse(Tracker::Uses& uses, Tracker::Use use) {
    for (Use& existing : uses) {
        if (existing.timeline == use.timeline) {
            existing.value = std::max(existing.value, use.value);
            return;
        }
    }
    uses.push_back(use);
}

bool Tracker::IsComplete(const Tracker::Uses& uses) const {
    for (const Use& use : uses) {
        if (use.value <= use.timeline->completed.load(std::memory_order_relaxed)) {
            continue;
        }

        uint64_t value = 0;
        if (m_vk.GetSemaphoreCounterValue(m_device, use.timeline->semaphore, &value) != VK_SUCCESS) {
            return true;  // device lost: nothing left to protect
        }

        StoreMax(use.timeline->completed, value);
        if (value < use.value) {
            return false;
        }
    }
    return true;
}

bool Tracker::Wait(const Tracker::Uses& uses, std::atomic<uint64_t>& counter, const char* what) {
    SmallVector<VkSemaphore, kInlineQueues> semaphores;
    SmallVector<uint64_t, kInlineQueues> values;

    for (const Use& use : uses) {
        semaphores.push_back(use.timeline->semaphore);
        values.push_back(use.value);
    }

    const VkSemaphoreWaitInfo waitInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = static_cast<uint32_t>(semaphores.size()),
        .pSemaphores = semaphores.data(),
        .pValues = values.data(),
    };

    const uint64_t start = NowNs();
    const VkResult result = m_vk.WaitSemaphores(m_device, &waitInfo, GetConfig().waitTimeoutNs);
    const uint64_t elapsed = NowNs() - start;

    counter.fetch_add(1, std::memory_order_relaxed);
    m_stats.waitNsTotal.fetch_add(elapsed, std::memory_order_relaxed);
    StoreMax(m_stats.waitNsMax, elapsed);

    if (result == VK_SUCCESS) {
        for (const Use& use : uses) {
            StoreMax(use.timeline->completed, use.value);
        }
    } else if (result == VK_TIMEOUT) {
        m_stats.timeouts.fetch_add(1, std::memory_order_relaxed);
        Log("%s: GPU work still pending after %llu ms, continuing without it", what,
            static_cast<unsigned long long>(elapsed / 1'000'000));
        return false;
    }

    if (GetConfig().verbose) {
        Log("%s: waited %.3f ms for the GPU", what, static_cast<double>(elapsed) / 1e6);
    }
    return true;  // success, or device lost
}

void Tracker::WaitIfPending(const Tracker::Uses& uses, std::atomic<uint64_t>& counter, const char* what) {
    if (!IsComplete(uses)) {
        Wait(uses, counter, what);
    }
}

// --- command buffers -------------------------------------------------------

// Completed uses are left in place: they are at most one per queue, and the
// completed-value cache answers them without the driver.
Tracker::Uses Tracker::PendingLocked(std::span<const VkCommandBuffer> commandBuffers, bool skipSimultaneousUse) const {
    Uses uses;
    for (VkCommandBuffer commandBuffer : commandBuffers) {
        const auto it = m_commandBuffers.find(commandBuffer);
        if (it == m_commandBuffers.end() || (skipSimultaneousUse && it->second.simultaneousUse)) {
            continue;
        }

        for (const Use& use : it->second.pending) {
            AddUse(uses, use);
        }
    }
    return uses;
}

// The list stays put after the lock is released: the application externally
// synchronizes the pool with allocating and freeing from it.
std::span<const VkCommandBuffer> Tracker::CommandBuffersOf(VkCommandPool pool) {
    std::lock_guard lock(m_mutex);
    const auto it = m_commandPools.find(pool);
    return it == m_commandPools.end() ? std::span<const VkCommandBuffer>() : it->second;
}

void Tracker::ClearRecordingLocked(std::span<const VkCommandBuffer> commandBuffers) {
    for (VkCommandBuffer commandBuffer : commandBuffers) {
        if (const auto it = m_commandBuffers.find(commandBuffer); it != m_commandBuffers.end()) {
            ClearRecording(it->second);
        }
    }
}

void Tracker::WaitForCommandBuffers(std::span<const VkCommandBuffer> commandBuffers, const char* what) {
    Uses uses;
    {
        std::lock_guard lock(m_mutex);
        uses = PendingLocked(commandBuffers, false);
    }
    WaitIfPending(uses, m_stats.commandBufferWaits, what);
}

void Tracker::OnAllocateCommandBuffers(VkCommandPool pool, std::span<const VkCommandBuffer> commandBuffers) {
    std::lock_guard lock(m_mutex);

    std::vector<VkCommandBuffer>& ofPool = m_commandPools[pool];
    ofPool.insert(ofPool.end(), commandBuffers.begin(), commandBuffers.end());

    for (VkCommandBuffer commandBuffer : commandBuffers) {
        m_commandBuffers[commandBuffer] = CommandBufferState{};
    }
}

void Tracker::BeforeFreeCommandBuffers(VkCommandPool pool, std::span<const VkCommandBuffer> commandBuffers) {
    WaitForCommandBuffers(commandBuffers, "vkFreeCommandBuffers");

    std::lock_guard lock(m_mutex);
    const auto ofPool = m_commandPools.find(pool);

    for (VkCommandBuffer commandBuffer : commandBuffers) {
        m_commandBuffers.erase(commandBuffer);

        if (ofPool != m_commandPools.end()) {
            std::erase(ofPool->second, commandBuffer);
        }
    }
}

void Tracker::BeforeBeginCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferUsageFlags flags) {
    WaitForCommandBuffers(std::span(&commandBuffer, 1), "vkBeginCommandBuffer");

    std::lock_guard lock(m_mutex);
    CommandBufferState& state = m_commandBuffers[commandBuffer];
    ClearRecording(state);
    state.simultaneousUse = (flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) != 0;
}

void Tracker::BeforeResetCommandBuffer(VkCommandBuffer commandBuffer) {
    WaitForCommandBuffers(std::span(&commandBuffer, 1), "vkResetCommandBuffer");

    std::lock_guard lock(m_mutex);
    ClearRecordingLocked(std::span(&commandBuffer, 1));
}

void Tracker::BeforeResetCommandPool(VkCommandPool pool) {
    const std::span<const VkCommandBuffer> commandBuffers = CommandBuffersOf(pool);
    WaitForCommandBuffers(commandBuffers, "vkResetCommandPool");

    std::lock_guard lock(m_mutex);
    ClearRecordingLocked(commandBuffers);
}

void Tracker::BeforeDestroyCommandPool(VkCommandPool pool) {
    const std::span<const VkCommandBuffer> commandBuffers = CommandBuffersOf(pool);
    WaitForCommandBuffers(commandBuffers, "vkDestroyCommandPool");

    std::lock_guard lock(m_mutex);
    for (VkCommandBuffer commandBuffer : commandBuffers) {
        m_commandBuffers.erase(commandBuffer);
    }

    m_commandPools.erase(pool);
}

void Tracker::OnExecuteCommands(VkCommandBuffer primary, std::span<const VkCommandBuffer> secondaries) {
    std::lock_guard lock(m_mutex);
    std::vector<VkCommandBuffer>& list = m_commandBuffers[primary].secondaries;
    list.insert(list.end(), secondaries.begin(), secondaries.end());
}

void Tracker::OnBindDescriptorSets(VkCommandBuffer commandBuffer, std::span<const VkDescriptorSet> sets) {
    std::lock_guard lock(m_mutex);

    std::vector<VkDescriptorPool>* pools = nullptr;
    for (VkDescriptorSet set : sets) {
        const auto owner = m_setOwners.find(set);
        if (owner == m_setOwners.end()) {
            continue;
        }

        if (pools == nullptr) {
            pools = &m_commandBuffers[commandBuffer].descriptorPools;
        }

        if (std::find(pools->begin(), pools->end(), owner->second->handle) == pools->end()) {
            pools->push_back(owner->second->handle);
        }
    }
}

// --- submission ------------------------------------------------------------

Tracker::QueueTimeline* Tracker::TimelineLocked(VkQueue queue) {
    std::unique_ptr<QueueTimeline>& timeline = m_timelines[queue];
    if (timeline) {
        return timeline.get();
    }

    const VkSemaphoreTypeCreateInfo typeInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &typeInfo,
        .flags = 0,
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (m_vk.CreateSemaphore(m_device, &createInfo, nullptr, &semaphore) != VK_SUCCESS) {
        m_timelines.erase(queue);
        return nullptr;
    }

    timeline = std::make_unique<QueueTimeline>();
    timeline->semaphore = semaphore;
    return timeline.get();
}

void Tracker::BeforeSubmit(std::span<const VkCommandBuffer> commandBuffers) {
    Uses uses;
    {
        std::lock_guard lock(m_mutex);
        uses = PendingLocked(commandBuffers, true);
    }
    WaitIfPending(uses, m_stats.submitWaits, "vkQueueSubmit (command buffer still pending)");
}

std::optional<Tracker::Signal> Tracker::ReserveSignal(VkQueue queue) {
    std::lock_guard lock(m_mutex);

    QueueTimeline* timeline = TimelineLocked(queue);
    if (timeline == nullptr) {
        return std::nullopt;
    }

    return Signal{
        .timeline = timeline,
        .semaphore = timeline->semaphore,
        .value = ++timeline->submitted,
    };
}

VkResult Tracker::SubmitSignal(VkQueue queue, const Tracker::Signal& signal) {
    const VkTimelineSemaphoreSubmitInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = 0,
        .pWaitSemaphoreValues = nullptr,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal.value,
    };
    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timelineInfo,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 0,
        .pCommandBuffers = nullptr,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal.semaphore,
    };
    const VkResult result = m_vk.QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result != VK_SUCCESS && !m_signalFailureLogged.exchange(true, std::memory_order_relaxed)) {
        Log("completion signal submit failed (%d); submissions are not tracked", static_cast<int>(result));
    }
    return result;
}

void Tracker::FinishSubmit(const Tracker::Signal& signal, bool signalled, std::span<const VkCommandBuffer> commandBuffers) {
    // Swapped with m_deferredPools, so both keep their capacity.
    thread_local std::vector<DeferredPool> deferred;
    {
        std::lock_guard lock(m_mutex);
        if (!signalled) {
            --signal.timeline->submitted;  // the caller still holds the queue
            return;
        }

        const Use use = {
            .timeline = signal.timeline,
            .value = signal.value,
        };
        for (VkCommandBuffer commandBuffer : commandBuffers) {
            MarkSubmittedLocked(commandBuffer, use, 0);
        }

        deferred.swap(m_deferredPools);
    }

    if (!deferred.empty()) {
        DestroyIdle(deferred);
    }

    MaybeLogStats(false);
}

void Tracker::MarkSubmittedLocked(VkCommandBuffer commandBuffer, const Tracker::Use& use, int depth) {
    const auto it = m_commandBuffers.find(commandBuffer);
    if (it == m_commandBuffers.end() || depth > kMaxSecondaryDepth) {
        return;
    }

    CommandBufferState& state = it->second;
    AddUse(state.pending, use);

    for (VkDescriptorPool pool : state.descriptorPools) {
        if (const auto real = m_realPools.find(pool); real != m_realPools.end()) {
            AddUse(real->second->uses, use);
        }
    }

    for (VkCommandBuffer secondary : state.secondaries) {
        MarkSubmittedLocked(secondary, use, depth + 1);
    }
}

// --- descriptor pools ------------------------------------------------------

VkResult Tracker::CreateDescriptorPool(const VkDescriptorPoolCreateInfo* createInfo, const VkAllocationCallbacks* allocator,
                                       VkDescriptorPool* pool) {
    const VkResult result = m_vk.CreateDescriptorPool(m_device, createInfo, allocator, pool);
    if (result != VK_SUCCESS) {
        return result;
    }

    if (allocator != nullptr) {
        // Extra and deferred pools would outlive the application's view of its
        // allocator; leave such pools alone.
        if (!m_allocatorPoolLogged.exchange(true, std::memory_order_relaxed)) {
            Log("descriptor pools with custom allocators are not protected");
        }
        return result;
    }

    auto logical = std::make_unique<LogicalPool>(LogicalPool{
        .appHandle = *pool,
        .current = nullptr,
        .pools = {},
        .retired = {},
        .spare = {},
        .recycling = {},
        .canSwap = true,
        .flags = createInfo->flags,
        .maxSets = createInfo->maxSets,
        .sizes = std::vector(createInfo->pPoolSizes, createInfo->pPoolSizes + createInfo->poolSizeCount),
        .maxInlineUniformBlockBindings = std::nullopt,
    });
    for (auto* next = static_cast<const VkBaseInStructure*>(createInfo->pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO) {
            logical->maxInlineUniformBlockBindings =
                reinterpret_cast<const VkDescriptorPoolInlineUniformBlockCreateInfo*>(next)->maxInlineUniformBlockBindings;
        } else {
            logical->canSwap = false;  // a copy could differ: wait instead of swapping
        }
    }

    std::lock_guard lock(m_mutex);
    logical->current = AddRealPoolLocked(*logical, *pool);
    m_logicalPools[*pool] = std::move(logical);
    return result;
}

Tracker::RealPool* Tracker::AddRealPoolLocked(Tracker::LogicalPool& logical, VkDescriptorPool handle) {
    auto& real = logical.pools.emplace_back(std::make_unique<RealPool>(RealPool{
        .handle = handle,
    }));
    m_realPools[handle] = real.get();
    return real.get();
}

Tracker::RealPool* Tracker::TakeSpare(Tracker::LogicalPool& logical) {
    {
        std::lock_guard lock(m_mutex);
        if (!logical.canSwap || logical.retired.size() >= GetConfig().maxRetiredPools) {
            return nullptr;
        }

        if (!logical.spare.empty()) {
            RealPool* real = logical.spare.back();
            logical.spare.pop_back();
            return real;
        }
    }

    const VkDescriptorPoolInlineUniformBlockCreateInfo inlineInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO,
        .pNext = nullptr,
        .maxInlineUniformBlockBindings = logical.maxInlineUniformBlockBindings.value_or(0),
    };
    const VkDescriptorPoolCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = logical.maxInlineUniformBlockBindings ? &inlineInfo : nullptr,
        .flags = logical.flags,
        .maxSets = logical.maxSets,
        .poolSizeCount = static_cast<uint32_t>(logical.sizes.size()),
        .pPoolSizes = logical.sizes.data(),
    };
    VkDescriptorPool handle = VK_NULL_HANDLE;
    if (m_vk.CreateDescriptorPool(m_device, &createInfo, nullptr, &handle) != VK_SUCCESS) {
        return nullptr;
    }

    std::lock_guard lock(m_mutex);
    return AddRealPoolLocked(logical, handle);
}

void Tracker::RecycleRetired(Tracker::LogicalPool& logical) {
    std::vector<std::pair<RealPool*, Uses>>& idle = logical.recycling;
    idle.clear();
    {
        std::lock_guard lock(m_mutex);
        for (RealPool* real : logical.retired) {
            idle.emplace_back(real, real->uses);
        }
    }

    std::erase_if(idle, [&](const auto& entry) { return !IsComplete(entry.second); });
    if (idle.empty()) {
        return;
    }

    for (const auto& [real, uses] : idle) {
        m_vk.ResetDescriptorPool(m_device, real->handle, 0);
    }

    std::lock_guard lock(m_mutex);
    for (const auto& [real, uses] : idle) {
        real->uses.clear();
        std::erase(logical.retired, real);
        logical.spare.push_back(real);
    }
}

void Tracker::ForgetSetsLocked(Tracker::RealPool& real) {
    for (VkDescriptorSet set : real.sets) {
        m_setOwners.erase(set);
    }

    real.sets.clear();
}

VkResult Tracker::AllocateDescriptorSets(const VkDescriptorSetAllocateInfo* allocateInfo, VkDescriptorSet* sets) {
    LogicalPool* logical = nullptr;
    RealPool* real = nullptr;
    {
        std::lock_guard lock(m_mutex);
        if (const auto it = m_logicalPools.find(allocateInfo->descriptorPool); it != m_logicalPools.end()) {
            logical = it->second.get();
            real = logical->current;
        }
    }

    if (logical == nullptr) {
        return m_vk.AllocateDescriptorSets(m_device, allocateInfo, sets);
    }

    RecycleRetired(*logical);

    VkDescriptorSetAllocateInfo translated = *allocateInfo;
    translated.descriptorPool = real->handle;
    const VkResult result = m_vk.AllocateDescriptorSets(m_device, &translated, sets);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(m_mutex);
        for (VkDescriptorSet set : std::span(sets, allocateInfo->descriptorSetCount)) {
            m_setOwners[set] = real;
            real->sets.push_back(set);
        }
    }

    return result;
}

VkResult Tracker::FreeDescriptorSets(VkDescriptorPool pool, std::span<const VkDescriptorSet> sets) {
    // Sets grouped by the real pool that owns them: one driver call per pool.
    thread_local std::vector<std::pair<VkDescriptorPool, VkDescriptorSet>> owned;
    thread_local std::vector<VkDescriptorSet> group;
    owned.clear();
    {
        std::lock_guard lock(m_mutex);
        if (!m_logicalPools.contains(pool)) {
            return m_vk.FreeDescriptorSets(m_device, pool, static_cast<uint32_t>(sets.size()), sets.data());
        }

        for (VkDescriptorSet set : sets) {
            const auto owner = m_setOwners.find(set);
            if (owner == m_setOwners.end()) {
                continue;  // null, or in a retired pool that is reset as a whole
            }

            owned.emplace_back(owner->second->handle, set);
            std::erase(owner->second->sets, set);
            m_setOwners.erase(owner);
        }
    }

    std::ranges::sort(owned, std::less{}, &std::pair<VkDescriptorPool, VkDescriptorSet>::first);

    for (auto first = owned.begin(); first != owned.end();) {
        const auto last = std::find_if(first, owned.end(), [&](const auto& entry) { return entry.first != first->first; });
        group.clear();
        std::transform(first, last, std::back_inserter(group), [](const auto& entry) { return entry.second; });
        m_vk.FreeDescriptorSets(m_device, first->first, static_cast<uint32_t>(group.size()), group.data());
        first = last;
    }

    return VK_SUCCESS;  // vkFreeDescriptorSets cannot fail
}

VkResult Tracker::ResetDescriptorPool(VkDescriptorPool pool, VkDescriptorPoolResetFlags flags) {
    LogicalPool* logical = nullptr;
    RealPool* current = nullptr;
    Uses uses;
    {
        std::lock_guard lock(m_mutex);
        if (const auto it = m_logicalPools.find(pool); it != m_logicalPools.end()) {
            logical = it->second.get();
            current = logical->current;
            uses = current->uses;
        }
    }

    if (logical == nullptr) {
        return m_vk.ResetDescriptorPool(m_device, pool, flags);
    }

    RecycleRetired(*logical);

    if (!IsComplete(uses)) {
        if (RealPool* next = TakeSpare(*logical)) {
            std::lock_guard lock(m_mutex);
            ForgetSetsLocked(*current);
            logical->retired.push_back(current);
            logical->current = next;
            m_stats.poolSwaps.fetch_add(1, std::memory_order_relaxed);
            return VK_SUCCESS;
        }

        Wait(uses, m_stats.poolWaits, "vkResetDescriptorPool (pool still in use)");
    }

    {
        std::lock_guard lock(m_mutex);
        current->uses.clear();
        ForgetSetsLocked(*current);
    }
    return m_vk.ResetDescriptorPool(m_device, current->handle, flags);
}

void Tracker::DestroyDescriptorPool(VkDescriptorPool pool, const VkAllocationCallbacks* allocator) {
    std::vector<DeferredPool> pools;
    std::unique_ptr<LogicalPool> logical;
    {
        std::lock_guard lock(m_mutex);
        const auto it = m_logicalPools.find(pool);

        if (it != m_logicalPools.end()) {
            logical = std::move(it->second);
            m_logicalPools.erase(it);

            for (const std::unique_ptr<RealPool>& real : logical->pools) {
                ForgetSetsLocked(*real);
                m_realPools.erase(real->handle);
                pools.push_back(DeferredPool{
                    .handle = real->handle,
                    .uses = real->uses,
                });
            }
        }
    }

    if (!logical) {
        m_vk.DestroyDescriptorPool(m_device, pool, allocator);
        return;
    }

    if (DestroyIdle(pools) > 0) {
        m_stats.deferredPoolDestroys.fetch_add(1, std::memory_order_relaxed);
    }
}

size_t Tracker::DestroyIdle(std::vector<Tracker::DeferredPool>& pools) {
    std::erase_if(pools, [&](const DeferredPool& pool) {
        if (!IsComplete(pool.uses)) {
            return false;
        }

        m_vk.DestroyDescriptorPool(m_device, pool.handle, nullptr);
        return true;
    });

    const size_t deferred = pools.size();
    if (deferred > 0) {
        std::lock_guard lock(m_mutex);
        m_deferredPools.insert(m_deferredPools.end(), pools.begin(), pools.end());
    }

    pools.clear();
    return deferred;
}

// --- lifetime and diagnostics ---------------------------------------------

void Tracker::MaybeLogStats(bool force) {
    const uint64_t now = NowNs();
    uint64_t last = m_lastStatsLogNs.load(std::memory_order_relaxed);

    if (force) {
        m_lastStatsLogNs.store(now, std::memory_order_relaxed);
    } else if (now - last < kStatsLogIntervalNs ||
               !m_lastStatsLogNs.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;  // too soon, or another thread is logging
    }

    const uint64_t commandBufferWaits = m_stats.commandBufferWaits.load();
    const uint64_t submitWaits = m_stats.submitWaits.load();
    const uint64_t poolWaits = m_stats.poolWaits.load();
    const uint64_t timeouts = m_stats.timeouts.load();
    const uint64_t poolSwaps = m_stats.poolSwaps.load();
    const uint64_t deferred = m_stats.deferredPoolDestroys.load();
    // Counters only grow, so an unchanged sum means nothing happened.
    const uint64_t signature = commandBufferWaits + submitWaits + poolWaits + timeouts + poolSwaps + deferred;
    if (m_lastStatsSignature.exchange(signature, std::memory_order_relaxed) == signature) {
        return;
    }

    Log("prevented reuse of in-flight GPU work: command buffer waits %llu, resubmit waits %llu, "
        "descriptor pool swaps %llu, pool waits %llu, deferred pool destroys %llu, timeouts %llu; "
        "wait total %.1f ms, max %.3f ms",
        static_cast<unsigned long long>(commandBufferWaits), static_cast<unsigned long long>(submitWaits),
        static_cast<unsigned long long>(poolSwaps), static_cast<unsigned long long>(poolWaits),
        static_cast<unsigned long long>(deferred), static_cast<unsigned long long>(timeouts),
        static_cast<double>(m_stats.waitNsTotal.load()) / 1e6, static_cast<double>(m_stats.waitNsMax.load()) / 1e6);
}

void Tracker::Shutdown() {
    m_vk.DeviceWaitIdle(m_device);

    MaybeLogStats(true);

    std::lock_guard lock(m_mutex);
    for (const DeferredPool& pool : m_deferredPools) {
        m_vk.DestroyDescriptorPool(m_device, pool.handle, nullptr);
    }

    // Pools the application did not destroy keep their own handle; the extra
    // real pools behind them belong to the layer.
    for (const auto& [handle, logical] : m_logicalPools) {
        for (const std::unique_ptr<RealPool>& real : logical->pools) {
            if (real->handle != logical->appHandle) {
                m_vk.DestroyDescriptorPool(m_device, real->handle, nullptr);
            }
        }
    }

    for (const auto& [queue, timeline] : m_timelines) {
        m_vk.DestroySemaphore(m_device, timeline->semaphore, nullptr);
    }

    m_deferredPools.clear();
    m_commandPools.clear();
    m_logicalPools.clear();
    m_realPools.clear();
    m_setOwners.clear();
    m_timelines.clear();
    m_commandBuffers.clear();
}

}  // namespace steamvr_compositor_sync
