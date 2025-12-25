#pragma once

#include <vector>

namespace SwapChainManager {
void CreateSwapChain();
void RecreateSwapchain();
void RegisterRecreateFunction(void (*func)());
}  // namespace SwapChainManager
