#pragma once

#include <vector>

class SwapChainManager {
 public:
  static void CreateSwapChain();
  static void RecreateSwapchain();
  static void RegisterRecreateFunction(void (*func)());
  static std::vector<void (*)()> recreate_functions;
};
