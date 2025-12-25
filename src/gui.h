#pragma once

#include <atomic>
#include <vector>

#include "third_part/glfw_headers.h"
#include "third_part/imgui_headers.h"
#include "third_part/vulkan_headers.h"

class Gui {
 public:
  void InitWindow();
  void CreateaSurface();
  void InitImGui();
  void Cleanup();
  bool Closed();
  void Update();
};
