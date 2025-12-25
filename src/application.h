#pragma once

#include "gui.h"
#include "render.h"
#include "third_part/vulkan_headers.h"

class Application {
 public:
  void Run();

 private:
  void Init();
  void InitWindow();
  void InitVulkan();
  void InitGui();
  void Work();
  bool Tick();
  bool PrepareData();
  bool DrawFrame();
  void Cleanup();
  double m_start_time = 0.0;
  double m_current_time = 0.0;
  double m_next_draw_time = 0.0;
  uint32_t m_frame_index = 0;
  Gui m_gui;
};
