#include "application.h"

#include <chrono>

#include "command_buffer.h"
#include "context.h"
#include "descriptor_set.h"
#include "device.h"
#include "model.h"
#include "render.h"
#include "swapchain.h"
#include "utils.h"
#include "vulkan_configure.h"

void Application::Run() {
  Init();
  Work();
  Cleanup();
}

void Application::Init() {
  InitWindow();
  InitVulkan();
  InitGui();
  m_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count() /
                 1000.0;
}

void Application::InitWindow() { m_gui.InitWindow(); }

void Application::InitVulkan() {
  LOG(std::string("DATA_FILE_PATH: ") + DATA_FILE_PATH);
  VulkanConfigure::CreateInstance();
  m_gui.CreateaSurface();
  InitDevice();
  SwapChainManager::CreateSwapChain();
  CreateCommandPool();
  LoadModel();
  RenderManager::Init();
}

void Application::InitGui() { m_gui.InitImGui(); }

void Application::Work() {
  while (!m_gui.Closed()) {
    constexpr uint32_t fps = 30;
    constexpr double draw_internal = 1.0 / fps;
    m_current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count() /
                     1000.0;
    Context::Instance()->g_time = m_current_time - m_start_time;
    m_gui.Update();
    if (m_next_draw_time < m_current_time && Tick()) {
      m_next_draw_time = m_current_time + draw_internal;
    }
  }

  Context::Instance()->g_device.waitIdle();
}

bool Application::Tick() { return PrepareData() && DrawFrame(); }

bool Application::PrepareData() {
  return RenderManager::PrepareData(m_frame_index);
}

bool Application::DrawFrame() {
  if (RenderManager::DrawFrame(m_frame_index)) {
    m_frame_index =
        (m_frame_index + 1) % Context::Instance()->g_frame_in_flight;
    return true;
  } else {
    return false;
  }
}

void Application::Cleanup() { m_gui.Cleanup(); }
