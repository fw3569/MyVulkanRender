
#include "swapchain.h"

#include <algorithm>
#include <cstdint>
#include <mutex>

#include "context.h"
#include "memory.h"

std::vector<void (*)()> SwapChainManager::recreate_functions;

void SwapChainManager::CreateSwapChain() {
  auto surface_capabilities =
      Context::Instance()->g_physical_device.getSurfaceCapabilitiesKHR(
          Context::Instance()->g_surface);
  auto surface_formats =
      Context::Instance()->g_physical_device.getSurfaceFormatsKHR(
          *Context::Instance()->g_surface);
  auto surface_present_modes =
      Context::Instance()->g_physical_device.getSurfacePresentModesKHR(
          Context::Instance()->g_surface);
  vk::SurfaceFormatKHR select_format = surface_formats[0];
  for (const auto& format : surface_formats) {
    if (format.format == vk::Format::eB8G8R8A8Srgb &&
        format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      select_format = format;
      break;
    }
  }
  vk::PresentModeKHR select_present_mode = vk::PresentModeKHR::eFifo;
  for (const auto& mode : surface_present_modes) {
    if (mode == vk::PresentModeKHR::eMailbox) {
      select_present_mode = mode;
      break;
    }
  }
  vk::Extent2D select_extent;
  if (surface_capabilities.currentExtent.width ==
      (std::numeric_limits<uint32_t>::max)()) {
    int width, height;
    glfwGetFramebufferSize(Context::Instance()->g_window, &width, &height);
    select_extent = {
        std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.width,
                             surface_capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.height,
                             surface_capabilities.maxImageExtent.height)};
  } else {
    select_extent = surface_capabilities.currentExtent;
  }
  auto min_image_count =
      (std::max)(3u, surface_capabilities.minImageCount + 1u);
  if (surface_capabilities.maxImageCount > 0 &&
      min_image_count > surface_capabilities.maxImageCount) {
    min_image_count = surface_capabilities.maxImageCount;
  }
  vk::SwapchainCreateInfoKHR swapchain_create_info{
      .flags = vk::SwapchainCreateFlagsKHR(),
      .surface = Context::Instance()->g_surface,
      .minImageCount = min_image_count,
      .imageFormat = select_format.format,
      .imageColorSpace = select_format.colorSpace,
      .imageExtent = select_extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eTransferDst,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .preTransform = surface_capabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = select_present_mode,
      .clipped = true,
      .oldSwapchain = Context::Instance()->g_swapchain};
  Context::Instance()->g_swapchain = vk::raii::SwapchainKHR(
      Context::Instance()->g_device, swapchain_create_info);
  Context::Instance()->g_swapchain_images =
      Context::Instance()->g_swapchain.getImages();
  Context::Instance()->g_frame_in_flight =
      Context::Instance()->g_swapchain_images.size();
  Context::Instance()->g_swapchain_image_format = select_format.format;
  Context::Instance()->g_swapchain_extent = select_extent;
  Context::Instance()->g_swapchain_image_views.clear();
  for (auto image : Context::Instance()->g_swapchain_images) {
    Context::Instance()->g_swapchain_image_views.emplace_back(CreateImageView(
        image, 0, 1, Context::Instance()->g_swapchain_image_format,
        vk::ImageAspectFlagBits::eColor));
  }
}

void SwapChainManager::RecreateSwapchain() {
  int width = 0, height = 0;
  {
    std::lock_guard lock(Context::Instance()->g_window_resized_mtx);
    glfwGetFramebufferSize(Context::Instance()->g_window, &width, &height);
    Context::Instance()->g_window_resized = false;
  }
  while (width == 0 && height == 0) {
    glfwWaitEvents();
    std::lock_guard lock(Context::Instance()->g_window_resized_mtx);
    glfwGetFramebufferSize(Context::Instance()->g_window, &width, &height);
    Context::Instance()->g_window_resized = false;
  }

  Context::Instance()->g_device.waitIdle();
  CreateSwapChain();
  for (auto func : recreate_functions) {
    func();
  }
}

void SwapChainManager::RegisterRecreateFunction(void (*func)()) {
  recreate_functions.emplace_back(func);
}
