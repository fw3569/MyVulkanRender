#include "device.h"

vk::SampleCountFlagBits GetMaxUsableSampleCount() {
  vk::PhysicalDeviceProperties properties =
      Context::Instance()->g_physical_device.getProperties();
  vk::SampleCountFlags counts = properties.limits.framebufferColorSampleCounts &
                                properties.limits.framebufferDepthSampleCounts;
  if (counts & vk::SampleCountFlagBits::e64) {
    return vk::SampleCountFlagBits::e64;
  }
  if (counts & vk::SampleCountFlagBits::e32) {
    return vk::SampleCountFlagBits::e32;
  }
  if (counts & vk::SampleCountFlagBits::e16) {
    return vk::SampleCountFlagBits::e16;
  }
  if (counts & vk::SampleCountFlagBits::e8) {
    return vk::SampleCountFlagBits::e8;
  }
  if (counts & vk::SampleCountFlagBits::e4) {
    return vk::SampleCountFlagBits::e4;
  }
  if (counts & vk::SampleCountFlagBits::e2) {
    return vk::SampleCountFlagBits::e2;
  }
  return vk::SampleCountFlagBits::e1;
}

void PickPhysicalDevice() {
  auto devices = Context::Instance()->g_vk_instance.enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  bool found = false;
  for (const auto& device : devices) {
    if (device.getProperties().apiVersion < VK_API_VERSION_1_3) {
      continue;
    }
    auto device_properties = device.getProperties();
    auto queue_families = device.getQueueFamilyProperties();
    uint32_t score = 0;
    if (device_properties.apiVersion < VK_API_VERSION_1_3 ||
        std::ranges::find_if(
            queue_families,
            [](const vk::QueueFamilyProperties& qfp) {
              return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) &&
                     (qfp.queueFlags & vk::QueueFlagBits::eCompute);
            }) == queue_families.end()) {
      continue;
    }
    auto extensions = device.enumerateDeviceExtensionProperties();
    bool all_found = true;
    for (const auto& req_extension :
         Context::Instance()->kRequiredDeviceExtensions) {
      all_found &=
          (std::ranges::find_if(
               extensions, [req_extension](const auto& extension) {
                 return strcmp(extension.extensionName, req_extension) == 0;
               }) != extensions.end());
    }
    if (!all_found) {
      continue;
    }
    auto features = device.getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
        vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
    bool supports_required_features =
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState &&
        features.get<vk::PhysicalDeviceFeatures2>()
            .features.samplerAnisotropy &&
        features.get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>()
            .timelineSemaphore &&
        features.get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>()
            .dynamicRenderingLocalRead;
    if (found == false ||
        (device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
         Context::Instance()->g_physical_device.getProperties().deviceType !=
             vk::PhysicalDeviceType::eDiscreteGpu &&
         supports_required_features)) {
      Context::Instance()->g_physical_device = device;
      found = true;
    }
  }
  if (found == false) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
  // Context::Instance()->g_msaa_samples = GetMaxUsableSampleCount();
  LOG("using physical device: ",
      Context::Instance()->g_physical_device.getProperties().deviceName);
}

void CreateLogicalDevice() {
  std::vector<vk::QueueFamilyProperties> queue_family_properties =
      Context::Instance()->g_physical_device.getQueueFamilyProperties();
  uint32_t queue_index = 0;
  for (; queue_index < queue_family_properties.size(); ++queue_index) {
    if ((queue_family_properties[queue_index].queueFlags &
         static_cast<vk::QueueFlags>(VK_QUEUE_GRAPHICS_BIT)) &&
        (Context::Instance()->g_physical_device.getSurfaceSupportKHR(
            queue_index, *Context::Instance()->g_surface))) {
      break;
    }
  }
  if (queue_index == queue_family_properties.size()) {
    throw std::runtime_error(
        "Could not find a queue for graphics and present!");
  }
  float queue_priorities[1] = {0.0};
  vk::DeviceQueueCreateInfo device_queue_create_info{
      .queueFamilyIndex = queue_index,
      .queueCount = 1,
      .pQueuePriorities = queue_priorities};
  vk::PhysicalDeviceFeatures devices_features;
  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                     vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
                     vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>
      feature_chain = {{.features = {.samplerAnisotropy = true}},
                       {.synchronization2 = true, .dynamicRendering = true},
                       {.extendedDynamicState = true},
                       {.timelineSemaphore = true},
                       {.dynamicRenderingLocalRead = true}};
  vk::DeviceCreateInfo device_create_info{
      .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &device_queue_create_info,
      .enabledExtensionCount = static_cast<uint32_t>(
          Context::Instance()->kRequiredDeviceExtensions.size()),
      .ppEnabledExtensionNames =
          Context::Instance()->kRequiredDeviceExtensions.data()};
  Context::Instance()->g_device = vk::raii::Device(
      Context::Instance()->g_physical_device, device_create_info);
  Context::Instance()->g_queue =
      vk::raii::Queue(Context::Instance()->g_device, queue_index, 0);
  Context::Instance()->g_queue_index = queue_index;
}
