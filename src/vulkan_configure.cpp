#include "vulkan_configure.h"

#include <algorithm>
#include <vector>

#include "context.h"

void VulkanConfigure::CreateInstance() {
  constexpr vk::ApplicationInfo app_info{
      .pApplicationName = "Proj",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14};

  std::vector<char const*> required_layers;
  if (Context::Instance()->kEnableValidationLayers) {
    required_layers.assign(Context::Instance()->kValidationLayers.begin(),
                           Context::Instance()->kValidationLayers.end());
  }
  auto layer_properties =
      Context::Instance()->g_vk_context.enumerateInstanceLayerProperties();
  if (std::ranges::any_of(
          required_layers, [&layer_properties](auto const& required_layer) {
            return std::ranges::none_of(
                layer_properties, [required_layer](auto const& layer_property) {
                  return strcmp(layer_property.layerName, required_layer) == 0;
                });
          })) {
    throw std::runtime_error("One or more required layers are not supported!");
  }

  uint32_t glfw_extension_count = 0;
  auto glfw_extensions =
      glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  auto extension_properties =
      Context::Instance()->g_vk_context.enumerateInstanceExtensionProperties();
  for (uint32_t i = 0; i < glfw_extension_count; ++i) {
    if (std::ranges::none_of(
            extension_properties, [&glfw_extension = glfw_extensions[i]](
                                      const auto& extension_property) {
              return strcmp(extension_property.extensionName, glfw_extension) ==
                     0;
            })) {
      throw std::runtime_error("Required GLFW extension not supported: " +
                               std::string(glfw_extensions[i]));
    }
  }

  vk::InstanceCreateInfo create_info{
      .pApplicationInfo = &app_info,
      .enabledLayerCount = static_cast<uint32_t>(required_layers.size()),
      .ppEnabledLayerNames = required_layers.data(),
      .enabledExtensionCount = glfw_extension_count,
      .ppEnabledExtensionNames = glfw_extensions};

  Context::Instance()->g_vk_instance =
      vk::raii::Instance(Context::Instance()->g_vk_context, create_info);
}
