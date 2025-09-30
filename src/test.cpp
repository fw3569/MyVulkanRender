#include <iostream>
#include <cstdlib>
#include <functional>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <array>

#define VK_USE_PLATFOEM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
import vulkan_hpp;
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using std::string;

#ifndef SHADER_FILE_PATH
#define SHADER_FILE_PATH ""
#endif
#ifndef DATA_FILE_PATH
#define DATA_FILE_PATH ""
#endif

#ifndef NDEBUG
#define LOG(...)\
{std::cout<<__FILE__<<":"<<__LINE__<<" : ";\
std::vector<std::string>v{__VA_ARGS__};\
for(const std::string& msg:v){\
  std::cout<<msg;\
}\
std::cout<<std::endl;}
#else
#define LOG(...) ;
#endif

namespace {
  struct Vertex{
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 tex_coord;

    static vk::VertexInputBindingDescription GetBindingDescription(){
      return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }
    static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescription(){
      return {
        vk::VertexInputAttributeDescription{0,0,vk::Format::eR32G32B32Sfloat,   offsetof(Vertex,position)},
        vk::VertexInputAttributeDescription{1,0,vk::Format::eR32G32B32Sfloat,offsetof(Vertex,color)},
        vk::VertexInputAttributeDescription{2,0,vk::Format::eR32G32Sfloat,offsetof(Vertex,tex_coord)}
      };
    }
  };
  std::vector<Vertex> g_vertex_in{
    {{-0.5f,-0.5f, 0.0f}, {1.0f,0.0f,0.0f}, {0.0f,0.0f}},
    {{ 0.5f,-0.5f, 0.0f}, {0.0f,1.0f,0.0f}, {1.0f,0.0f}},
    {{ 0.5f, 0.5f, 0.0f}, {0.0f,0.0f,1.0f}, {1.0f,1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f,1.0f,1.0f}, {0.0f,1.0f}},
    
    {{-0.5f,-0.5f,-0.5f}, {1.0f,0.0f,0.0f}, {0.0f,0.0f}},
    {{ 0.5f,-0.5f,-0.5f}, {0.0f,1.0f,0.0f}, {1.0f,0.0f}},
    {{ 0.5f, 0.5f,-0.5f}, {0.0f,0.0f,1.0f}, {1.0f,1.0f}},
    {{-0.5f, 0.5f,-0.5f}, {1.0f,1.0f,1.0f}, {0.0f,1.0f}}
  };
  std::vector<uint16_t> g_index_in{
    0,1,2,
    2,3,0,
    4,5,6,
    6,7,4
  };

  struct UniformBufferObject{
    alignas(16) glm::mat4 modu;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
  };
} // namespace

namespace {
  constexpr uint32_t kWindowWeight = 800;
  constexpr uint32_t kWindowHeight = 600;
  GLFWwindow*g_window;
  std::atomic<bool> g_window_resized = false;
  vk::raii::Context g_vk_context;
  vk::raii::Instance g_vk_instance = nullptr;
  static const std::vector kRequiredDeviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
  };
  vk::raii::PhysicalDevice g_physical_device = nullptr;
  vk::raii::Device g_device = nullptr;
  vk::raii::Queue g_queue = nullptr;
  uint32_t g_queue_index = 0;
  vk::raii::SurfaceKHR g_surface = nullptr;
  vk::raii::SwapchainKHR g_swapchain = nullptr;
  vk::Format g_swapchain_image_format = vk::Format::eUndefined;
  vk::Extent2D g_swapchain_extent;
  vk::raii::PipelineLayout g_pipeline_layout = nullptr;
  vk::raii::Pipeline g_graphics_pipeline = nullptr;
  std::vector<vk::Image> g_swapchain_images;
  std::vector<vk::raii::ImageView> g_swapchain_image_views;
  vk::raii::Buffer g_vertex_buffer = nullptr;
  vk::raii::DeviceMemory g_vertex_buffer_memory = nullptr;
  vk::raii::Buffer g_index_buffer = nullptr;
  vk::raii::DeviceMemory g_index_buffer_memory = nullptr;
  std::vector<vk::raii::Buffer> g_ubo_buffer;
  std::vector<vk::raii::DeviceMemory> g_ubo_buffer_memory;
  std::vector<void*> g_ubo_buffer_maped;
  vk::raii::Buffer g_transfer_buffer = nullptr;
  vk::raii::DeviceMemory g_transfer_buffer_memory = nullptr;
  vk::raii::Image g_texture_image = nullptr;
  vk::raii::DeviceMemory g_texture_image_memory  = nullptr;
  vk::raii::ImageView g_texture_image_view = nullptr;
  vk::raii::Sampler g_texture_image_sampler = nullptr;
  vk::raii::Image g_depth_image = nullptr;
  vk::raii::DeviceMemory g_depth_image_memory  = nullptr;
  vk::raii::ImageView g_depth_image_view = nullptr;
  vk::Format g_depth_image_format = vk::Format::eUndefined;
  vk::raii::CommandPool g_command_pool = nullptr;
  std::vector<vk::raii::CommandBuffer> g_command_buffer;
  std::vector<vk::raii::Semaphore> g_present_complete_semaphore;
  std::vector<vk::raii::Semaphore> g_render_finished_semaphore;
  std::vector<vk::raii::Fence> g_draw_fence;
  vk::raii::DescriptorPool g_descriptor_pool = nullptr;
  vk::raii::DescriptorSetLayout g_descriptor_set_layout = nullptr;
  std::vector<vk::raii::DescriptorSet> g_descriptor_sets;
  const std::vector kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
  };
#ifdef NDEBUG
  constexpr bool kEnableValidationLayers = false;
#else
  constexpr bool kEnableValidationLayers = true;
#endif

  std::vector<char> ReadFile(const char* file_path){
    std::ifstream file(file_path, std::ios::ate|std::ios::binary);
    if(!file.is_open()){
      throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
  }
  void RecreateSwapchain();
  void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
  void InitWindow(){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    g_window = glfwCreateWindow(kWindowWeight, kWindowHeight, "Vulkan", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(g_window, FramebufferSizeCallback);
  }
  void CreateInstance();
  void CreateaSurface();
  void PickPhysicalDevice();
  void CreateLogicalDevice();
  void CreateSwapChain();
  void CreateImageViews();
  void CreateDepthResources();
  void CreateDescriptorSetLayout();
  void CreateGraphicsPipeline();
  void CreateCommandPool();
  void CreateTextureImage();
  void CreateTextureImageView();
  void CreateTextureSampler();
  void CreateDateBuffers();
  void CreateVertexBuffer();
  void CreateIndexBuffer();
  void CreateUboBuffer();
  void CreateDescriptorPool();
  void CreateDescriptorSets();
  void CreateCommandBuffer();
  void CreateSyncObjects();
  void InitVulkan(){
    CreateInstance();
    CreateaSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    CreateImageViews();
    CreateDepthResources();
    CreateDescriptorSetLayout();
    CreateGraphicsPipeline();
    CreateCommandPool();
    CreateTextureImage();
    CreateTextureImageView();
    CreateTextureSampler();
    CreateDateBuffers();
    CreateDescriptorPool();
    CreateDescriptorSets();
    CreateCommandBuffer();
    CreateSyncObjects();
  }
  void CreateInstance(){
    constexpr vk::ApplicationInfo app_info{
      .pApplicationName = "Test Triangle",
      .applicationVersion = VK_MAKE_VERSION(1,0,0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1,0,0),
      .apiVersion = vk::ApiVersion14
    };

    std::vector<char const*> required_layers;
    if (kEnableValidationLayers) {
      required_layers.assign(kValidationLayers.begin(), kValidationLayers.end());
    }
    auto layer_properties = g_vk_context.enumerateInstanceLayerProperties();
    if (std::ranges::any_of(required_layers, [&layer_properties](auto const& required_layer) {
        return std::ranges::none_of(layer_properties,
                                  [required_layer](auto const& layer_property)
                                  { return strcmp(layer_property.layerName, required_layer) == 0; });
    }))
    {
        throw std::runtime_error("One or more required layers are not supported!");
    }


    uint32_t glfw_extension_count = 0;
    auto glfw_extensions=glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    auto extension_properties = g_vk_context.enumerateInstanceExtensionProperties();
    for(uint32_t i=0;i<glfw_extension_count;++i){
      if(std::ranges::none_of(extension_properties,[&glfw_extension=glfw_extensions[i]](const auto& extension_property){return strcmp(extension_property.extensionName, glfw_extension)==0;})){
        throw std::runtime_error("Required GLFW extension not supported: " + std::string(glfw_extensions[i]));
      }
    }

    vk::InstanceCreateInfo create_info {
      .pApplicationInfo = &app_info,
      .enabledLayerCount = static_cast<uint32_t>(required_layers.size()),
      .ppEnabledLayerNames = required_layers.data(),
      .enabledExtensionCount = glfw_extension_count,
      .ppEnabledExtensionNames = glfw_extensions
    };

    g_vk_instance = vk::raii::Instance(g_vk_context, create_info);
  }
  void CreateaSurface(){
    VkSurfaceKHR surface;
    if(glfwCreateWindowSurface(*g_vk_instance,g_window,nullptr,&surface)==0){
      g_surface = vk::raii::SurfaceKHR(g_vk_instance, surface);
    }else{
      throw std::runtime_error("failed to create window surface!");
    }
  }
  void PickPhysicalDevice(){
    auto devices = g_vk_instance.enumeratePhysicalDevices();
    if(devices.empty()){
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    bool found = false;
    for(const auto&device:devices){
      if(device.getProperties().apiVersion < VK_API_VERSION_1_3){
        continue;
      }
      auto device_properties = device.getProperties();
      auto queue_families = device.getQueueFamilyProperties();
      uint32_t score = 0;
      if(device_properties.apiVersion<VK_API_VERSION_1_3||std::ranges::find_if(queue_families, [](const vk::QueueFamilyProperties& qfp){
        return (qfp.queueFlags&vk::QueueFlagBits::eGraphics)!=static_cast<vk::QueueFlags>(0);
      })==queue_families.end()){
        continue;
      }
      auto extensions = device.enumerateDeviceExtensionProperties();
      bool all_found = true;
      for(const auto&req_extension:kRequiredDeviceExtensions){
        all_found &=(std::ranges::find_if(extensions,[req_extension](const auto&extension){return strcmp(extension.extensionName,req_extension)==0;})!=extensions.end());
      }
      if(!all_found){
        continue;
      }
      auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
      bool supports_required_features =
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState&&
        features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy;
      if(found == false||(device_properties.deviceType==vk::PhysicalDeviceType::eDiscreteGpu&&g_physical_device.getProperties().deviceType!=vk::PhysicalDeviceType::eDiscreteGpu&&supports_required_features)){
        g_physical_device = device;
        found = true;
      }
      LOG("using physical device: ", device_properties.deviceName);
    }
    if (found == false) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }
  void CreateLogicalDevice(){
    std::vector<vk::QueueFamilyProperties> queue_family_properties = g_physical_device.getQueueFamilyProperties();
    uint32_t queue_index = 0;
    for(;queue_index<queue_family_properties.size();++queue_index){
      if((queue_family_properties[queue_index].queueFlags&static_cast<vk::QueueFlags>(VK_QUEUE_GRAPHICS_BIT))&&(g_physical_device.getSurfaceSupportKHR(queue_index, *g_surface))){
        break;
      }
    }
    if(queue_index==queue_family_properties.size()){
      throw std::runtime_error("Could not find a queue for graphics and present!");
    }
    float queue_priorities[1]={0.0};
    vk::DeviceQueueCreateInfo device_queue_create_info{
      .queueFamilyIndex = queue_index,
      .queueCount = 1,
      .pQueuePriorities = queue_priorities
    };
    vk::PhysicalDeviceFeatures devices_features;
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> feature_chain = {
      {.features = {.samplerAnisotropy = true}},
      {.synchronization2 = true, .dynamicRendering = true},
      {.extendedDynamicState = true}
    };
    vk::DeviceCreateInfo device_create_info{
      .pNext = feature_chain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &device_queue_create_info,
      .enabledExtensionCount = static_cast<uint32_t>(kRequiredDeviceExtensions.size()),
      .ppEnabledExtensionNames = kRequiredDeviceExtensions.data()
    };
    g_device = vk::raii::Device(g_physical_device, device_create_info);
    g_queue = vk::raii::Queue(g_device, queue_index, 0);
    g_queue_index = queue_index;
  }
  void CreateSwapChain(){
    auto surface_capabilities = g_physical_device.getSurfaceCapabilitiesKHR(g_surface);
    auto surface_formats = g_physical_device.getSurfaceFormatsKHR(*g_surface);
    auto surface_present_modes = g_physical_device.getSurfacePresentModesKHR(g_surface);
    vk::SurfaceFormatKHR select_format = surface_formats[0];
    for(const auto& format:surface_formats){
      if(format.format==vk::Format::eB8G8R8A8Srgb&&format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear){
        select_format = format;
        break;
      }
    }
    vk::PresentModeKHR select_present_mode = vk::PresentModeKHR::eFifo;
    for(const auto&mode:surface_present_modes){
      if(mode==vk::PresentModeKHR::eMailbox){
        select_present_mode=mode;
        break;
      }
    }
    vk::Extent2D select_extent;
    if(surface_capabilities.currentExtent.width ==
       (std::numeric_limits<uint32_t>::max)()){
      int width, height;
      glfwGetFramebufferSize(g_window,&width,&height);
      select_extent = {
        std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.width, surface_capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.height, surface_capabilities.maxImageExtent.height)
      };
    } else {
      select_extent = surface_capabilities.currentExtent;
    }
    auto min_image_count = (std::max)(3u, surface_capabilities.minImageCount+1u);
    if(surface_capabilities.maxImageCount>0&&
       min_image_count>surface_capabilities.maxImageCount){
      min_image_count= surface_capabilities.maxImageCount;
    }
    vk::SwapchainCreateInfoKHR swapchain_create_info{
      .flags = vk::SwapchainCreateFlagsKHR(),
      .surface = g_surface,
      .minImageCount = min_image_count,
      .imageFormat = select_format.format,
      .imageColorSpace = select_format.colorSpace,
      .imageExtent = select_extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .preTransform = surface_capabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = select_present_mode,
      .clipped = true,
      .oldSwapchain = g_swapchain
    };
    g_swapchain = vk::raii::SwapchainKHR(g_device,swapchain_create_info);
    g_swapchain_images = g_swapchain.getImages();
    g_swapchain_image_format = select_format.format;
    g_swapchain_extent=select_extent;
  }
  vk::raii::ImageView CreateImageView(const vk::Image&image, vk::Format format, vk::ImageAspectFlagBits aspect){
    vk::ImageViewCreateInfo image_view_create_info{
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .components = {
        .r = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
        .g = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
        .b = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
        .a = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
      },
      .subresourceRange = {
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    return vk::raii::ImageView(g_device, image_view_create_info);
  }
  void CreateImageViews(){
    g_swapchain_image_views.clear();
    for(auto image:g_swapchain_images){
      g_swapchain_image_views.emplace_back(
        CreateImageView(image,g_swapchain_image_format,vk::ImageAspectFlagBits::eColor));
    }
  }
  vk::raii::ShaderModule CreateShaderModule(const char*shader_file_path){
    LOG(string("SHADER_FILE_PATH: ")+shader_file_path);
    std::vector<char> shader_code = ReadFile(shader_file_path);
    vk::ShaderModuleCreateInfo create_info{
      .codeSize = shader_code.size()*sizeof(char),
      .pCode = reinterpret_cast<uint32_t*>(shader_code.data())
    };
    return vk::raii::ShaderModule{g_device, create_info};
  }
  void CreateDescriptorSetLayout(){
    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings{
      {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      }
    };
    vk::DescriptorSetLayoutCreateInfo set_layout_info{
      .flags = {},
      .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
      .pBindings = layout_bindings.data()
    };
    g_descriptor_set_layout = vk::raii::DescriptorSetLayout(g_device, set_layout_info);
  }
  void CreateGraphicsPipeline(){
    vk::raii::ShaderModule shader_module = CreateShaderModule(SHADER_FILE_PATH);
    vk::PipelineShaderStageCreateInfo pipeline_shader_stage_create_info[2] = {
      {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shader_module,
        .pName = "vertMain",
        .pSpecializationInfo = nullptr
      },
      {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shader_module,
        .pName = "fragMain",
        .pSpecializationInfo = nullptr
      }
    };
    std::vector dynamic_states = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dyanmic_state_create_info = {
      .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
      .pDynamicStates = dynamic_states.data()
    };
    auto binding_desc = Vertex::GetBindingDescription();
    auto attribute_desc = Vertex::GetAttributeDescription();
    vk::PipelineVertexInputStateCreateInfo vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding_desc,
      .vertexAttributeDescriptionCount = attribute_desc.size(),
      .pVertexAttributeDescriptions = attribute_desc.data(),
    };
    vk::PipelineInputAssemblyStateCreateInfo input_assembly_info{
      .topology = vk::PrimitiveTopology::eTriangleList
    };
    // vk::Viewport viewport{
    //   .x = 0.0f,
    //   .y = 0.0f,
    //   .width = static_cast<float>(g_swapchain_extent.width),
    //   .height = static_cast<float>(g_swapchain_extent.height),
    //   .minDepth = 0.0f,
    //   .maxDepth = 1.0f
    // };
    // vk::Rect2D scissor{vk::Offset2D{0,0},g_swapchain_extent};
    vk::PipelineViewportStateCreateInfo viewport_state_info{
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr
    };
    vk::PipelineRasterizationStateCreateInfo rasterization_create_info{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasConstantFactor = 1.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth =1.0f 
    };
    vk::PipelineMultisampleStateCreateInfo multisample_create_info{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False,
    };
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_info{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False
    };
    vk::PipelineColorBlendAttachmentState color_blend_attachment{
      .blendEnable = vk::True,
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eDstAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOne,
      .dstAlphaBlendFactor = vk::BlendFactor::eZero,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
      // .blendConstants = ,
    };
    vk::PipelineLayoutCreateInfo pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*g_descriptor_set_layout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };
    g_pipeline_layout = vk::raii::PipelineLayout(g_device, pipeline_layout_info);

    vk::PipelineRenderingCreateInfo pipeline_rending_info{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &g_swapchain_image_format,
      .depthAttachmentFormat = g_depth_image_format,
      // .stencilAttachmentFormat = ,
    };
    vk::GraphicsPipelineCreateInfo pipeline_info{
      .pNext = &pipeline_rending_info,
      .stageCount = 2,
      .pStages = pipeline_shader_stage_create_info,
      .pVertexInputState = &vertex_input_info,
      .pInputAssemblyState = &input_assembly_info,
      // .pTessellationState = ,
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = &depth_stencil_info,
      .pColorBlendState = &color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = g_pipeline_layout,
      .renderPass = nullptr
      // .subpass = ,
      // .basePipelineHandle = ,
      // .basePipelineIndex =
    };
    g_graphics_pipeline = vk::raii::Pipeline(g_device, nullptr, pipeline_info);
  }
  uint32_t FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties){
    vk::PhysicalDeviceMemoryProperties memory_properties = g_physical_device.getMemoryProperties();
    for(uint32_t i=0;i<memory_properties.memoryTypeCount;++i){
      if((type_filter&(1<<i))&&(memory_properties.memoryTypes[i].propertyFlags&properties)==properties){
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }
  void CreateBuffer(uint32_t size,vk::BufferUsageFlags usage,vk::SharingMode sharing_mode,vk::MemoryPropertyFlags properties,vk::raii::Buffer&buffer, vk::raii::DeviceMemory&memory){
    vk::BufferCreateInfo vertex_buffer_info{
      .flags = {},
      .size = size,
      .usage = usage,
      .sharingMode = sharing_mode
    };
    buffer = vk::raii::Buffer(g_device,vertex_buffer_info);
    vk::MemoryRequirements memory_requirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo memory_alloc_info{
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex = FindMemoryType(
        memory_requirements.memoryTypeBits, properties)
    };
    memory = vk::raii::DeviceMemory(g_device, memory_alloc_info);
    buffer.bindMemory(*memory, 0);
  }
  vk::raii::CommandBuffer BeginOneTimeCommandBuffer(){
    vk::raii::CommandBuffer command_buffer = nullptr;
    vk::CommandBufferAllocateInfo alloc_info{
      .commandPool = g_command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1
    };
    command_buffer = std::move(vk::raii::CommandBuffers(g_device, alloc_info).front());
    command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return command_buffer;
  }
  void EndOneTimeCommandBuffer(vk::raii::CommandBuffer&command_buffer){
    command_buffer.end();
    g_queue.submit(vk::SubmitInfo{.commandBufferCount = 1,.pCommandBuffers = &*command_buffer}, nullptr);
    g_queue.waitIdle();
  }
  void CopyBuffer(vk::raii::Buffer& src_buffer, vk::raii::Buffer& dst_buffer, uint32_t size){
    vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
    command_buffer.copyBuffer(src_buffer, dst_buffer, vk::BufferCopy{0, 0, size});
    EndOneTimeCommandBuffer(command_buffer);
  }
  void CreateVertexBuffer(){
    uint32_t size = sizeof(g_vertex_in[0])*g_vertex_in.size();
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, g_transfer_buffer, g_transfer_buffer_memory);
    void* data = g_transfer_buffer_memory.mapMemory(0, size);
    memcpy(data, g_vertex_in.data(), size);
    g_transfer_buffer_memory.unmapMemory();
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eDeviceLocal, g_vertex_buffer, g_vertex_buffer_memory);
    CopyBuffer(g_transfer_buffer,g_vertex_buffer, size);
  }
  void CreateIndexBuffer(){
    uint32_t size = sizeof(g_index_in[0])*g_index_in.size();
    void* data = g_transfer_buffer_memory.mapMemory(0, size);
    memcpy(data, g_index_in.data(), size);
    g_transfer_buffer_memory.unmapMemory();
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eDeviceLocal, g_index_buffer, g_index_buffer_memory);
    CopyBuffer(g_transfer_buffer,g_index_buffer, size);
  }
  void CreateUboBuffer(){
    g_ubo_buffer.clear();
    g_ubo_buffer_memory.clear();
    g_ubo_buffer_maped.clear();

    for(uint32_t i=0;i<g_swapchain_images.size();++i){
      uint32_t size = sizeof(UniformBufferObject);
      vk::raii::Buffer buffer = nullptr;
      vk::raii::DeviceMemory memory = nullptr;
      CreateBuffer(
        size, vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, memory);
        void* data = memory.mapMemory(0, size);
        g_ubo_buffer.emplace_back(std::move(buffer));
        g_ubo_buffer_memory.emplace_back(std::move(memory));
        g_ubo_buffer_maped.emplace_back(data);
    }
  }
  void CreateDateBuffers(){
    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateUboBuffer();
  }
  void CreateDescriptorPool(){
    uint32_t frame_size = g_swapchain_images.size();
    std::vector<vk::DescriptorPoolSize> pool_sizes{
      {
        .type=vk::DescriptorType::eUniformBuffer,
        .descriptorCount=frame_size
      },
      {
        .type=vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount=frame_size
      }
    };
    vk::DescriptorPoolCreateInfo descriptor_pool_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = static_cast<uint32_t>(g_swapchain_images.size()*pool_sizes.size()),
      .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
    };
    g_descriptor_pool = vk::raii::DescriptorPool(g_device,descriptor_pool_info);
  }
  void CreateDescriptorSets(){
    std::vector<vk::DescriptorSetLayout> layouts(g_swapchain_images.size(), *g_descriptor_set_layout);
    vk::DescriptorSetAllocateInfo alloc_info{
      .descriptorPool = *g_descriptor_pool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()
    };
    g_descriptor_sets = g_device.allocateDescriptorSets(alloc_info);
    for(uint32_t i=0;i<g_swapchain_images.size();++i){
      vk::DescriptorBufferInfo buffer_info{.buffer = g_ubo_buffer[i], .offset = 0, .range = sizeof(UniformBufferObject)};
      vk::DescriptorImageInfo image_info{
        .sampler = *g_texture_image_sampler,
        .imageView = *g_texture_image_view, 
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      std::vector<vk::WriteDescriptorSet> descriptor_write{
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo = &buffer_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 1,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .pImageInfo = &image_info
        }
      };
      g_device.updateDescriptorSets(descriptor_write, {});
    }
  }
  void TransitionImageLayout(vk::raii::Image& image, vk::ImageLayout  old_layout, vk::ImageLayout new_layout){
    vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
    vk::ImageMemoryBarrier barrier{
      .srcAccessMask = {},
      .dstAccessMask = {},
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    vk::PipelineStageFlags src_stage;
    vk::PipelineStageFlags dst_stage;
    if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
      dst_stage = vk::PipelineStageFlagBits::eTransfer;
    } else if (old_layout == vk::ImageLayout::eTransferDstOptimal && new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      src_stage = vk::PipelineStageFlagBits::eTransfer;
      dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }
    command_buffer.pipelineBarrier(src_stage,dst_stage,{},{},nullptr,barrier);
    EndOneTimeCommandBuffer(command_buffer);
  }
  void CreateImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& memory){
    vk::ImageCreateInfo image_info{
      .flags = {},
      .imageType = vk::ImageType::e2D,
      .format = format,
      .extent = {width,height,1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &g_queue_index,
      .initialLayout = vk::ImageLayout::eUndefined,
    };
    image = vk::raii::Image(g_device, image_info);
    vk::MemoryRequirements memory_requirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo memory_alloc_info{
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex = FindMemoryType(
        memory_requirements.memoryTypeBits, properties)
    };
    memory = vk::raii::DeviceMemory(g_device, memory_alloc_info);
    image.bindMemory(*memory, 0);
  }
  void CopyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height){
    vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
    vk::BufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {width, height, 1}
    };
    command_buffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
    EndOneTimeCommandBuffer(command_buffer);
  }
  void CreateTextureImage(){
    int tex_width,tex_height,tex_channels;
    stbi_uc*pixels = stbi_load(DATA_FILE_PATH"/test_image.png",&tex_width,&tex_height,&tex_channels,STBI_rgb_alpha);
    vk::DeviceSize image_size = tex_width*tex_height*4;
    if(!pixels){
      throw std::runtime_error("failed to load texture image!");
    }
    vk::raii::Buffer buffer = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    CreateBuffer(
      image_size,vk::BufferUsageFlagBits::eTransferSrc,
      vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
      buffer,memory
    );
    void*data=memory.mapMemory(0,image_size);
    memcpy(data,pixels,image_size);
    memory.unmapMemory();
    // stbi need free
    stbi_image_free(pixels);
    CreateImage(
      tex_width,tex_height,vk::Format::eR8G8B8A8Srgb,vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_texture_image,g_texture_image_memory
    );
    TransitionImageLayout(g_texture_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    CopyBufferToImage(buffer, g_texture_image, tex_width, tex_height);
    TransitionImageLayout(g_texture_image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
  }
  void CreateTextureImageView(){
    g_texture_image_view = CreateImageView(*g_texture_image,vk::Format::eR8G8B8A8Srgb,vk::ImageAspectFlagBits::eColor);
  }
  void CreateTextureSampler(){
    vk::PhysicalDeviceProperties properties = g_physical_device.getProperties();
    vk::SamplerCreateInfo sampler_info{
      .flags = {},
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0,
      .anisotropyEnable = vk::True,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::True,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0,
      .maxLod = 0,
      .borderColor = vk::BorderColor::eIntTransparentBlack,
      .unnormalizedCoordinates = vk::False,
    };
    g_texture_image_sampler = vk::raii::Sampler(g_device, sampler_info);
  }
  vk::Format FindSupportFormat(const std::vector<vk::Format>&candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags flags){
    for(const vk::Format&format:candidates){
      vk::FormatProperties properties = g_physical_device.getFormatProperties(format);
      if(tiling==vk::ImageTiling::eLinear&&(properties.linearTilingFeatures&flags)==flags){
        return format;
      }
      if(tiling==vk::ImageTiling::eOptimal&&(properties.optimalTilingFeatures&flags)==flags){
        return format;
      }
    }
    throw std::runtime_error("failed to find supported format!");
  }
  vk::Format FindSupportDepthFormat(){
    return FindSupportFormat(
      {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
      vk::ImageTiling::eOptimal,
      vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
  }
  bool HasStencilCompoent(vk::Format format){
    return format==vk::Format::eD32SfloatS8Uint||format == vk::Format::eD24UnormS8Uint;
  }
  void CreateDepthResources(){
    g_depth_image_format = FindSupportDepthFormat();
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,g_depth_image_format,vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_depth_image,g_depth_image_memory
    );
    g_depth_image_view = CreateImageView(*g_depth_image,g_depth_image_format,vk::ImageAspectFlagBits::eDepth);
  }
  void CreateCommandPool(){
    vk::CommandPoolCreateInfo pool_info{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = g_queue_index
    };
    g_command_pool = vk::raii::CommandPool(g_device, pool_info);
  }
  void CreateCommandBuffer(){
    vk::CommandBufferAllocateInfo alloc_info{
      .commandPool = g_command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = static_cast<uint32_t>(g_swapchain_images.size())
    };
    g_command_buffer = vk::raii::CommandBuffers(g_device, alloc_info);
  }
  void TransformImageLayout(
    uint32_t image_index,
    uint32_t frame_index,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask
  ){
    vk::ImageMemoryBarrier2 barrier={
      .srcStageMask = src_stage_mask,
      .srcAccessMask = src_access_mask,
      .dstStageMask = dst_stage_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = g_swapchain_images[image_index],
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    vk::DependencyInfo dependency_info{
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
    };
    g_command_buffer[frame_index].pipelineBarrier2(dependency_info);
  }
  void RecordCommandBuffer(
    uint32_t image_index,
    uint32_t frame_index,
    vk::Viewport viewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(g_swapchain_extent.width),static_cast<float>(g_swapchain_extent.height), 0.0f, 1.0f),
    vk::Rect2D scissor = vk::Rect2D(vk::Offset2D(0,0), g_swapchain_extent)){
    g_command_buffer[frame_index].begin({});
    TransformImageLayout(
      image_index,
      frame_index,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal,
      {},
      vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::PipelineStageFlagBits2::eTopOfPipe,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput
    );
    vk::ImageMemoryBarrier2 depth_barrier={
      .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
      .srcAccessMask = {},
      .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests|vk::PipelineStageFlagBits2::eLateFragmentTests,
      .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead|vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
      .oldLayout = vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = g_depth_image,
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eDepth,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    vk::DependencyInfo dependency_info{
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &depth_barrier,
    };
    g_command_buffer[frame_index].pipelineBarrier2(dependency_info);
    vk::RenderingAttachmentInfo attachment_info{
      .imageView = g_swapchain_image_views[image_index],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}
    };
    vk::RenderingAttachmentInfo depth_info{
      .imageView = g_depth_image_view,
      .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .clearValue = vk::ClearDepthStencilValue{1.0f, 0}
    };
    vk::RenderingInfo rendering_info{
      .renderArea = {.offset = {0, 0}, .extent = g_swapchain_extent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachment_info,
      .pDepthAttachment = &depth_info
    };
    g_command_buffer[frame_index].beginRendering(rendering_info);
    g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_graphics_pipeline);
    g_command_buffer[frame_index].bindVertexBuffers(0, *g_vertex_buffer, {0});
    g_command_buffer[frame_index].bindIndexBuffer(*g_index_buffer, 0, vk::IndexType::eUint16);
    g_command_buffer[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_pipeline_layout,0,*g_descriptor_sets[image_index],nullptr);
    g_command_buffer[frame_index].setViewport(0, viewport);
    g_command_buffer[frame_index].setScissor(0, scissor);
    g_command_buffer[frame_index].drawIndexed(g_index_in.size(),1,0,0,0);
    g_command_buffer[frame_index].endRendering();
    TransformImageLayout(
      image_index,
      frame_index,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::ePresentSrcKHR,
      vk::AccessFlagBits2::eColorAttachmentWrite,
      {},
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::PipelineStageFlagBits2::eBottomOfPipe
    );
    g_command_buffer[frame_index].end();
  }
  void CreateSyncObjects(){
    uint32_t image_count = g_swapchain_images.size();
    for(uint32_t i=0;i<image_count;++i){
      g_present_complete_semaphore.emplace_back(vk::raii::Semaphore(g_device, vk::SemaphoreCreateInfo()));
      g_render_finished_semaphore.emplace_back(vk::raii::Semaphore(g_device, vk::SemaphoreCreateInfo()));
      g_draw_fence.emplace_back(vk::raii::Fence(g_device, {.flags = vk::FenceCreateFlagBits::eSignaled}));
    }
  }
  void RecreateSwapchain(){
    int width = 0, height = 0;
    glfwGetFramebufferSize(g_window,&width,&height);
    while(width == 0 && height == 0) {
      glfwWaitEvents();
      glfwGetFramebufferSize(g_window,&width,&height);
    }

    g_device.waitIdle();
    
    g_window_resized = false;
    CreateSwapChain();
    CreateImageViews();
    CreateDepthResources();
  }
  void FramebufferSizeCallback(GLFWwindow* /* window */, int /* width */, int /* height */){
    g_window_resized = true;
  }
} // namespace

class TriangleRhi{
 public:
  void Run(){
    Init();
    Work();
    Cleanup();
  }
 private:
  void Init(){
    m_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()/1000.0;
  }
  void Work(){
    while(!glfwWindowShouldClose(g_window)){
      constexpr uint32_t fps = 30;
      constexpr double draw_internal = 1.0/fps;
      static double next_draw_time = 0.0;
      m_current_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()/1000.0;
      glfwPollEvents();
      if(next_draw_time < m_current_time && Tick()){
        next_draw_time = m_current_time+draw_internal;
      }
      // std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    g_device.waitIdle();
  }
  bool Tick(){
    return UpdateDate()&&DrawFrame();
  }
  bool UpdateDate(){
    UniformBufferObject ubo;
    ubo.modu = glm::rotate<float>(glm::mat4(1.0f),(m_current_time-m_start_time)*glm::radians(90.0f),glm::vec3(0.0f,0.0f,1.0f));
    // https://learnopengl.com/Getting-started/Coordinate-Systems
    // https://learnopengl.com/Getting-started/Camera
    ubo.view = glm::lookAt(glm::vec3(0.0f,-1.0f,1.0f),glm::vec3(0.0f,0.0f,0.0f),glm::vec3(0.0f,1.0f,0.0f));
    ubo.proj = glm::perspective<float>(glm::radians(90.0f),static_cast<float>(g_swapchain_extent.width) / static_cast<float>(g_swapchain_extent.height), 0.1f, 10.0f);
    // flip input and output y axis
    ubo.modu=glm::scale(ubo.modu,glm::vec3{1.0f,-1.0f,1.0f});
    ubo.proj[1][1]*=-1;
    memcpy(g_ubo_buffer_maped[m_frame_index], &ubo, sizeof(ubo));
    return true;
  }
  bool DrawFrame(){
    static uint32_t kFrameCount = 0;
    if(kFrameCount==0){
      kFrameCount = g_swapchain_images.size();
    }
    auto [result, image_index] = g_swapchain.acquireNextImage(UINT64_MAX, *g_present_complete_semaphore[m_frame_index], nullptr);
    bool window_resized = false;
    if(result!=vk::Result::eSuccess){
      LOG("acquireNextImage: "+to_string(result));
      if(result==vk::Result::eErrorOutOfDateKHR){
        RecreateSwapchain();
        return false;
      } else if(result==vk::Result::eSuboptimalKHR) {
        window_resized = true;
      }else{
        return false;
      }
    }
    RecordCommandBuffer(image_index, m_frame_index);
    vk::PipelineStageFlags wait_dst_stage_mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    vk::SubmitInfo submit_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*g_present_complete_semaphore[m_frame_index],
      .pWaitDstStageMask = &wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &*g_command_buffer[m_frame_index],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &*g_render_finished_semaphore[m_frame_index],
    };
    g_device.resetFences(*g_draw_fence[m_frame_index]);
    g_queue.submit(submit_info, *g_draw_fence[m_frame_index]);
    while(vk::Result::eTimeout == g_device.waitForFences(*g_draw_fence[m_frame_index], vk::True, UINT64_MAX));
    const vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*g_render_finished_semaphore[m_frame_index],
      .swapchainCount = 1,
      .pSwapchains = &*g_swapchain,
      .pImageIndices = &image_index
    };
    result = g_queue.presentKHR(present_info);
    m_frame_index = (m_frame_index+1)%kFrameCount;
    if(result!=vk::Result::eSuccess){
      LOG("presentKHR: "+to_string(result));
      if(result==vk::Result::eErrorOutOfDateKHR||
         result==vk::Result::eSuboptimalKHR){
        window_resized = true;
      }
    }
    if(window_resized||g_window_resized){
      RecreateSwapchain();
    }
    return result==vk::Result::eSuccess||result==vk::Result::eSuboptimalKHR;
  }
  void Cleanup(){
    glfwDestroyWindow(g_window);
    glfwTerminate();
  }
  double m_start_time = 0.0;
  double m_current_time = 0.0;
  uint32_t m_frame_index = 0;
};

int main (int argc, char** argv){
  std::cout<<"start"<<std::endl;
  TriangleRhi worker;
  try{
    InitWindow();
    InitVulkan();
    worker.Run();
  }catch(const std::exception& e){
    std::cerr<<e.what()<<std::endl;
    return EXIT_FAILURE;
  }
  std::cout<<"stop"<<std::endl;
  return EXIT_SUCCESS;
}
