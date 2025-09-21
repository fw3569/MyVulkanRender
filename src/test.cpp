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

#define VK_USE_PLATFOEM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
import vulkan_hpp;
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

using std::string;

#ifndef SHADER_FILE_PATH
#define SHADER_FILE_PATH ""
#endif

#ifndef NDEBUG
#define LOG(...)\
std::cout<<__FILE__<<":"<<__LINE__<<" : ";\
std::vector<std::string>v{__VA_ARGS__};\
for(const std::string& msg:v){\
  std::cout<<msg;\
}\
std::cout<<std::endl;
#else
#define LOG(...) ;
#endif

namespace {
  constexpr uint32_t kWindowWeight = 800;
  constexpr uint32_t kWindowHeight = 600;
  GLFWwindow*g_window;
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
  vk::raii::PipelineLayout g_pipeline_layerout = nullptr;
  vk::raii::Pipeline g_graphics_pipeline = nullptr;
  std::vector<vk::Image> g_swapchain_images;
  std::vector<vk::raii::ImageView> g_swapchain_image_views;
  vk::raii::CommandPool g_command_pool = nullptr;
  vk::raii::CommandBuffer g_command_buffer = nullptr;
  std::vector<vk::raii::Semaphore> g_present_complete_semaphore;
  std::vector<vk::raii::Semaphore> g_render_finished_semaphore;
  vk::raii::Fence g_draw_fence = nullptr;
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
  void InitWindow(){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    g_window = glfwCreateWindow(kWindowWeight, kWindowHeight, "Vulkan", nullptr, nullptr);
  }
  void CreateInstance();
  void CreateaSurface();
  void PickPhysicalDevice();
  void CreateLogicalDevice();
  void CreateSwapChain();
  void CreateImageViews();
  void CreateGraphicsPipeline();
  void CreateCommandPool();
  void CreateCommandBuffer();
  void CreateSyncObjects();
  void InitVulkan(){
    CreateInstance();
    CreateaSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    CreateImageViews();
    CreateGraphicsPipeline();
    CreateCommandPool();
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
      bool supports_required_features = features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                      features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;
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
      {},
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
      .oldSwapchain = nullptr
    };
    g_swapchain = vk::raii::SwapchainKHR(g_device,swapchain_create_info);
    g_swapchain_images = g_swapchain.getImages();
    g_swapchain_image_format = select_format.format;
    g_swapchain_extent=select_extent;
  }
  void CreateImageViews(){
    g_swapchain_image_views.clear();
    vk::ImageViewCreateInfo image_view_create_info{
      .viewType = vk::ImageViewType::e2D,
      .format = g_swapchain_image_format,
      .components = {
        .r = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
        .g = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
        .b = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
        .a = static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_IDENTITY),
      },
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    for(const auto&image:g_swapchain_images){
      image_view_create_info.image = image;
      g_swapchain_image_views.emplace_back(g_device,image_view_create_info);
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
    vk::PipelineVertexInputStateCreateInfo vertex_input_info;
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
    //????
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
    // VkPipelineDepthStencilStateCreateInfo
    // z order
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
      .setLayoutCount = 0,
      .pSetLayouts = nullptr,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };
    g_pipeline_layerout = vk::raii::PipelineLayout(g_device, pipeline_layout_info);

    vk::PipelineRenderingCreateInfo pipeline_rending_info{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &g_swapchain_image_format,
      // .depthAttachmentFormat = ,
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
      // .pDepthStencilState = ,
      .pColorBlendState = &color_blend_info,
      .pDynamicState = &dyanmic_state_create_info,
      .layout = g_pipeline_layerout,
      .renderPass = nullptr
      // .subpass = ,
      // .basePipelineHandle = ,
      // .basePipelineIndex =
    };
    g_graphics_pipeline = vk::raii::Pipeline(g_device, nullptr, pipeline_info);
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
      .commandBufferCount = 1
    };
    g_command_buffer = std::move(vk::raii::CommandBuffers(g_device, alloc_info).front());
  }
  void TransformImageLayout(
    uint32_t image_index,
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
    g_command_buffer.pipelineBarrier2(dependency_info);
  }
  void RecordCommandBuffer(
    uint32_t image_index,
    uint32_t vertex_count,
    uint32_t instance_count = 1,
    uint32_t first_vertex = 0,
    uint32_t first_instance = 0,
    vk::Viewport viewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(g_swapchain_extent.width),static_cast<float>(g_swapchain_extent.height), 0.0f, 1.0f),
    vk::Rect2D scissor = vk::Rect2D(vk::Offset2D(0,0), g_swapchain_extent)){
    g_command_buffer.begin({});
    // ???
    TransformImageLayout(
      image_index,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal,
      {},
      vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::PipelineStageFlagBits2::eTopOfPipe,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput
    );
    vk::RenderingAttachmentInfo attachment_info{
      .imageView = g_swapchain_image_views[image_index],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}
    };
    vk::RenderingInfo rendering_info{
      .renderArea = {.offset = {0, 0}, .extent = g_swapchain_extent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachment_info,
    };
    g_command_buffer.beginRendering(rendering_info);
    g_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, g_graphics_pipeline);
    g_command_buffer.setViewport(0, viewport);
    g_command_buffer.setScissor(0, scissor);
    g_command_buffer.draw(vertex_count,
                          instance_count,
                          first_vertex,
                          first_instance);
    g_command_buffer.endRendering();
    TransformImageLayout(
      image_index,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::ePresentSrcKHR,
      vk::AccessFlagBits2::eColorAttachmentWrite,
      {},
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::PipelineStageFlagBits2::eBottomOfPipe
    );
    g_command_buffer.end();
  }
  void CreateSyncObjects(){
    uint32_t image_count = g_swapchain_images.size();
    for(uint32_t i=0;i<image_count;++i){
      g_present_complete_semaphore.emplace_back(vk::raii::Semaphore(g_device, vk::SemaphoreCreateInfo()));
      g_render_finished_semaphore.emplace_back(vk::raii::Semaphore(g_device, vk::SemaphoreCreateInfo()));
    }
    g_draw_fence = vk::raii::Fence(g_device, {.flags = vk::FenceCreateFlagBits::eSignaled});
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
    ;
  }
  void Work(){
    while(!glfwWindowShouldClose(g_window)){
      constexpr uint32_t fps = 30;
      constexpr double draw_internal = 1.0/fps;
      static double next_draw_time = 0.0;
      double now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()/1000.0;
      glfwPollEvents();
      if(next_draw_time < now && DrawFrame()){
        next_draw_time = now+draw_internal;
      }
      // std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    g_device.waitIdle();
  }
  bool DrawFrame(){
    static uint32_t frame_index = 0;
    static uint32_t kFrameCount = 0;
    if(kFrameCount==0){
      kFrameCount = g_swapchain_images.size();
    }
    auto [result, image_index] = g_swapchain.acquireNextImage(UINT64_MAX, *g_present_complete_semaphore[frame_index], nullptr);
    if(result!=vk::Result::eSuccess){
      LOG(to_string(result));
      return false;
    }
    RecordCommandBuffer(image_index, 3);
    g_device.resetFences(*g_draw_fence);
    vk::PipelineStageFlags wait_dst_stage_mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    vk::SubmitInfo submit_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*g_present_complete_semaphore[frame_index],
      .pWaitDstStageMask = &wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &*g_command_buffer,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &*g_render_finished_semaphore[frame_index],
    };
    g_queue.submit(submit_info, *g_draw_fence);
    while(vk::Result::eTimeout == g_device.waitForFences(*g_draw_fence, vk::True, UINT64_MAX));
    const vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*g_render_finished_semaphore[frame_index],
      .swapchainCount = 1,
      .pSwapchains = &*g_swapchain,
      .pImageIndices = &image_index
    };
    result = g_queue.presentKHR(present_info);
    frame_index = (frame_index+1)%kFrameCount;
    if(result!=vk::Result::eSuccess){
      LOG(to_string(result));
      return false;
    }
    return true;
  }
  void Cleanup(){
    glfwDestroyWindow(g_window);
    glfwTerminate();
  }
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
