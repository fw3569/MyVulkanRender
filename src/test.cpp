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
#include <random>

#define VK_USE_PLATFOEM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
import vulkan_hpp;
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

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
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec4 roughness_f0;
    alignas(16) glm::vec3 normal;
    float metallic;
    glm::vec2 tex_coord;

    static vk::VertexInputBindingDescription GetBindingDescription(){
      return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }
    static std::array<vk::VertexInputAttributeDescription, 5> GetAttributeDescription(){
      return {
        vk::VertexInputAttributeDescription{0,0,vk::Format::eR32G32B32Sfloat,   offsetof(Vertex,position)},
        vk::VertexInputAttributeDescription{1,0,vk::Format::eR32G32B32A32Sfloat,offsetof(Vertex,roughness_f0)},
        vk::VertexInputAttributeDescription{2,0,vk::Format::eR32G32B32Sfloat,offsetof(Vertex,normal)},
        vk::VertexInputAttributeDescription{3,0,vk::Format::eR32Sfloat,offsetof(Vertex,metallic)},
        vk::VertexInputAttributeDescription{4,0,vk::Format::eR32G32Sfloat,offsetof(Vertex,tex_coord)}
      };
    }
    bool operator==(const Vertex& other) const {
      return position == other.position && roughness_f0 == other.roughness_f0 && normal == other.normal && metallic == other.metallic && tex_coord == other.tex_coord;
    }
  };

  struct UniformBufferObject{
    alignas(16) glm::mat4 modu;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    struct Light {
      alignas(16) glm::vec3 pos;
      alignas(16) glm::vec3 intensities;
    };
    alignas(16) Light light;
    alignas(16) glm::vec3 camera_pos;
    alignas(16) glm::mat4 light_view;
    alignas(16) glm::mat4 light_proj;
    alignas(8) glm::vec2 shadowmap_resolution;
    alignas(8) glm::vec2 shadowmap_scale;
  };
  // push_constants size should be multiple of 4
  struct LightingPushConstants{
    int32_t enable_ssao;
  };
  
  struct BloomPushConstants{
    uint32_t bloom_mip_level;
    float bloom_factor;
  };

  struct Particle{
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 v;
    alignas(16) glm::vec3 color;
    static vk::VertexInputBindingDescription GetBindingDescription(){
      return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
    }
    static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescription(){
      return {
        vk::VertexInputAttributeDescription{0,0,vk::Format::eR32G32B32Sfloat,   offsetof(Particle,pos)},
        vk::VertexInputAttributeDescription{1,0,vk::Format::eR32G32B32Sfloat,offsetof(Particle,v)},
        vk::VertexInputAttributeDescription{2,0,vk::Format::eR32G32B32Sfloat,offsetof(Particle,color)}
      };
    }
  };
  struct ParticleUbo{
    float delta_time = 0;
  };
} // namespace

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^
                   (hash<glm::vec3>()(vertex.roughness_f0) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.tex_coord) << 1);
        }
    };
}
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
    vk::KHRCreateRenderpass2ExtensionName,
    vk::KHRDynamicRenderingLocalReadExtensionName
  };
  vk::raii::PhysicalDevice g_physical_device = nullptr;
  vk::raii::Device g_device = nullptr;
  vk::raii::Queue g_queue = nullptr;
  uint32_t g_queue_index = 0;
  vk::raii::SurfaceKHR g_surface = nullptr;
  vk::raii::SwapchainKHR g_swapchain = nullptr;
  vk::Format g_swapchain_image_format = vk::Format::eUndefined;
  vk::Format g_gbuffer_format = vk::Format::eR32G32B32A32Sfloat;
  vk::Extent2D g_swapchain_extent;
  vk::raii::PipelineLayout g_pipeline_layout = nullptr;
  vk::raii::Pipeline g_graphics_pipeline = nullptr;
  vk::raii::PipelineLayout g_lighting_pipeline_layout = nullptr;
  vk::raii::Pipeline g_lighting_pipeline = nullptr;
  uint32_t g_frame_in_flight = 2;
  std::vector<vk::Image> g_swapchain_images;
  std::vector<vk::raii::ImageView> g_swapchain_image_views;
  vk::raii::Image g_color_image = nullptr;
  vk::raii::DeviceMemory g_color_image_memory = nullptr;
  vk::raii::ImageView g_color_image_view = nullptr;
  vk::SampleCountFlagBits g_msaa_samples = vk::SampleCountFlagBits::e1;
  vk::raii::Buffer g_vertex_buffer = nullptr;
  vk::raii::DeviceMemory g_vertex_buffer_memory = nullptr;
  vk::raii::Buffer g_index_buffer = nullptr;
  vk::raii::DeviceMemory g_index_buffer_memory = nullptr;
  std::vector<vk::raii::Buffer> g_ubo_buffer;
  std::vector<vk::raii::DeviceMemory> g_ubo_buffer_memory;
  std::vector<void*> g_ubo_buffer_maped;
  vk::raii::Buffer g_transfer_buffer = nullptr;
  vk::raii::DeviceMemory g_transfer_buffer_memory = nullptr;
  void* g_transfer_buffer_maped = nullptr;
  uint32_t g_mip_levels = 1;
  vk::raii::Image g_texture_image = nullptr;
  vk::raii::DeviceMemory g_texture_image_memory  = nullptr;
  vk::raii::ImageView g_texture_image_view = nullptr;
  vk::raii::Sampler g_texture_image_sampler = nullptr;
  vk::raii::Image g_depth_image = nullptr;
  vk::raii::DeviceMemory g_depth_image_memory  = nullptr;
  vk::raii::ImageView g_depth_image_view = nullptr;
  vk::raii::Sampler g_depth_image_sampler = nullptr;
  vk::raii::Image g_gbuffer_color_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_color_image_memory  = nullptr;
  vk::raii::ImageView g_gbuffer_color_image_view = nullptr;
  vk::raii::Image g_gbuffer_position_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_position_image_memory  = nullptr;
  vk::raii::ImageView g_gbuffer_position_image_view = nullptr;
  vk::raii::Image g_gbuffer_normal_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_normal_image_memory  = nullptr;
  vk::raii::ImageView g_gbuffer_normal_image_view = nullptr;
  vk::raii::Image g_gbuffer_roughness_f0_image = nullptr;
  vk::raii::DeviceMemory g_gbuffer_roughness_f0_image_memory  = nullptr;
  vk::raii::ImageView g_gbuffer_roughness_f0_image_view = nullptr;
  vk::Format g_depth_image_format = vk::Format::eUndefined;
  vk::raii::CommandPool g_command_pool = nullptr;
  std::vector<vk::raii::CommandBuffer> g_command_buffer;
  std::vector<vk::raii::Semaphore> g_present_complete_semaphore;
  std::vector<vk::raii::Semaphore> g_render_finished_semaphore;
  std::vector<vk::raii::Fence> g_draw_fence;
  vk::raii::DescriptorPool g_descriptor_pool = nullptr;
  vk::raii::DescriptorSetLayout g_descriptor_set_layout = nullptr;
  std::vector<vk::raii::DescriptorSet> g_descriptor_sets;
  std::vector<Vertex> g_vertex_in;
  std::vector<uint32_t> g_index_in;
  constexpr uint32_t kParticleCount = 256;
  vk::raii::PipelineLayout g_particle_pipeline_layout = nullptr;
  vk::raii::Pipeline g_particle_pipeline = nullptr;
  vk::raii::PipelineLayout g_compute_pipeline_layout = nullptr;
  vk::raii::Pipeline g_compute_pipeline = nullptr;
  std::vector<vk::raii::Buffer> g_particle_ubo_buffer;
  std::vector<vk::raii::DeviceMemory> g_particle_ubo_buffer_memory;
  std::vector<void*> g_particle_ubo_buffer_maped;
  std::vector<vk::raii::Buffer> g_particle_buffer;
  std::vector<vk::raii::DeviceMemory> g_particle_buffer_memory;
  static uint64_t g_particle_compute_count = 0;
  vk::raii::Semaphore g_particle_compute_semaphore = nullptr;
  vk::raii::PipelineLayout g_shadowmap_pipeline_layout = nullptr;
  vk::raii::Pipeline g_shadowmap_pipeline = nullptr;
  vk::Format g_shadowmap_image_format = vk::Format::eD32Sfloat;
  vk::raii::Image g_shadowmap_image = nullptr;
  vk::raii::DeviceMemory g_shadowmap_image_memory  = nullptr;
  vk::raii::ImageView g_shadowmap_image_view = nullptr;
  uint32_t g_shadowmap_width = 1600;
  uint32_t g_shadowmap_height = 1200;
  vk::raii::Image g_bloom_image = nullptr;
  vk::raii::DeviceMemory g_bloom_image_memory  = nullptr;
  vk::raii::ImageView g_bloom_image_view = nullptr;
  std::vector<vk::raii::ImageView> g_bloom_image_views;
  constexpr uint32_t g_bloom_mip_levels = 6;
  vk::raii::PipelineLayout g_bloom_downsample_pipeline_layout = nullptr;
  vk::raii::Pipeline g_bloom_downsample_pipeline = nullptr;
  vk::raii::PipelineLayout g_bloom_upsample_pipeline_layout = nullptr;
  vk::raii::Pipeline g_bloom_upsample_pipeline = nullptr;
  constexpr float kBloomRate = 1.5f;
  ImGuiContext* g_imgui_context;
	vk::raii::DescriptorPool g_imgui_pool = nullptr;
  float g_pbr_roughness = 0.5f;
  float g_pbr_f0 = 0.04f;
  float g_pbr_metallic = 0.0f;
  float g_light_intensity = 10.0f;
  bool g_enable_ssao = true;
  bool g_enable_bloom = true;

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
  void InitImGui(){
    g_imgui_context = ImGui::CreateContext();
    vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eSampler, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eSampledImage, 1000},
      {vk::DescriptorType::eStorageImage, 1000},
      {vk::DescriptorType::eUniformTexelBuffer, 1000},
      {vk::DescriptorType::eStorageTexelBuffer, 1000},
      {vk::DescriptorType::eUniformBuffer, 1000},
      {vk::DescriptorType::eStorageBuffer, 1000},
      {vk::DescriptorType::eUniformBufferDynamic, 1000},
      {vk::DescriptorType::eStorageBufferDynamic, 1000},
      {vk::DescriptorType::eInputAttachment, 100 }
    };
    vk::DescriptorPoolCreateInfo pool_info = {
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = 1000,
      .poolSizeCount = (uint32_t)std::size(pool_sizes),
      .pPoolSizes = pool_sizes
    };
    g_imgui_pool = vk::raii::DescriptorPool(g_device, pool_info);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = *g_vk_instance;
    init_info.PhysicalDevice = *g_physical_device;
    init_info.Device = *g_device;
    init_info.Queue = *g_queue;
    init_info.DescriptorPool = *g_imgui_pool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;
    init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    VkFormat format = static_cast<VkFormat>(g_swapchain_image_format);
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &format;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_CreateFontsTexture();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(g_window, true);
  }
  void CreateInstance();
  void CreateaSurface();
  void PickPhysicalDevice();
  void CreateLogicalDevice();
  void CreateSwapChain();
  void CreateImageViews();
  void CreateColorResources();
  void CreateDepthResources();
  void CreateShadowmapResources();
  void CreateGbufferResources();
  void CreateBloomResources();
  void CreateDescriptorSetLayout();
  void CreatePipelines();
  void CreateCommandPool();
  void CreateTextureImage();
  void CreateTextureSampler();
  void CreateDateBuffers();
  void LoadModel();
  void UpdateModel();
  void CreateVertexBuffer();
  void CreateIndexBuffer();
  void CreateUboBuffer();
  void CreateParticleResources();
  void CreateDescriptorPool();
  void CreateDescriptorSets();
  void CreateCommandBuffer();
  void CreateSyncObjects();
  void Logs(){
    LOG(string("DATA_FILE_PATH: ")+DATA_FILE_PATH);
  }
  void InitVulkan(){
    Logs();
    CreateInstance();
    CreateaSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    CreateImageViews();
    CreateColorResources();
    CreateDepthResources();
    CreateShadowmapResources();
    CreateGbufferResources();
    CreateBloomResources();
    CreateDescriptorSetLayout();
    CreatePipelines();
    CreateCommandPool();
    LoadModel();
    CreateTextureImage();
    CreateTextureSampler();
    CreateDateBuffers();
    CreateParticleResources();
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
  vk::SampleCountFlagBits GetMaxUsableSampleCount() {
    vk::PhysicalDeviceProperties properties = g_physical_device.getProperties();
    vk::SampleCountFlags counts = properties.limits.framebufferColorSampleCounts & properties.limits.framebufferDepthSampleCounts;
    if(counts & vk::SampleCountFlagBits::e64){
      return vk::SampleCountFlagBits::e64;
    }
    if(counts & vk::SampleCountFlagBits::e32){
      return vk::SampleCountFlagBits::e32;
    }
    if(counts & vk::SampleCountFlagBits::e16){
      return vk::SampleCountFlagBits::e16;
    }
    if(counts & vk::SampleCountFlagBits::e8){
      return vk::SampleCountFlagBits::e8;
    }
    if(counts & vk::SampleCountFlagBits::e4){
      return vk::SampleCountFlagBits::e4;
    }
    if(counts & vk::SampleCountFlagBits::e2){
      return vk::SampleCountFlagBits::e2;
    }
    return vk::SampleCountFlagBits::e1;
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
        return (qfp.queueFlags&vk::QueueFlagBits::eGraphics)&&(qfp.queueFlags&vk::QueueFlagBits::eCompute);
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
      auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
      bool supports_required_features =
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering && features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 && features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState && features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy && features.get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore && features.get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>().dynamicRenderingLocalRead;
      if(found == false||(device_properties.deviceType==vk::PhysicalDeviceType::eDiscreteGpu&&g_physical_device.getProperties().deviceType!=vk::PhysicalDeviceType::eDiscreteGpu&&supports_required_features)){
        g_physical_device = device;
        found = true;
      }
    }
    if (found == false) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
    // g_msaa_samples = GetMaxUsableSampleCount();
    LOG("using physical device: ", g_physical_device.getProperties().deviceName);
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
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR> feature_chain = {
      {.features = {.samplerAnisotropy = true}},
      {.synchronization2 = true, .dynamicRendering = true},
      {.extendedDynamicState = true},
      {.timelineSemaphore = true},
      {.dynamicRenderingLocalRead = true}
    };
    vk::DeviceCreateInfo device_create_info{
      .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
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
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eTransferDst,
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
    g_frame_in_flight = g_swapchain_images.size();
    g_swapchain_image_format = select_format.format;
    g_swapchain_extent=select_extent;
  }
  vk::raii::ImageView CreateImageView(const vk::Image&image, uint32_t base_mip_level, uint32_t mip_levels, vk::Format format, vk::ImageAspectFlagBits aspect){
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
        .baseMipLevel = base_mip_level,
        .levelCount = mip_levels,
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
        CreateImageView(image,0,1,g_swapchain_image_format,vk::ImageAspectFlagBits::eColor));
    }
  }
  vk::raii::ShaderModule CreateShaderModule(const char*shader_file_path){
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
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 2,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 3,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 4,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 5,
        .descriptorType = vk::DescriptorType::eInputAttachment,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 6,
        .descriptorType = vk::DescriptorType::eInputAttachment,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 7,
        .descriptorType = vk::DescriptorType::eInputAttachment,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 8,
        .descriptorType = vk::DescriptorType::eInputAttachment,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 9,
        .descriptorType = vk::DescriptorType::eSampledImage,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 10,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      {
        .binding = 11,
        .descriptorType = vk::DescriptorType::eSampledImage,
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
  void CreatePipelines(){
    LOG(string("SHADER_FILE_PATH: ")+SHADER_FILE_PATH);
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
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasConstantFactor = 1.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth =1.0f 
    };
    vk::PipelineMultisampleStateCreateInfo multisample_create_info{
      .rasterizationSamples = g_msaa_samples,
      .sampleShadingEnable = vk::False,
    };
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_info{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False
    };
    vk::PipelineColorBlendAttachmentState opaque_blend_attachment{
      .blendEnable = vk::False,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    std::vector<vk::PipelineColorBlendAttachmentState>color_blend_attachments(5,opaque_blend_attachment);
    vk::PipelineColorBlendStateCreateInfo color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = static_cast<uint32_t>(color_blend_attachments.size()),
      .pAttachments = color_blend_attachments.data(),
      // .blendConstants = ,
    };
    vk::PipelineLayoutCreateInfo pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*g_descriptor_set_layout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };
    g_pipeline_layout = vk::raii::PipelineLayout(g_device, pipeline_layout_info);
    std::vector<vk::Format> graphsic_formats{g_gbuffer_format,g_gbuffer_format,g_gbuffer_format,g_gbuffer_format,g_gbuffer_format};
    vk::PipelineRenderingCreateInfo pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
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

    vk::GraphicsPipelineCreateInfo lighting_pipeline_info = pipeline_info;
    vk::PipelineRenderingCreateInfo lighting_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
      .depthAttachmentFormat = g_depth_image_format,
    };
    vk::PipelineShaderStageCreateInfo lighting_pipeline_shader_stage_create_info[2] = {
      {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shader_module,
        .pName = "vertLighting",
        .pSpecializationInfo = nullptr
      },
      {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shader_module,
        .pName = "fragLighting",
        .pSpecializationInfo = nullptr
      }
    };
    vk::PipelineVertexInputStateCreateInfo lighting_vertex_input_info{};
    vk::PipelineInputAssemblyStateCreateInfo lighting_input_assembly_info{
      .topology = vk::PrimitiveTopology::eTriangleStrip
    };
    vk::PipelineRasterizationStateCreateInfo lighting_rasterization_create_info{
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
    vk::PipelineDepthStencilStateCreateInfo lighting_depth_stencil_info{
      .depthTestEnable = vk::False,
      .depthWriteEnable = vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False
    };
    std::vector<vk::PushConstantRange>lighting_push_constant_range{
      {
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(LightingPushConstants),
      }
    };
    vk::PipelineLayoutCreateInfo lighting_pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*g_descriptor_set_layout,
      .pushConstantRangeCount = static_cast<uint32_t>(lighting_push_constant_range.size()),
      .pPushConstantRanges = lighting_push_constant_range.data()
    };
    g_lighting_pipeline_layout = vk::raii::PipelineLayout(g_device, lighting_pipeline_layout_info);
    lighting_pipeline_info.pNext = &lighting_pipeline_rending_info;
    lighting_pipeline_info.pStages = lighting_pipeline_shader_stage_create_info;
    lighting_pipeline_info.pVertexInputState = &lighting_vertex_input_info;
    lighting_pipeline_info.pInputAssemblyState = &lighting_input_assembly_info;
    lighting_pipeline_info.pRasterizationState = &lighting_rasterization_create_info;
    lighting_pipeline_info.pDepthStencilState = &lighting_depth_stencil_info;
    lighting_pipeline_info.layout = g_lighting_pipeline_layout;
    g_lighting_pipeline = vk::raii::Pipeline(g_device, nullptr, lighting_pipeline_info);

    vk::GraphicsPipelineCreateInfo bloom_pipeline_info = lighting_pipeline_info;
    std::vector<vk::Format> bloom_formats(g_bloom_mip_levels,g_gbuffer_format);
    vk::PipelineRenderingCreateInfo bloom_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(bloom_formats.size()),
      .pColorAttachmentFormats = bloom_formats.data(),
    };
    vk::PipelineShaderStageCreateInfo bloom_downsample_pipeline_shader_stage_create_info[2] = {
      {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shader_module,
        .pName = "vertBloomDownsample",
        .pSpecializationInfo = nullptr
      },
      {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shader_module,
        .pName = "fragBloomDownsample",
        .pSpecializationInfo = nullptr
      }
    };
    vk::PipelineShaderStageCreateInfo bloom_upsample_pipeline_shader_stage_create_info[2] = {
      {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shader_module,
        .pName = "vertBloomUpsample",
        .pSpecializationInfo = nullptr
      },
      {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shader_module,
        .pName = "fragBloomUpsample",
        .pSpecializationInfo = nullptr
      }
    };
    std::vector<vk::PipelineColorBlendAttachmentState>bloom_color_blend_attachments(g_bloom_mip_levels,opaque_blend_attachment);
    vk::PipelineColorBlendStateCreateInfo bloom_color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = static_cast<uint32_t>(bloom_color_blend_attachments.size()),
      .pAttachments = bloom_color_blend_attachments.data(),
    };
    std::vector<vk::PushConstantRange>bloom_push_constant_range{
      {
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(BloomPushConstants),
      }
    };
    vk::PipelineLayoutCreateInfo bloom_pipeline_layout_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*g_descriptor_set_layout,
      .pushConstantRangeCount = static_cast<uint32_t>(bloom_push_constant_range.size()),
      .pPushConstantRanges = bloom_push_constant_range.data()
    };
    g_bloom_downsample_pipeline_layout = vk::raii::PipelineLayout(g_device, bloom_pipeline_layout_info);
    bloom_pipeline_info.pNext = bloom_pipeline_rending_info;
    bloom_pipeline_info.pStages = bloom_downsample_pipeline_shader_stage_create_info;
    bloom_pipeline_info.pColorBlendState = &bloom_color_blend_info;
    bloom_pipeline_info.layout = g_bloom_downsample_pipeline_layout;
    g_bloom_downsample_pipeline = vk::raii::Pipeline(g_device, nullptr, bloom_pipeline_info);
    g_bloom_upsample_pipeline_layout = vk::raii::PipelineLayout(g_device, bloom_pipeline_layout_info);
    bloom_pipeline_info.pStages = bloom_upsample_pipeline_shader_stage_create_info;
    bloom_pipeline_info.layout = g_bloom_upsample_pipeline_layout;
    g_bloom_upsample_pipeline = vk::raii::Pipeline(g_device, nullptr, bloom_pipeline_info);

    vk::GraphicsPipelineCreateInfo shodowmap_pipeline_info = pipeline_info;
    vk::PipelineRenderingCreateInfo shodowmap_pipeline_rending_info{
      .colorAttachmentCount = 0,
      .depthAttachmentFormat = g_shadowmap_image_format,
    };
    vk::PipelineShaderStageCreateInfo shodowmap_pipeline_shader_stage_create_info[2] = {
      {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shader_module,
        .pName = "vertShadowmap",
        .pSpecializationInfo = nullptr
      },
      {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shader_module,
        .pName = "fragShadowmap",
        .pSpecializationInfo = nullptr
      }
    };
    vk::PipelineRasterizationStateCreateInfo shadowmap_rasterization_create_info{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasConstantFactor = 1.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth =1.0f 
    };
    g_shadowmap_pipeline_layout = vk::raii::PipelineLayout(g_device, pipeline_layout_info);
    shodowmap_pipeline_info.pNext = &shodowmap_pipeline_rending_info;
    shodowmap_pipeline_info.pStages = shodowmap_pipeline_shader_stage_create_info;
    shodowmap_pipeline_info.pRasterizationState = &shadowmap_rasterization_create_info;
    shodowmap_pipeline_info.layout = g_shadowmap_pipeline_layout;
    g_shadowmap_pipeline = vk::raii::Pipeline(g_device, nullptr, shodowmap_pipeline_info);

    vk::GraphicsPipelineCreateInfo particle_pipeline_info = pipeline_info;
    g_particle_pipeline_layout = vk::raii::PipelineLayout(g_device, pipeline_layout_info);
    vk::PipelineRenderingCreateInfo particle_pipeline_rending_info{
      .colorAttachmentCount = static_cast<uint32_t>(graphsic_formats.size()),
      .pColorAttachmentFormats = graphsic_formats.data(),
      .depthAttachmentFormat = g_depth_image_format,
    };
    particle_pipeline_info.pNext = &particle_pipeline_rending_info;
    vk::PipelineShaderStageCreateInfo particle_pipeline_shader_stage_create_info[2] = {
      {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shader_module,
        .pName = "vertParticle",
        .pSpecializationInfo = nullptr
      },
      {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shader_module,
        .pName = "fragParticle",
        .pSpecializationInfo = nullptr
      }
    };
    particle_pipeline_info.pStages = particle_pipeline_shader_stage_create_info;
    binding_desc = Particle::GetBindingDescription();
    auto particle_attribute_desc = Particle::GetAttributeDescription();
    vk::PipelineVertexInputStateCreateInfo particle_vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding_desc,
      .vertexAttributeDescriptionCount = particle_attribute_desc.size(),
      .pVertexAttributeDescriptions = particle_attribute_desc.data(),
    };
    particle_pipeline_info.pVertexInputState = &particle_vertex_input_info;
    input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::ePointList
    };
    particle_pipeline_info.pInputAssemblyState = &input_assembly_info;
    vk::PipelineDepthStencilStateCreateInfo particle_depth_stencil_info{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False
    };
    particle_pipeline_info.pDepthStencilState = &particle_depth_stencil_info;
    vk::PipelineColorBlendAttachmentState transparent_blend_attachment{
      .blendEnable = vk::True,
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOne,
      .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    std::vector<vk::PipelineColorBlendAttachmentState>particle_color_blend_infos(5,transparent_blend_attachment);
    vk::PipelineColorBlendStateCreateInfo particle_color_blend_info{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = static_cast<uint32_t>(particle_color_blend_infos.size()),
      .pAttachments = particle_color_blend_infos.data(),
    };
    particle_pipeline_info.pColorBlendState = &particle_color_blend_info;
    g_particle_pipeline = vk::raii::Pipeline(g_device, nullptr, particle_pipeline_info);

    vk::PipelineShaderStageCreateInfo compute_pipeline_shader_stage_create_info{
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = shader_module,
      .pName = "compParticle",
      .pSpecializationInfo = nullptr
    };
    vk::PipelineLayoutCreateInfo compute_pipeline_layout_info{.setLayoutCount = 1, .pSetLayouts = &*g_descriptor_set_layout};
    g_compute_pipeline_layout = vk::raii::PipelineLayout(g_device, compute_pipeline_layout_info); 
    vk::ComputePipelineCreateInfo compute_pipeline_info{
      .stage = compute_pipeline_shader_stage_create_info,
      .layout = g_compute_pipeline_layout,
    };
    g_compute_pipeline = vk::raii::Pipeline(g_device, nullptr, compute_pipeline_info);
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
  void LoadModel(){
    tinyobj::attrib_t attr;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err,warn;
    if(!tinyobj::LoadObj(&attr,&shapes,&materials,&warn,&err,DATA_FILE_PATH"/viking_room.obj")){
      throw std::runtime_error(warn + err);
    }
    std::unordered_map<Vertex,uint32_t> index_map;
    for(const auto& shape:shapes){
      for(const auto& index: shape.mesh.indices){
        Vertex vertex{
          .position = {
            attr.vertices[3*index.vertex_index+0],
            attr.vertices[3*index.vertex_index+1],
            attr.vertices[3*index.vertex_index+2]
          },
          .roughness_f0 = {g_pbr_roughness, g_pbr_f0, g_pbr_f0, g_pbr_f0},
          .normal = {
            attr.normals[3*index.normal_index+0],
            attr.normals[3*index.normal_index+1],
            attr.normals[3*index.normal_index+2],
          },
          .metallic = g_pbr_metallic,
          .tex_coord = {
            attr.texcoords[2*index.texcoord_index+0],
            1.0f-attr.texcoords[2*index.texcoord_index+1]
          }
        };
        if(!index_map.count(vertex)){
          index_map[vertex] = g_vertex_in.size();
          g_vertex_in.emplace_back(vertex);
        }
        g_index_in.emplace_back(index_map[vertex]);
      }
    }
  }
  void UpdateModel(){
    for(Vertex& vertex : g_vertex_in){
      vertex.roughness_f0 = {g_pbr_roughness, g_pbr_f0, g_pbr_f0, g_pbr_f0};
      vertex.metallic = g_pbr_metallic;
    }
    uint32_t size = sizeof(g_vertex_in[0])*g_vertex_in.size();
    memcpy(g_transfer_buffer_maped, g_vertex_in.data(), size);
    CopyBuffer(g_transfer_buffer,g_vertex_buffer, size);
  }
  void CreateVertexBuffer(){
    uint32_t size = sizeof(g_vertex_in[0])*g_vertex_in.size();
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, g_transfer_buffer, g_transfer_buffer_memory);
    g_transfer_buffer_maped = g_transfer_buffer_memory.mapMemory(0, size);
    memcpy(g_transfer_buffer_maped, g_vertex_in.data(), size);
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eDeviceLocal, g_vertex_buffer, g_vertex_buffer_memory);
    CopyBuffer(g_transfer_buffer,g_vertex_buffer, size);
  }
  void CreateIndexBuffer(){
    uint32_t size = sizeof(g_index_in[0])*g_index_in.size();
    memcpy(g_transfer_buffer_maped, g_index_in.data(), size);
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eDeviceLocal, g_index_buffer, g_index_buffer_memory);
    CopyBuffer(g_transfer_buffer,g_index_buffer, size);
  }
  void CreateUboBuffer(){
    g_ubo_buffer.clear();
    g_ubo_buffer_memory.clear();
    g_ubo_buffer_maped.clear();

    for(uint32_t i=0;i<g_frame_in_flight;++i){
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
  void CreateParticleResources(){
    g_particle_ubo_buffer.clear();
    g_particle_ubo_buffer_memory.clear();
    g_particle_ubo_buffer_maped.clear();
    g_particle_buffer.clear();
    g_particle_buffer_memory.clear();

    uint32_t size = sizeof(ParticleUbo);
    for(uint32_t i=0;i<g_frame_in_flight;++i){
      vk::raii::Buffer buffer = nullptr;
      vk::raii::DeviceMemory memory = nullptr;
      CreateBuffer(
        size, vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, memory);
      void* data = memory.mapMemory(0, size);
      g_particle_ubo_buffer.emplace_back(std::move(buffer));
      g_particle_ubo_buffer_memory.emplace_back(std::move(memory));
      g_particle_ubo_buffer_maped.emplace_back(data);
      size = sizeof(Particle)*kParticleCount;
      buffer = nullptr;
      memory = nullptr;
      CreateBuffer(
        size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
        vk::MemoryPropertyFlagBits::eDeviceLocal, buffer, memory);
      g_particle_buffer.emplace_back(std::move(buffer));
      g_particle_buffer_memory.emplace_back(std::move(memory));
    }
    std::vector<Particle>particles{kParticleCount};
    constexpr float kParticleRange = 1.3f;
    for(uint32_t i=0;i<kParticleCount;++i){
      particles[i].pos.x=std::rand()/float(RAND_MAX)*kParticleRange-kParticleRange/2.0f;
      particles[i].pos.y=std::rand()/float(RAND_MAX)*kParticleRange-kParticleRange/2.0f;
      particles[i].pos.z=0.3f;
      particles[i].v=glm::vec3{0.0f,0.0f,0.2f};
      particles[i].color=glm::vec3(1.0f);
    }
    vk::raii::Buffer temp_buffer = nullptr;
    vk::raii::DeviceMemory temp_memory = nullptr;
    size = sizeof(Particle)*kParticleCount;
    CreateBuffer(
      size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, temp_buffer, temp_memory);
    void* data = temp_memory.mapMemory(0, size);
    memcpy(data,particles.data(),size);
    temp_memory.unmapMemory();
    CopyBuffer(temp_buffer,g_particle_buffer[g_frame_in_flight-1],size);
  }
  void CreateDescriptorPool(){
    std::vector<vk::DescriptorPoolSize> pool_sizes{
      {
        .type=vk::DescriptorType::eUniformBuffer,
        .descriptorCount=g_frame_in_flight*2
      },
      {
        .type=vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount=g_frame_in_flight*2
      },
      {
        .type=vk::DescriptorType::eStorageBuffer,
        .descriptorCount=g_frame_in_flight*2
      },
      {
        .type=vk::DescriptorType::eInputAttachment,
        .descriptorCount=g_frame_in_flight*4
      },
      {
        .type=vk::DescriptorType::eSampledImage,
        .descriptorCount=g_frame_in_flight*2
      }
    };
    vk::DescriptorPoolCreateInfo descriptor_pool_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = static_cast<uint32_t>(g_frame_in_flight*pool_sizes.size()),
      .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
    };
    g_descriptor_pool = vk::raii::DescriptorPool(g_device,descriptor_pool_info);
  }
  void UpdateDescriptorSets(){
    for(uint32_t i=0;i<g_frame_in_flight;++i){
      vk::DescriptorBufferInfo buffer_info{.buffer = g_ubo_buffer[i], .offset = 0, .range = sizeof(UniformBufferObject)};
      vk::DescriptorBufferInfo particle_ubo_buffer_info{.buffer = g_particle_ubo_buffer[i], .offset = 0, .range = sizeof(ParticleUbo)};
      vk::DescriptorBufferInfo particle_last_frame_buffer_info{.buffer = g_particle_buffer[(i-1+g_frame_in_flight)%g_frame_in_flight], .offset = 0, .range = sizeof(Particle)*kParticleCount};
      vk::DescriptorBufferInfo particle_this_frame_buffer_info{.buffer = g_particle_buffer[i], .offset = 0, .range = sizeof(Particle)*kParticleCount};
      vk::DescriptorImageInfo image_info{
        .sampler = *g_texture_image_sampler,
        .imageView = *g_texture_image_view, 
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::DescriptorImageInfo gbuffer_color_info{
        .imageView = *g_gbuffer_color_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::DescriptorImageInfo gbuffer_position_info{
        .imageView = *g_gbuffer_position_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::DescriptorImageInfo gbuffer_normal_info{
        .imageView = *g_gbuffer_normal_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::DescriptorImageInfo gbuffer_roughness_f0_info{
        .imageView = *g_gbuffer_roughness_f0_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::DescriptorImageInfo shadowmap_info{
        .imageView = *g_shadowmap_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::DescriptorImageInfo depth_info{
        .sampler = *g_depth_image_sampler,
        .imageView = *g_depth_image_view,
        .imageLayout = vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
      };
      vk::DescriptorImageInfo bloom_info{
        .imageView = *g_bloom_image_view,
        .imageLayout = vk::ImageLayout::eGeneral
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
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 2,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo = &particle_ubo_buffer_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 3,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .pBufferInfo = &particle_last_frame_buffer_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 4,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageBuffer,
          .pBufferInfo = &particle_this_frame_buffer_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 5,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eInputAttachment,
          .pImageInfo = &gbuffer_color_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 6,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eInputAttachment,
          .pImageInfo = &gbuffer_position_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 7,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eInputAttachment,
          .pImageInfo = &gbuffer_normal_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 8,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eInputAttachment,
          .pImageInfo = &gbuffer_roughness_f0_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 9,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eSampledImage,
          .pImageInfo = &shadowmap_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 10,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .pImageInfo = &depth_info
        },
        {
          .dstSet = g_descriptor_sets[i],
          .dstBinding = 11,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eSampledImage,
          .pImageInfo = &bloom_info
        }
      };
      g_device.updateDescriptorSets(descriptor_write, {});
    }
  }
  void CreateDescriptorSets(){
    std::vector<vk::DescriptorSetLayout> layouts(g_frame_in_flight, *g_descriptor_set_layout);
    vk::DescriptorSetAllocateInfo alloc_info{
      .descriptorPool = *g_descriptor_pool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()
    };
    g_descriptor_sets = g_device.allocateDescriptorSets(alloc_info);
    UpdateDescriptorSets();
  }
  void TransitionImageLayout(vk::raii::Image& image, uint32_t mip_levels, vk::ImageLayout  old_layout, vk::ImageLayout new_layout){
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
        .levelCount = mip_levels,
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
  void CreateImage(uint32_t width, uint32_t height, uint32_t mip_levels,vk::SampleCountFlagBits sample_count,vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& memory){
    vk::ImageCreateInfo image_info{
      .flags = {},
      .imageType = vk::ImageType::e2D,
      .format = format,
      .extent = {width,height,1},
      .mipLevels = mip_levels,
      .arrayLayers = 1,
      .samples = sample_count,
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
  void GenerateMipmaps(const vk::raii::Image& image, vk::Format format, int32_t width, int32_t height, uint32_t mip_levels){
    vk::FormatProperties properties = g_physical_device.getFormatProperties(format);
    if(!(properties.optimalTilingFeatures&vk::FormatFeatureFlagBits::eSampledImageFilterLinear)){
      throw std::runtime_error("texture image format does not support linear blitting!");
    }
    vk::raii::CommandBuffer command_buffer = BeginOneTimeCommandBuffer();
    vk::ImageMemoryBarrier barrier{
      .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
      .dstAccessMask = vk::AccessFlagBits::eTransferRead,
      .oldLayout = vk::ImageLayout::eTransferDstOptimal,
      .newLayout = vk::ImageLayout::eTransferSrcOptimal,
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
    int32_t mip_width = width, mip_height = height;
    for(uint32_t i=1;i<mip_levels;++i){
      barrier.subresourceRange.baseMipLevel = i-1;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eTransfer,{},{},nullptr,barrier);
      vk::ImageBlit image_blit{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor,i-1,0,1},
        .srcOffsets = std::array{vk::Offset3D{0,0,0},vk::Offset3D{mip_width,mip_height,1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor,i,0,1},
        .dstOffsets = std::array{vk::Offset3D{0,0,0},vk::Offset3D{mip_width>1?mip_width/2:1,mip_height>1?mip_height/2:1,1}},
      };
      command_buffer.blitImage(
        image,vk::ImageLayout::eTransferSrcOptimal,image,vk::ImageLayout::eTransferDstOptimal,image_blit,vk::Filter::eLinear
      );
      barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eFragmentShader,{},{},nullptr,barrier);
      mip_width = mip_width>1?mip_width/2:1;
      mip_height = mip_height>1?mip_height/2:1;
    }
    barrier.subresourceRange.baseMipLevel = mip_levels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, nullptr, barrier);
    EndOneTimeCommandBuffer(command_buffer);
  }
  void CreateTextureImage(){
    int tex_width,tex_height,tex_channels;
    stbi_uc*pixels = stbi_load(DATA_FILE_PATH"/viking_room.png",&tex_width,&tex_height,&tex_channels,STBI_rgb_alpha);
    vk::DeviceSize image_size = tex_width*tex_height*4;
    uint32_t mip_levels=std::floor(std::log2(std::max(tex_width,tex_height)))+1;
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
      tex_width,tex_height,mip_levels,vk::SampleCountFlagBits::e1,vk::Format::eR8G8B8A8Srgb,vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_texture_image,g_texture_image_memory
    );
    TransitionImageLayout(g_texture_image,mip_levels,vk::ImageLayout::eUndefined,vk::ImageLayout::eTransferDstOptimal);
    CopyBufferToImage(buffer,g_texture_image,tex_width,tex_height);
    GenerateMipmaps(g_texture_image,vk::Format::eR8G8B8A8Srgb,tex_width,tex_height,mip_levels);
    g_texture_image_view = CreateImageView(*g_texture_image,0,mip_levels,vk::Format::eR8G8B8A8Srgb,vk::ImageAspectFlagBits::eColor);
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
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::True,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0,
      .maxLod = vk::LodClampNone,
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
  void CreateColorResources(){
    vk::Format format = g_swapchain_image_format;
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,1,g_msaa_samples,format,vk::ImageTiling::eOptimal,vk::ImageUsageFlagBits::eTransientAttachment|
      vk::ImageUsageFlagBits::eColorAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_color_image,g_color_image_memory
    );
    g_color_image_view = CreateImageView(*g_color_image,0,1,format,vk::ImageAspectFlagBits::eColor);
  }
  void CreateDepthResources(){
    g_depth_image_format = FindSupportDepthFormat();
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,1,g_msaa_samples,g_depth_image_format,vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment|
      vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_depth_image,g_depth_image_memory
    );
    g_depth_image_view = CreateImageView(*g_depth_image,0,1,g_depth_image_format,vk::ImageAspectFlagBits::eDepth);
    vk::PhysicalDeviceProperties properties = g_physical_device.getProperties();
    vk::SamplerCreateInfo sampler_info{
      .flags = {},
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eNearest,
      .addressModeU = vk::SamplerAddressMode::eClampToEdge,
      .addressModeV = vk::SamplerAddressMode::eClampToEdge,
      .addressModeW = vk::SamplerAddressMode::eClampToEdge,
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::False,
      .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0,
      .maxLod = 0,
      .borderColor = vk::BorderColor::eIntTransparentBlack,
      .unnormalizedCoordinates = vk::False,
    };
    g_depth_image_sampler = vk::raii::Sampler(g_device, sampler_info);
  }
  void CreateShadowmapResources(){
    CreateImage(
      g_shadowmap_width,g_shadowmap_height,1,vk::SampleCountFlagBits::e1,g_shadowmap_image_format,vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment|
      vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_shadowmap_image,g_shadowmap_image_memory
    );
    g_shadowmap_image_view = CreateImageView(*g_shadowmap_image,0,1,g_shadowmap_image_format,vk::ImageAspectFlagBits::eDepth);
  }
  void CreateGbufferResources(){
    vk::Format format = g_gbuffer_format;
    // dont use msaa if using deferred lighting
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,1,g_msaa_samples,format,vk::ImageTiling::eOptimal,vk::ImageUsageFlagBits::eTransientAttachment|
      vk::ImageUsageFlagBits::eColorAttachment|
      vk::ImageUsageFlagBits::eInputAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_gbuffer_color_image,g_gbuffer_color_image_memory
    );
    g_gbuffer_color_image_view = CreateImageView(*g_gbuffer_color_image,0,1,format,vk::ImageAspectFlagBits::eColor);
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,1,g_msaa_samples,format,vk::ImageTiling::eOptimal,vk::ImageUsageFlagBits::eTransientAttachment|
      vk::ImageUsageFlagBits::eColorAttachment|
      vk::ImageUsageFlagBits::eInputAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_gbuffer_position_image,g_gbuffer_position_image_memory
    );
    g_gbuffer_position_image_view = CreateImageView(*g_gbuffer_position_image,0,1,format,vk::ImageAspectFlagBits::eColor);
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,1,g_msaa_samples,format,vk::ImageTiling::eOptimal,vk::ImageUsageFlagBits::eTransientAttachment|
      vk::ImageUsageFlagBits::eColorAttachment|
      vk::ImageUsageFlagBits::eInputAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_gbuffer_normal_image,g_gbuffer_normal_image_memory
    );
    g_gbuffer_normal_image_view = CreateImageView(*g_gbuffer_normal_image,0,1,format,vk::ImageAspectFlagBits::eColor);
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,1,g_msaa_samples,format,vk::ImageTiling::eOptimal,vk::ImageUsageFlagBits::eTransientAttachment|
      vk::ImageUsageFlagBits::eColorAttachment|
      vk::ImageUsageFlagBits::eInputAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_gbuffer_roughness_f0_image,g_gbuffer_roughness_f0_image_memory
    );
    g_gbuffer_roughness_f0_image_view = CreateImageView(*g_gbuffer_roughness_f0_image,0,1,format,vk::ImageAspectFlagBits::eColor);
  }
  void CreateBloomResources(){
    CreateImage(
      g_swapchain_extent.width,g_swapchain_extent.height,g_bloom_mip_levels,vk::SampleCountFlagBits::e1,g_gbuffer_format,vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eColorAttachment|
      vk::ImageUsageFlagBits::eTransferSrc|
      vk::ImageUsageFlagBits::eTransferDst|
      vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      g_bloom_image,g_bloom_image_memory
    );
    g_bloom_image_view = CreateImageView(*g_bloom_image,0,g_bloom_mip_levels,g_gbuffer_format,vk::ImageAspectFlagBits::eColor);
    g_bloom_image_views.clear();
    for(int i=0;i<g_bloom_mip_levels;++i){
      g_bloom_image_views.emplace_back(CreateImageView(*g_bloom_image,i,1,g_gbuffer_format,vk::ImageAspectFlagBits::eColor));
    }
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
      .commandBufferCount = static_cast<uint32_t>(g_frame_in_flight)*3
    };
    g_command_buffer = vk::raii::CommandBuffers(g_device, alloc_info);
  }
  void TransformImageLayout(
    uint32_t image_index,
    uint32_t command_buffer_index,
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
    g_command_buffer[command_buffer_index].pipelineBarrier2(dependency_info);
  }
  void TransformImageLayoutCustom(
    vk::raii::Image& image, 
    uint32_t command_buffer_index,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags aspect_flags,
    uint32_t base_mip_level = 0,
    uint32_t level_count = 1
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
      .image = image,
      .subresourceRange = {
        .aspectMask = aspect_flags,
        .baseMipLevel = base_mip_level,
        .levelCount = level_count,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    vk::DependencyInfo dependency_info{
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
    };
    g_command_buffer[command_buffer_index].pipelineBarrier2(dependency_info);
  }
  void RecordCommandBuffer(
    uint32_t image_index,
    uint32_t frame_index,
    vk::Viewport viewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(g_swapchain_extent.width),static_cast<float>(g_swapchain_extent.height), 0.0f, 1.0f),
    vk::Rect2D scissor = vk::Rect2D({0,0}, g_swapchain_extent)){
    g_command_buffer[frame_index].begin({});
    TransformImageLayout(
      image_index,
      frame_index,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eTransferDstOptimal,
      {},
      vk::AccessFlagBits2::eTransferWrite,
      vk::PipelineStageFlagBits2::eTopOfPipe,
      vk::PipelineStageFlagBits2::eTransfer
    );
    TransformImageLayoutCustom(g_color_image, frame_index, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {}, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    vk::ImageAspectFlagBits::eColor);
    TransformImageLayoutCustom(g_depth_image, frame_index, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, {}, vk::AccessFlagBits2::eDepthStencilAttachmentRead|vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests|vk::PipelineStageFlagBits2::eLateFragmentTests, vk::ImageAspectFlagBits::eDepth);
    TransformImageLayoutCustom(g_bloom_image, frame_index, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    vk::ImageAspectFlagBits::eColor, 0, g_bloom_mip_levels);
    // https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_dynamic_rendering_local_read.html
    // can not change attachments inside renderpass, use superset and remapping
    std::vector<vk::RenderingAttachmentInfo> attachment_infos{
      // {
      //   .imageView = g_color_image_view,
      //   .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      //   .resolveMode = vk::ResolveModeFlagBits::eAverage,
      //   .resolveImageView = g_swapchain_image_views[image_index],
      //   .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      //   .loadOp = vk::AttachmentLoadOp::eClear,
      //   .storeOp = vk::AttachmentStoreOp::eStore,
      //   .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
      // },
      {
        .imageView = g_gbuffer_color_image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
      },
      {
        .imageView = g_gbuffer_position_image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
      },
      {
        .imageView = g_gbuffer_normal_image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
      },
      {
        .imageView = g_gbuffer_roughness_f0_image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
      },
      {
        .imageView = g_bloom_image_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
      }
    };
    vk::RenderingAttachmentInfo depth_info{
      .imageView = g_depth_image_view,
      .imageLayout = vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .clearValue = vk::ClearDepthStencilValue{1.0f, 0}
    };
    vk::RenderingInfo rendering_info{
      .renderArea = {.offset = {0, 0}, .extent = g_swapchain_extent},
      .layerCount = 1,
      .colorAttachmentCount = static_cast<uint32_t>(attachment_infos.size()),
      .pColorAttachments = attachment_infos.data(),
      .pDepthAttachment = &depth_info
    };
    // shadowmap pass
    TransformImageLayoutCustom(g_shadowmap_image, frame_index, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, {}, vk::AccessFlagBits2::eDepthStencilAttachmentRead|vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests|vk::PipelineStageFlagBits2::eLateFragmentTests, vk::ImageAspectFlagBits::eDepth);
    vk::RenderingAttachmentInfo shadowmap_depth_info{
      .imageView = g_shadowmap_image_view,
      .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = vk::ClearDepthStencilValue{1.0f, 0}
    };
    vk::RenderingInfo shadowmap_rendering_info{
      .renderArea = {
        .offset = {0, 0},
        .extent = {g_shadowmap_width, g_shadowmap_height}
      },
      .layerCount = 1,
      .colorAttachmentCount = 0,
      .pColorAttachments = nullptr,
      .pDepthAttachment = &shadowmap_depth_info
    };
    vk::Viewport shadermap_viewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(g_shadowmap_width),static_cast<float>(g_shadowmap_height), 0.0f, 1.0f);
    vk::Rect2D shadermap_scissor = vk::Rect2D({0, 0}, {g_shadowmap_width, g_shadowmap_height});
    g_command_buffer[frame_index].beginRendering(shadowmap_rendering_info);
    g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_shadowmap_pipeline);
    g_command_buffer[frame_index].bindVertexBuffers(0, *g_vertex_buffer, {0});
    g_command_buffer[frame_index].bindIndexBuffer(*g_index_buffer, 0, vk::IndexType::eUint32);
    g_command_buffer[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_shadowmap_pipeline_layout,0,*g_descriptor_sets[frame_index],nullptr);
    g_command_buffer[frame_index].setViewport(0, shadermap_viewport);
    g_command_buffer[frame_index].setScissor(0, shadermap_scissor);
    g_command_buffer[frame_index].drawIndexed(g_index_in.size(),1,0,0,0);
    g_command_buffer[frame_index].endRendering();
    // graphsic pass
    TransformImageLayoutCustom(g_shadowmap_image, frame_index, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal, vk::AccessFlagBits2::eDepthStencilAttachmentRead|vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eDepthStencilAttachmentRead, vk::PipelineStageFlagBits2::eEarlyFragmentTests|vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eDepth);
    g_command_buffer[frame_index].beginRendering(rendering_info);
    g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_graphics_pipeline);
    g_command_buffer[frame_index].bindVertexBuffers(0, *g_vertex_buffer, {0});
    g_command_buffer[frame_index].bindIndexBuffer(*g_index_buffer, 0, vk::IndexType::eUint32);
    g_command_buffer[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_pipeline_layout,0,*g_descriptor_sets[frame_index],nullptr);
    g_command_buffer[frame_index].setViewport(0, viewport);
    g_command_buffer[frame_index].setScissor(0, scissor);
    g_command_buffer[frame_index].drawIndexed(g_index_in.size(),1,0,0,0);
    // lighting pass
    TransformImageLayoutCustom(g_depth_image, frame_index, vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal , vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal , vk::AccessFlagBits2::eDepthStencilAttachmentRead|vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eDepthStencilAttachmentRead, vk::PipelineStageFlagBits2::eEarlyFragmentTests|vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eDepth);
    std::vector<uint32_t>lighting_attachment_locations{vk::AttachmentUnused,vk::AttachmentUnused,vk::AttachmentUnused,vk::AttachmentUnused,0};
    g_command_buffer[frame_index].setRenderingAttachmentLocations(vk::RenderingAttachmentLocationInfo{
      .colorAttachmentCount = static_cast<uint32_t>(lighting_attachment_locations.size()),
      .pColorAttachmentLocations = lighting_attachment_locations.data(),
    });
    g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_lighting_pipeline);
    g_command_buffer[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_lighting_pipeline_layout,0,*g_descriptor_sets[frame_index],nullptr);
    LightingPushConstants lighting_push_constants{.enable_ssao = g_enable_ssao};
    g_command_buffer[frame_index].pushConstants<LightingPushConstants>(g_lighting_pipeline_layout,vk::ShaderStageFlagBits::eFragment,0,lighting_push_constants);
    g_command_buffer[frame_index].draw(4,1,0,0);
    // barrier
    std::vector<vk::ImageMemoryBarrier2> barriers={
      {
        .srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests|
                        vk::PipelineStageFlagBits2::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests|
                        vk::PipelineStageFlagBits2::eLateFragmentTests,
        .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead|
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
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
      }
    };
    vk::DependencyInfo dependency_info{
      .dependencyFlags = vk::DependencyFlagBits::eByRegion,
      .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
      .pImageMemoryBarriers = barriers.data(),
    };
    g_command_buffer[frame_index].pipelineBarrier2(dependency_info);
    // particle pass
    g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_particle_pipeline);
    g_command_buffer[frame_index].bindVertexBuffers(0, *g_particle_buffer[g_frame_in_flight-1], {0});
    g_command_buffer[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_particle_pipeline_layout,0,*g_descriptor_sets[frame_index],nullptr);
    g_command_buffer[frame_index].draw(kParticleCount,1,0,0);
    g_command_buffer[frame_index].endRendering();
    
    // bloom pass
    vk::ImageMemoryBarrier bloom_barrier{
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = g_bloom_image,
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    if (g_enable_bloom){
      bloom_barrier.subresourceRange.baseMipLevel = 0;
      bloom_barrier.subresourceRange.levelCount = g_bloom_mip_levels;
      bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
      bloom_barrier.newLayout = vk::ImageLayout::eGeneral;
      bloom_barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
      bloom_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite;
      g_command_buffer[frame_index].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,vk::PipelineStageFlagBits::eFragmentShader,{},{},nullptr,bloom_barrier);
      bloom_barrier.subresourceRange.levelCount = 1;
      std::vector<vk::RenderingAttachmentInfo>bloom_attachment_infos;
      for(int i=0;i<g_bloom_mip_levels;++i){
        bloom_attachment_infos.emplace_back(
          vk::RenderingAttachmentInfo{
            .imageView = g_bloom_image_views[i],
            .imageLayout = vk::ImageLayout::eGeneral,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f}
          }
        );
      }
      vk::RenderingInfo bloom_rendering_info = rendering_info;
      bloom_rendering_info.colorAttachmentCount = static_cast<uint32_t>(bloom_attachment_infos.size());
      bloom_rendering_info.pColorAttachments = bloom_attachment_infos.data();
      bloom_rendering_info.pDepthAttachment = nullptr;
      g_command_buffer[frame_index].beginRendering(bloom_rendering_info);
      g_command_buffer[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_bloom_upsample_pipeline_layout,0,*g_descriptor_sets[frame_index],nullptr);
      vk::Viewport bloom_viewport = viewport;
      g_command_buffer[frame_index].setScissor(0, scissor);
      std::vector<uint32_t>bloom_attachment_locations(g_bloom_mip_levels,vk::AttachmentUnused);
      int32_t mip_width = g_swapchain_extent.width, mip_height = g_swapchain_extent.height;
      std::vector<int32_t>bloom_widths, bloom_heights;
      bloom_widths.emplace_back(mip_width);
      bloom_heights.emplace_back(mip_height);
      g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_bloom_downsample_pipeline);
      BloomPushConstants bloom_push_constants{.bloom_mip_level = 0, .bloom_factor = 0.0f};
      for(uint32_t i=1;i<g_bloom_mip_levels;++i){
        mip_width = mip_width>1?mip_width/2:1;
        mip_height = mip_height>1?mip_height/2:1;
        bloom_widths.emplace_back(mip_width);
        bloom_heights.emplace_back(mip_height);
        bloom_viewport.width = mip_width;
        bloom_viewport.height = mip_height;
        g_command_buffer[frame_index].setViewport(0, bloom_viewport);
        bloom_push_constants.bloom_mip_level = i;
        g_command_buffer[frame_index].pushConstants<BloomPushConstants>(g_bloom_upsample_pipeline_layout,vk::ShaderStageFlagBits::eFragment,0,bloom_push_constants);
        bloom_attachment_locations[i]=0;
        g_command_buffer[frame_index].setRenderingAttachmentLocations(vk::RenderingAttachmentLocationInfo{
          .colorAttachmentCount = static_cast<uint32_t>(bloom_attachment_locations.size()),
          .pColorAttachmentLocations = bloom_attachment_locations.data(),
        });
        g_command_buffer[frame_index].draw(4,1,0,0);
        bloom_attachment_locations[i]=vk::AttachmentUnused;
        bloom_barrier.subresourceRange.baseMipLevel = i;
        bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
        bloom_barrier.newLayout = vk::ImageLayout::eGeneral;
        bloom_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        bloom_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        g_command_buffer[frame_index].pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,vk::PipelineStageFlagBits::eFragmentShader,vk::DependencyFlagBits::eByRegion,{},nullptr,bloom_barrier);
      }
      g_command_buffer[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, g_bloom_upsample_pipeline);
      bloom_push_constants.bloom_factor = kBloomRate / (kBloomRate + 1);
      for(int i=g_bloom_mip_levels-2;i>=0;--i){
        bloom_viewport.width = bloom_widths[i];
        bloom_viewport.height = bloom_heights[i];
        g_command_buffer[frame_index].setViewport(0, bloom_viewport);
        bloom_push_constants.bloom_mip_level = i;
        g_command_buffer[frame_index].pushConstants<BloomPushConstants>(g_bloom_upsample_pipeline_layout,vk::ShaderStageFlagBits::eFragment,0,bloom_push_constants);
        bloom_attachment_locations[i]=0;
        g_command_buffer[frame_index].setRenderingAttachmentLocations(vk::RenderingAttachmentLocationInfo{
          .colorAttachmentCount = static_cast<uint32_t>(bloom_attachment_locations.size()),
          .pColorAttachmentLocations = bloom_attachment_locations.data(),
        });
        g_command_buffer[frame_index].draw(4,1,0,0);
        bloom_attachment_locations[i]=vk::AttachmentUnused;
        bloom_barrier.subresourceRange.baseMipLevel = i;
        bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
        bloom_barrier.newLayout = vk::ImageLayout::eGeneral;
        bloom_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        bloom_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        g_command_buffer[frame_index].pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,vk::PipelineStageFlagBits::eFragmentShader,vk::DependencyFlagBits::eByRegion,{},nullptr,bloom_barrier);
      }
      g_command_buffer[frame_index].endRendering();
    }

    bloom_barrier.subresourceRange.baseMipLevel = 0;
    bloom_barrier.oldLayout = vk::ImageLayout::eGeneral;
    bloom_barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    bloom_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    bloom_barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    g_command_buffer[frame_index].pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,vk::PipelineStageFlagBits::eTransfer,{},{},nullptr,bloom_barrier);
    vk::ImageBlit image_blit{
      .srcSubresource = {vk::ImageAspectFlagBits::eColor,0,0,1},
      .srcOffsets = std::array{vk::Offset3D{0,0,0},vk::Offset3D{static_cast<int32_t>(g_swapchain_extent.width),static_cast<int32_t>(g_swapchain_extent.height),1}},
      .dstSubresource = {vk::ImageAspectFlagBits::eColor,0,0,1},
      .dstOffsets = std::array{vk::Offset3D{0,0,0},vk::Offset3D{static_cast<int32_t>(g_swapchain_extent.width),static_cast<int32_t>(g_swapchain_extent.height),1}},
    };
    g_command_buffer[frame_index].blitImage(
      g_bloom_image,vk::ImageLayout::eTransferSrcOptimal,g_swapchain_images[image_index],vk::ImageLayout::eTransferDstOptimal,image_blit,vk::Filter::eLinear
    );

    TransformImageLayout(
      image_index,
      frame_index,
      vk::ImageLayout::eTransferDstOptimal,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::AccessFlagBits2::eTransferWrite,
      vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::PipelineStageFlagBits2::eTransfer,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput
    );
    std::vector<vk::RenderingAttachmentInfo>imgui_attachment_infos{
      {
        .imageView = g_swapchain_image_views[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
      }
    };
    vk::RenderingInfo imgui_rendering_info = rendering_info;
    imgui_rendering_info.colorAttachmentCount = static_cast<uint32_t>(imgui_attachment_infos.size());
    imgui_rendering_info.pColorAttachments = imgui_attachment_infos.data();
    imgui_rendering_info.pDepthAttachment = nullptr;
    g_command_buffer[frame_index].beginRendering(imgui_rendering_info);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *g_command_buffer[frame_index]);
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
    for(uint32_t i=0;i<g_frame_in_flight;++i){
      g_present_complete_semaphore.emplace_back(vk::raii::Semaphore(g_device, vk::SemaphoreCreateInfo()));
      g_render_finished_semaphore.emplace_back(vk::raii::Semaphore(g_device, vk::SemaphoreCreateInfo()));
      g_draw_fence.emplace_back(vk::raii::Fence(g_device, {.flags = vk::FenceCreateFlagBits::eSignaled}));
    }
    vk::SemaphoreTypeCreateInfo timeline_type_info{
      .semaphoreType = vk::SemaphoreType::eTimeline,
      .initialValue = 0,
    };
    g_particle_compute_semaphore = vk::raii::Semaphore(g_device, {.pNext = &timeline_type_info});
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
    CreateColorResources();
    CreateDepthResources();
    CreateShadowmapResources();
    CreateGbufferResources();
    CreateBloomResources();
    UpdateDescriptorSets();
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

      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      ImGui::Begin("Variables");
      if (ImGui::BeginTable("VariablesTable", 2)) {
        ImGui::TableSetupColumn("##Col0", ImGuiTableColumnFlags_WidthFixed); 
        ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Roughness:");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat("##RoughnessSlider", &g_pbr_roughness, 0.0f, 1.0f, "%.2f");
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("F0:");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat("##F0Slider", &g_pbr_f0, 0.0f, 1.0f, "%.2f");
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Metallic:");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat("##MetallicSlider", &g_pbr_metallic, 0.0f, 1.0f, "%.2f");
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Light:");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat("##LightSlider", &g_light_intensity, 0.0f, 20.0f, "%.2f");
        ImGui::EndTable();
      }
      ImGui::Checkbox("SSAO", &g_enable_ssao);
      ImGui::Checkbox("Bloom", &g_enable_bloom);
      ImGui::End();
      ImGui::Render();

      if(next_draw_time < m_current_time && Tick()){
        next_draw_time = m_current_time+draw_internal;
      }
    }

    g_device.waitIdle();
  }
  bool Tick(){
    return CpuPrepareData()&&GpuPrepareData()&&DrawFrame();
  }
  bool CpuPrepareData(){
    UniformBufferObject ubo;
    glm::vec3 camera_pos{1.0f,1.0f,1.0f};
    ubo.modu = glm::rotate<float>(glm::mat4(1.0f),(m_current_time-m_start_time)*glm::radians(10.0f),glm::vec3(0.0f,0.0f,1.0f));
    // https://learnopengl.com/Getting-started/Coordinate-Systems
    // https://learnopengl.com/Getting-started/Camera
    ubo.view = glm::lookAt(camera_pos,glm::vec3(0.0f,0.0f,0.0f),glm::vec3(0.0f,0.0f,-1.0f));
    ubo.proj = glm::perspective<float>(glm::radians(90.0f),static_cast<float>(g_swapchain_extent.width) / static_cast<float>(g_swapchain_extent.height), 0.1f, 3.0f);
    ubo.light.pos = glm::vec3(1.0f,0.0f,2.0f);
    ubo.light.intensities = glm::vec3(g_light_intensity);
    ubo.camera_pos = camera_pos;
    ubo.light_view = glm::lookAt(ubo.light.pos,glm::vec3(0.0f,0.0f,0.0f),glm::vec3(0.0f,0.0f,-1.0f));
    ubo.light_proj = glm::perspective<float>(glm::radians(90.0f),static_cast<float>(g_shadowmap_width) / static_cast<float>(g_shadowmap_height), 0.8f, 3.0f);
    ubo.shadowmap_resolution = glm::vec2(g_shadowmap_width, g_shadowmap_height);
    ubo.shadowmap_scale = glm::vec2(g_shadowmap_width/(float)g_swapchain_extent.width,g_shadowmap_height/(float)g_swapchain_extent.height);
    memcpy(g_ubo_buffer_maped[m_frame_index], &ubo, sizeof(ubo));
    UpdateModel();
    return true;
  }
  void UpdateParticle(){
    ParticleUbo ubo;
    static double last_particle_update_time = 0.0f;
    if(last_particle_update_time == 0.0f){
      last_particle_update_time = m_current_time;
    }
    ubo.delta_time = m_current_time - last_particle_update_time;
    last_particle_update_time = m_current_time;
    memcpy(g_particle_ubo_buffer_maped[m_frame_index],&ubo,sizeof(ParticleUbo));

    uint32_t compute_cb_index = m_frame_index+g_frame_in_flight*2;
    g_command_buffer[compute_cb_index].reset();
    g_command_buffer[compute_cb_index].begin({});
    g_command_buffer[compute_cb_index].bindPipeline(vk::PipelineBindPoint::eCompute, g_compute_pipeline);
    g_command_buffer[compute_cb_index].bindDescriptorSets(vk::PipelineBindPoint::eCompute, g_compute_pipeline_layout,0,*g_descriptor_sets[m_frame_index],nullptr);
    g_command_buffer[compute_cb_index].dispatch(kParticleCount/256,1,1);
    g_command_buffer[compute_cb_index].end();
    vk::PipelineStageFlags compute_wait_dst_stage_mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    uint64_t wait_semaphore_value = g_particle_compute_count;
    uint64_t signal_semaphore_value = ++g_particle_compute_count;
    vk::TimelineSemaphoreSubmitInfo compute_semaphore_submit_info{
      .waitSemaphoreValueCount = 1,
      .pWaitSemaphoreValues = &wait_semaphore_value,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &signal_semaphore_value,
    };
    vk::SubmitInfo compute_submit_info{
      .pNext = &compute_semaphore_submit_info,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*g_particle_compute_semaphore,
      .pWaitDstStageMask = &compute_wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &*g_command_buffer[compute_cb_index],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &*g_particle_compute_semaphore,
    };
    g_queue.submit(compute_submit_info,nullptr);
  }
  bool GpuPrepareData(){
    UpdateParticle();
    return true;
  }
  bool DrawFrame(){
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
    vk::PipelineStageFlags wait_dst_stage_mask =  
      vk::PipelineStageFlagBits::eColorAttachmentOutput |
      vk::PipelineStageFlagBits::eVertexInput;
    std::vector<uint64_t>wait_values{g_particle_compute_count,1};
    vk::TimelineSemaphoreSubmitInfo graphics_semaphore_submit_info{
      .waitSemaphoreValueCount = 2,
      .pWaitSemaphoreValues = wait_values.data(),
      .signalSemaphoreValueCount = 0,
    };
    std::vector<vk::Semaphore> graphics_wait_semaphores{*g_particle_compute_semaphore,*g_present_complete_semaphore[m_frame_index]};
    vk::SubmitInfo submit_info{
      .pNext = &graphics_semaphore_submit_info,
      .waitSemaphoreCount = static_cast<uint32_t>(graphics_wait_semaphores.size()),
      .pWaitSemaphores = graphics_wait_semaphores.data(),
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
    m_frame_index = (m_frame_index+1)%g_frame_in_flight;
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
		ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
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
    InitImGui();
    worker.Run();
  }catch(const std::exception& e){
    std::cerr<<e.what()<<std::endl;
    return EXIT_FAILURE;
  }
  std::cout<<"stop"<<std::endl;
  return EXIT_SUCCESS;
}
