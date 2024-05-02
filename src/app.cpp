#include "app.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <array>
#include <optional>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include "math.hpp"
#include "transformation.hpp"
#include "stb_image.h"

#define RequireOk(ok, msg) if (!(ok)) throw std::runtime_error(msg)
#define ShowVar(x) std::cout << #x << " = " << x << '\n'

constexpr static std::array<char const*, 1> s_validation_layers =
{
	"VK_LAYER_KHRONOS_validation"
};

struct Vertex
{
	maya::Fvec2 pos;
	maya::Fvec3 color;
	maya::Fvec2 texture_coord;
};

static const std::vector<Vertex> s_vertices =
{
	{{-0.5f, -0.5f},	{1.0f, 0.0f, 0.0f},		{1.0f, 0.0f}},
	{{0.5f, -0.5f},		{0.0f, 1.0f, 0.0f},		{0.0f, 0.0f}},
	{{0.5f, 0.5f},		{0.0f, 0.0f, 1.0f},		{0.0f, 1.0f}},
	{{-0.5f, 0.5f},		{1.0f, 1.0f, 1.0f},		{1.0f, 1.0f}}
};

static const std::vector<uint16_t> s_indices =
{
	0, 1, 2, 2, 3, 0
};

void VulkanApp::Create()
{
	InitWindow();
	InitVulkan();
	InitDebugMessenger();
	InitSurface();
	InitDevice();
	InitSwapchain();
	InitImageViews();
	InitRenderPass();
	InitDescriptorSetLayout();
	InitGraphicsPipeline();
	InitFrameBuffers();
	InitCommandPool();
	InitSyncObjects();
	InitCommandBuffers();
	InitVertexBuffer();
	InitIndexBuffer();
	InitUniformBuffers();
	InitTextureImage();
	InitTextureImageView();
	InitTextureSampler();
	InitDescriptorPool();
	InitDescriptorSets();
}

void VulkanApp::Loop()
{
	uint32_t current_frame = 0;

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		RenderFrame(current_frame);
		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	vkDeviceWaitIdle(logical_device);
}

void VulkanApp::Destroy()
{
	FreeDescriptorSets();
	FreeDescriptorPool();
	FreeTextureSampler();
	FreeTextureImageView();
	FreeTextureImage();
	FreeUniformBuffers();
	FreeIndexBuffer();
	FreeVertexBuffer();
	FreeCommandBuffers();
	FreeSyncObjects();
	FreeCommandPool();
	FreeFrameBuffers();
	FreeGraphicsPipeline();
	FreeDescriptorSetLayout();
	FreeRenderPass();
	FreeImageViews();
	FreeSwapchain();
	FreeDevice();
	FreeSurface();
	FreeDebugMessenger();
	FreeVulkan();
	FreeWindow();
}

static void s_FramebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	bool& is_framebuffer_resized = *reinterpret_cast<bool*>(glfwGetWindowUserPointer(window));
	is_framebuffer_resized = true;
}

void VulkanApp::InitWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(1600, 900, "Vulkan Example", 0, 0);
	RequireOk(window, "Cannot create window.");
	is_framebuffer_resized = false;
	glfwSetWindowUserPointer(window, &is_framebuffer_resized);
	glfwSetFramebufferSizeCallback(window, s_FramebufferResizeCallback);
}

void VulkanApp::FreeWindow()
{
	glfwDestroyWindow(window);
	glfwTerminate();
}

#ifndef NDEBUG

static VKAPI_ATTR VkBool32 VKAPI_CALL s_DebugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	bool require_attention = messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;

	if (require_attention) std::cout << "\n";
	std::cout << "[Vulkan] ";
	if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		std::cout << "\033[91m";
	else if (require_attention)
		std::cout << "\033[93m";
	std::cout << pCallbackData->pMessage << "\n\033[0m";
	if (require_attention) std::cout << "\n";

	return VK_FALSE;
}

static inline void s_AssertValidationLayers(char const* const* layers, size_t layer_count)
{
	uint32_t count;
	vkEnumerateInstanceLayerProperties(&count, nullptr);
	std::vector<VkLayerProperties> available(count);
	vkEnumerateInstanceLayerProperties(&count, available.data());

	for (int i = 0; i < layer_count; i++)
	{
		bool found = false;

		for (const auto& props : available) {
			if (std::strcmp(layers[i], props.layerName) == 0) {
				found = true;
				break;
			}
		}

		RequireOk(found, "Cannot find validation layer.");
	}
}

static constexpr VkDebugUtilsMessengerCreateInfoEXT s_CreateVkDebugUtilsMessengerCreateInfoEXT()
{
	VkDebugUtilsMessengerCreateInfoEXT createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = s_DebugMessageCallback;
	createInfo.pUserData = nullptr;
	return createInfo;
}

#endif

void VulkanApp::InitVulkan()
{
	VkApplicationInfo app_info{};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "Vulkan Example";
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName = "No Engine";
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.apiVersion = VK_API_VERSION_1_3;

	VkInstanceCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create_info.pApplicationInfo = &app_info;

	uint32_t glfw_ext_count = 0;
	const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

#ifndef NDEBUG

	s_AssertValidationLayers(s_validation_layers.data(), s_validation_layers.size());
	create_info.enabledLayerCount = static_cast<uint32_t>(s_validation_layers.size());
	create_info.ppEnabledLayerNames = s_validation_layers.data();

	std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	create_info.ppEnabledExtensionNames = extensions.data();
	auto dci = s_CreateVkDebugUtilsMessengerCreateInfoEXT();
	create_info.pNext = &dci;

#else

	create_info.enabledLayerCount = 0;
	createInfo.enabledExtensionCount = glfw_ext_count;
	createInfo.ppEnabledExtensionNames = glfw_exts;

#endif

	VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
	RequireOk(result == VK_SUCCESS, "Cannot create Vulkan instance.");
}

void VulkanApp::FreeVulkan()
{
	vkDestroyInstance(instance, nullptr);
}

void VulkanApp::InitDebugMessenger()
{
#ifndef NDEBUG
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	RequireOk(func, "Cannot create debug messenger.");
	auto info = s_CreateVkDebugUtilsMessengerCreateInfoEXT();
	func(instance, &info, nullptr, &debug_messenger);
#endif
}

void VulkanApp::FreeDebugMessenger()
{
#ifndef NDEBUG
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	RequireOk(func, "Cannot free debug messenger.");
	func(instance, debug_messenger, nullptr);
#endif
}

void VulkanApp::InitSurface()
{
	VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
	RequireOk(result == VK_SUCCESS, "Cannot create window surface.");
}

void VulkanApp::FreeSurface()
{
	vkDestroySurfaceKHR(instance, surface, nullptr);
}

void VulkanApp::InitDevice()
{
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	RequireOk(count, "No usable physical device found.");
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(instance, &count, devices.data());

	constexpr std::array<char const*, 1> device_extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	for (auto device : devices)
	{
		VkPhysicalDeviceFeatures supported_features;
		vkGetPhysicalDeviceFeatures(device, &supported_features);
		if (!supported_features.samplerAnisotropy) continue;

		uint32_t ext_count;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
		std::vector<VkExtensionProperties> available_exts(ext_count);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, available_exts.data());

		bool ext_ok = 1;
		for (auto req : device_extensions) {
			ext_ok = false;
			for (auto& ae : available_exts) {
				if (std::strcmp(req, ae.extensionName) == 0) {
					ext_ok = true;
					break;
				}
			}
			if (!ext_ok)
				break;
		}

		if (!ext_ok) continue;

		uint32_t sfcount, spmcount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &sfcount, nullptr);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &spmcount, nullptr);
		if (!sfcount || !spmcount) continue;

		uint32_t fc = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &fc, nullptr);
		std::vector<VkQueueFamilyProperties> families(fc);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &fc, families.data());

		std::optional<uint32_t> graphics;
		std::optional<uint32_t> present;

		for (int i = 0; i < families.size(); i++)
		{
			if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics = i;

			VkBool32 present_support;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
			if (present_support) present = i;

			if (graphics.has_value() && present.has_value()) {
				physical_device = device;
				graphics_queue.indices = graphics;
				present_queue.indices = present;
				goto physical_device_ok;
			}
		}
	}

	RequireOk(false, "Cannot find a suitable physical device.");

physical_device_ok:

	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
	float priority = 1.0f;
	std::unordered_set<uint32_t> index_uset = {
		graphics_queue.indices.value(),
		present_queue.indices.value()
	};

	for (uint32_t index : index_uset)
	{
		VkDeviceQueueCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		info.queueFamilyIndex = index;
		info.queueCount = 1;
		info.pQueuePriorities = &priority;
		queue_create_infos.push_back(info);
	}

	VkPhysicalDeviceFeatures device_features{};
	device_features.samplerAnisotropy = VK_TRUE;

	VkDeviceCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	create_info.pQueueCreateInfos = queue_create_infos.data();
	create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	create_info.pEnabledFeatures = &device_features;
	create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
	create_info.ppEnabledExtensionNames = device_extensions.data();

#ifndef NDEBUG
	create_info.enabledLayerCount = static_cast<uint32_t>(s_validation_layers.size());
	create_info.ppEnabledLayerNames = s_validation_layers.data();
#else
	create_info.enabledLayerCount = 0;
#endif

	VkResult result = vkCreateDevice(physical_device, &create_info, nullptr, &logical_device);
	RequireOk(result == VK_SUCCESS, "Cannot create logical device.");

	vkGetDeviceQueue(logical_device, graphics_queue.indices.value(), 0, &graphics_queue.queue);
	vkGetDeviceQueue(logical_device, present_queue.indices.value(), 0, &present_queue.queue);
}

void VulkanApp::FreeDevice()
{
	vkDestroyDevice(logical_device, nullptr);
}

static VkSurfaceFormatKHR s_ChooseSurfaceFormat(VkPhysicalDevice device, VkSurfaceKHR surface)
{
	std::vector<VkSurfaceFormatKHR> formats;
	uint32_t count;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
	formats.resize(count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, formats.data());

	for (auto const& format : formats) {
		if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			return format;
	}

	return formats[0];
}

static VkPresentModeKHR s_ChoosePresentModes(VkPhysicalDevice device, VkSurfaceKHR surface)
{
	std::vector<VkPresentModeKHR> modes;
	uint32_t count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
	modes.resize(count);
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, modes.data());

	for (auto const& mode : modes) {
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
			return mode;
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D s_ChooseSwapExtent(GLFWwindow* window, VkSurfaceCapabilitiesKHR const& caps)
{
	if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
		return caps.currentExtent;

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	VkExtent2D actual = {
		static_cast<uint32_t>(width),
		static_cast<uint32_t>(height)
	};

	actual.width = std::clamp(actual.width, caps.minImageExtent.width, caps.maxImageExtent.width);
	actual.height = std::clamp(actual.height, caps.minImageExtent.height, caps.maxImageExtent.height);

	return actual;
}

void VulkanApp::InitSwapchain()
{
	VkSurfaceCapabilitiesKHR caps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps);

	VkSurfaceFormatKHR format = s_ChooseSurfaceFormat(physical_device, surface);
	VkPresentModeKHR present_mode = s_ChoosePresentModes(physical_device, surface);
	VkExtent2D extent = s_ChooseSwapExtent(window, caps);

	uint32_t image_count = caps.minImageCount + 1;
	if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
		image_count = caps.maxImageCount;

	VkSwapchainCreateInfoKHR create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	create_info.surface = surface;
	create_info.minImageCount = image_count;
	create_info.imageFormat = format.format;
	create_info.imageColorSpace = format.colorSpace;
	create_info.imageExtent = extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queue_family_indices[] = { graphics_queue.indices.value(), present_queue.indices.value() };

	if (graphics_queue.indices != present_queue.indices)
	{
		create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		create_info.queueFamilyIndexCount = 2;
		create_info.pQueueFamilyIndices = queue_family_indices;
	}
	else
	{
		create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		create_info.queueFamilyIndexCount = 0;
		create_info.pQueueFamilyIndices = nullptr;
	}

	create_info.preTransform = caps.currentTransform;
	create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	create_info.clipped = VK_TRUE;
	create_info.presentMode = present_mode;
	create_info.clipped = VK_TRUE;
	create_info.oldSwapchain = VK_NULL_HANDLE;

	VkResult ok = vkCreateSwapchainKHR(logical_device, &create_info, nullptr, &swapchain);
	RequireOk(ok == VK_SUCCESS, "Cannot create swapchain.");
	swapchain_image_format = format.format;
	swapchain_extent = extent;
}

void VulkanApp::FreeSwapchain()
{
	vkDestroySwapchainKHR(logical_device, swapchain, nullptr);
}

static VkImageView s_CreateImageView(VkDevice ld, VkImage image, VkFormat format)
{
	VkImageViewCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	create_info.image = image;
	create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	create_info.format = format;
	create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	create_info.subresourceRange.baseMipLevel = 0;
	create_info.subresourceRange.levelCount = 1;
	create_info.subresourceRange.baseArrayLayer = 0;
	create_info.subresourceRange.layerCount = 1;

	VkImageView iv;
	VkResult r = vkCreateImageView(ld, &create_info, nullptr, &iv);
	RequireOk(r == VK_SUCCESS, "Cannot create image view.");
	return iv;
}

void VulkanApp::InitImageViews()
{
	std::vector<VkImage> images;
	uint32_t imcount;
	vkGetSwapchainImagesKHR(logical_device, swapchain, &imcount, nullptr);
	images.resize(imcount);
	vkGetSwapchainImagesKHR(logical_device, swapchain, &imcount, images.data());
	swapchain_image_views.resize(imcount);

	for (uint32_t i = 0; i < imcount; i++)
		swapchain_image_views[i] = s_CreateImageView(logical_device, images[i], swapchain_image_format);

}

void VulkanApp::FreeImageViews()
{
	for (auto view : swapchain_image_views)
		vkDestroyImageView(logical_device, view, nullptr);
}

void VulkanApp::InitRenderPass()
{
	VkAttachmentDescription attach{};
	attach.format = swapchain_image_format;
	attach.samples = VK_SAMPLE_COUNT_1_BIT;
	attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference attach_ref{};
	attach_ref.attachment = 0;
	attach_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &attach_ref;

	VkRenderPassCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	create_info.attachmentCount = 1;
	create_info.pAttachments = &attach;
	create_info.subpassCount = 1;
	create_info.pSubpasses = &subpass;

	VkResult result = vkCreateRenderPass(logical_device, &create_info, nullptr, &render_pass);
	RequireOk(result == VK_SUCCESS, "Cannot create render pass.");
}

void VulkanApp::FreeRenderPass()
{
	vkDestroyRenderPass(logical_device, render_pass, nullptr);
}

void VulkanApp::InitDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding ubo_layout{};
	ubo_layout.binding = 0;
	ubo_layout.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	ubo_layout.descriptorCount = 1;
	ubo_layout.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	ubo_layout.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding sampler_layout{};
	sampler_layout.binding = 1;
	sampler_layout.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	sampler_layout.descriptorCount = 1;
	sampler_layout.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	sampler_layout.pImmutableSamplers = nullptr;

	auto bindings = std::array { ubo_layout, sampler_layout };

	VkDescriptorSetLayoutCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	ci.bindingCount = static_cast<uint32_t>(bindings.size());
	ci.pBindings = bindings.data();

	VkResult r = vkCreateDescriptorSetLayout(logical_device, &ci, nullptr, &descriptor_set_layout);
	RequireOk(r == VK_SUCCESS, "Cannot create descriptor set layout.");
}

void VulkanApp::FreeDescriptorSetLayout()
{
	vkDestroyDescriptorSetLayout(logical_device, descriptor_set_layout, nullptr);
}

static std::vector<char> s_ReadShaderSpirV(char const* path)
{
	std::ifstream ifs(path, std::ios::ate | std::ios::binary);
	RequireOk(ifs.is_open(), "Cannot open SPIR-V shader file.");
	size_t fsize = ifs.tellg();
	std::vector<char> buffer(fsize);
	ifs.seekg(0);
	ifs.read(buffer.data(), fsize);
	ifs.close();
	return buffer;
}

static VkShaderModule s_CreateShaderModule(VkDevice device, std::vector<char> const& data)
{
	VkShaderModuleCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	create_info.codeSize = data.size();
	create_info.pCode = reinterpret_cast<const uint32_t*>(data.data());

	VkShaderModule smodule;
	VkResult result = vkCreateShaderModule(device, &create_info, nullptr, &smodule);
	RequireOk(result == VK_SUCCESS, "Cannot create shader module.");
	return smodule;
}

static VkVertexInputBindingDescription s_GetVertexBindingDescription()
{
	VkVertexInputBindingDescription desc{};
	desc.binding = 0;
	desc.stride = sizeof(Vertex);
	desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	return desc;
}

static auto s_GetVertexAttributeDescriptions()
{
	std::array<VkVertexInputAttributeDescription, 3> descs{};

	descs[0].binding = 0;
	descs[0].location = 0;
	descs[0].format = VK_FORMAT_R32G32_SFLOAT;
	descs[0].offset = offsetof(Vertex, pos);

	descs[1].binding = 0;
	descs[1].location = 1;
	descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	descs[1].offset = offsetof(Vertex, color);

	descs[2].binding = 0;
	descs[2].location = 2;
	descs[2].format = VK_FORMAT_R32G32_SFLOAT;
	descs[2].offset = offsetof(Vertex, texture_coord);

	return descs;
}

void VulkanApp::InitGraphicsPipeline()
{
	auto vert_src = s_ReadShaderSpirV(SHADERS_BIN_DIR "vert.spv");
	auto frag_src = s_ReadShaderSpirV(SHADERS_BIN_DIR "frag.spv");

	VkShaderModule vert = s_CreateShaderModule(logical_device, vert_src);
	VkShaderModule frag = s_CreateShaderModule(logical_device, frag_src);

	VkPipelineShaderStageCreateInfo vert_stage{};
	vert_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vert_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vert_stage.module = vert;
	vert_stage.pName = "main";

	VkPipelineShaderStageCreateInfo frag_stage{};
	frag_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	frag_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	frag_stage.module = frag;
	frag_stage.pName = "main";

	VkPipelineShaderStageCreateInfo shader_stages[] = { vert_stage, frag_stage };

	auto binding_desc = s_GetVertexBindingDescription();
	auto attr_descs = s_GetVertexAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vii{};
	vii.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vii.vertexBindingDescriptionCount = 1;
	vii.pVertexBindingDescriptions = &binding_desc;
	vii.vertexAttributeDescriptionCount = static_cast<uint32_t>(attr_descs.size());
	vii.pVertexAttributeDescriptions = attr_descs.data();

	VkPipelineInputAssemblyStateCreateInfo ia{};
	ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	ia.primitiveRestartEnable = VK_FALSE;

	VkPipelineViewportStateCreateInfo vstate{};
	vstate.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	vstate.viewportCount = 1;
	vstate.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f; // Optional
	rasterizer.depthBiasClamp = 0.0f; // Optional
	rasterizer.depthBiasSlopeFactor = 0.0f;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f; // Optional
	multisampling.pSampleMask = nullptr; // Optional
	multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
	multisampling.alphaToOneEnable = VK_FALSE; // Optional

	VkPipelineColorBlendAttachmentState blend_attach{};
	blend_attach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blend_attach.blendEnable = VK_TRUE;
	blend_attach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	blend_attach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	blend_attach.colorBlendOp = VK_BLEND_OP_ADD;
	blend_attach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	blend_attach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	blend_attach.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo blend{};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.logicOpEnable = VK_FALSE;
	blend.logicOp = VK_LOGIC_OP_COPY; // Optional
	blend.attachmentCount = 1;
	blend.pAttachments = &blend_attach;
	blend.blendConstants[0] = 0.0f; // Optional
	blend.blendConstants[1] = 0.0f; // Optional
	blend.blendConstants[2] = 0.0f; // Optional
	blend.blendConstants[3] = 0.0f; // Optional

	VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	VkPipelineDynamicStateCreateInfo dstate{};
	dstate.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dstate.dynamicStateCount = sizeof(dynamic_states) / sizeof(VkDynamicState);
	dstate.pDynamicStates = dynamic_states;

	VkPipelineLayoutCreateInfo pli{};
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &descriptor_set_layout;
	pli.pushConstantRangeCount = 0; // Optional
	pli.pPushConstantRanges = nullptr; // Optional

	VkResult pr = vkCreatePipelineLayout(logical_device, &pli, nullptr, &pipeline_layout);
	RequireOk(pr == VK_SUCCESS, "Cannot create pipeline layout.");

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shader_stages;
	pipelineInfo.pVertexInputState = &vii;
	pipelineInfo.pInputAssemblyState = &ia;
	pipelineInfo.pViewportState = &vstate;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &blend;
	pipelineInfo.pDynamicState = &dstate;
	pipelineInfo.layout = pipeline_layout;
	pipelineInfo.renderPass = render_pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	if (vkCreateGraphicsPipelines(logical_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphics_pipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	vkDestroyShaderModule(logical_device, vert, nullptr);
	vkDestroyShaderModule(logical_device, frag, nullptr);
}

void VulkanApp::FreeGraphicsPipeline()
{
	vkDestroyPipeline(logical_device, graphics_pipeline, nullptr);
	vkDestroyPipelineLayout(logical_device, pipeline_layout, nullptr);
}

void VulkanApp::InitFrameBuffers()
{
	swapchain_framebuffers.resize(swapchain_image_views.size());

	for (int i = 0; i < swapchain_framebuffers.size(); i++)
	{
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = render_pass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = &swapchain_image_views[i];
		framebufferInfo.width = swapchain_extent.width;
		framebufferInfo.height = swapchain_extent.height;
		framebufferInfo.layers = 1;

		VkResult result = vkCreateFramebuffer(logical_device, &framebufferInfo, nullptr, &swapchain_framebuffers[i]);
		RequireOk(result == VK_SUCCESS, "Cannot create framebuffers.");
	}
}

void VulkanApp::FreeFrameBuffers()
{
	for (auto framebuffer : swapchain_framebuffers)
		vkDestroyFramebuffer(logical_device, framebuffer, nullptr);
}

void VulkanApp::InitCommandPool()
{
	VkCommandPoolCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	create_info.queueFamilyIndex = graphics_queue.indices.value();

	VkResult result = vkCreateCommandPool(logical_device, &create_info, nullptr, &command_pool);
	RequireOk(result == VK_SUCCESS, "Cannot create command pool.");
}

void VulkanApp::FreeCommandPool()
{
	vkDestroyCommandPool(logical_device, command_pool, nullptr);
}

void VulkanApp::InitSyncObjects()
{
	VkSemaphoreCreateInfo si{};
	si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkFenceCreateInfo fi{};
	fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		uint32_t result = vkCreateSemaphore(logical_device, &si, nullptr, &image_available_semaphores[i])
			| vkCreateSemaphore(logical_device, &si, nullptr, &render_finished_semaphores[i])
			| vkCreateFence(logical_device, &fi, nullptr, &in_flight_fences[i]);

		RequireOk(result == VK_SUCCESS, "Cannot create semaphore or fence.");
	}
}

void VulkanApp::FreeSyncObjects()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
		vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
		vkDestroyFence(logical_device, in_flight_fences[i], nullptr);
	}
}

void VulkanApp::InitCommandBuffers()
{
	VkCommandBufferAllocateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	info.commandPool = command_pool;
	info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());

	VkResult result = vkAllocateCommandBuffers(logical_device, &info, command_buffers.data());
	RequireOk(result == VK_SUCCESS, "Cannot create command buffer.");
}

void VulkanApp::FreeCommandBuffers()
{
	vkFreeCommandBuffers(logical_device, command_pool, static_cast<uint32_t>(command_buffers.size()), command_buffers.data());
}

void VulkanApp::RecordCommandBuffer(uint32_t crnt, uint32_t image_index)
{
	auto cb = command_buffers[crnt];

	VkCommandBufferBeginInfo begin{};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = 0; // Optional
	begin.pInheritanceInfo = nullptr; // Optional
	VkResult begin_ok = vkBeginCommandBuffer(cb, &begin);
	RequireOk(begin_ok == VK_SUCCESS, "Cannot begin command buffer.");

	VkRenderPassBeginInfo rpi{};
	rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpi.renderPass = render_pass;
	rpi.framebuffer = swapchain_framebuffers[image_index];
	rpi.renderArea.offset = { 0, 0 };
	rpi.renderArea.extent = swapchain_extent;

	VkClearValue clear_color = { {{0.07f, 0.13f, 0.17f, 1.0f}} };
	rpi.clearValueCount = 1;
	rpi.pClearValues = &clear_color;
	vkCmdBeginRenderPass(cb, &rpi, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_sets[crnt], 0, nullptr);
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)swapchain_extent.width;
	viewport.height = (float)swapchain_extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cb, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = swapchain_extent;
	vkCmdSetScissor(cb, 0, 1, &scissor);

	VkBuffer buffers[] = { vertex_buffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(cb, 0, 1, buffers, offsets);
	vkCmdBindIndexBuffer(cb, index_buffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdDrawIndexed(cb, static_cast<uint32_t>(s_indices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(cb);

	VkResult end_ok = vkEndCommandBuffer(cb);
	RequireOk(end_ok == VK_SUCCESS, "Cannot record command buffer.");
}

void VulkanApp::UpdateSwapchain()
{
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(logical_device);

	FreeFrameBuffers();
	FreeImageViews();
	FreeSwapchain();

	InitSwapchain();
	InitImageViews();
	InitFrameBuffers();
}

void VulkanApp::RenderFrame(uint32_t crnt)
{
	uint32_t index;
	vkWaitForFences(logical_device, 1, &in_flight_fences[crnt], VK_TRUE, UINT64_MAX);

	VkResult result = vkAcquireNextImageKHR(logical_device, swapchain, UINT64_MAX, image_available_semaphores[crnt], VK_NULL_HANDLE, &index);

	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		UpdateSwapchain();
		return;
	}
	
	RequireOk(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR, "Cannot acquire swapchain image.");

	vkResetFences(logical_device, 1, &in_flight_fences[crnt]);

	UpdateUniformBuffer(crnt);

	vkResetCommandBuffer(command_buffers[crnt], 0);
	RecordCommandBuffer(crnt, index);

	VkSubmitInfo submit_info{};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore wait_semaphores[] = { image_available_semaphores[crnt]};
	VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = wait_semaphores;
	submit_info.pWaitDstStageMask = wait_stages;

	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffers[crnt];

	VkSemaphore signal_semaphores[] = { render_finished_semaphores[crnt] };
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = signal_semaphores;

	VkResult submit_ok = vkQueueSubmit(graphics_queue.queue, 1, &submit_info, in_flight_fences[crnt]);
	RequireOk(submit_ok == VK_SUCCESS, "Cannot submit draw command buffer.");

	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = signal_semaphores;

	VkSwapchainKHR swapchains[] = { swapchain };
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swapchains;
	present_info.pImageIndices = &index;

	result = vkQueuePresentKHR(present_queue.queue, &present_info);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || is_framebuffer_resized) {
		UpdateSwapchain();
		is_framebuffer_resized = false;
		return;
	}

	RequireOk(result == VK_SUCCESS, "Cannot present swapchain image.");
}

static uint32_t s_FindMemoryTypeIndex(VkPhysicalDevice pd, uint32_t filter, VkMemoryPropertyFlags flags)
{
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(pd, &mp);

	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
	{
		if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) {
			return i;
		}
	}

	RequireOk(false, "Cannot find a suitable memory type.");
}

static void s_CreateVulkanBuffer(VkPhysicalDevice pd, VkDevice ld,
					VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
					VkBuffer* buffer, VkDeviceMemory* mem)
{
	VkBufferCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size = size;
	info.usage = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult r1 = vkCreateBuffer(ld, &info, nullptr, buffer);
	RequireOk(r1 == VK_SUCCESS, "Cannot create vertex buffer.");

	VkMemoryRequirements mem_req;
	vkGetBufferMemoryRequirements(ld, *buffer, &mem_req);

	VkMemoryAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = mem_req.size;
	ai.memoryTypeIndex = s_FindMemoryTypeIndex(pd, mem_req.memoryTypeBits, props);

	VkResult r2 = vkAllocateMemory(ld, &ai, nullptr, mem);
	RequireOk(r2 == VK_SUCCESS, "Cannot allocate device memory.");

	vkBindBufferMemory(ld, *buffer, *mem, 0);
}

static VkCommandBuffer s_BeginSingleUseCommandBuffer(VkDevice ld, VkCommandPool pool)
{
	VkCommandBufferAllocateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	info.commandPool = pool;
	info.commandBufferCount = 1;

	VkCommandBuffer cb;
	vkAllocateCommandBuffers(ld, &info, &cb);

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cb, &bi);

	return cb;
}

static void s_EndAndExecuteSingleUseCommandBuffer(VkDevice ld, VkCommandPool pool, VkQueue q, VkCommandBuffer cb)
{
	vkEndCommandBuffer(cb);

	VkSubmitInfo si{};
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;

	vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(q);

	vkFreeCommandBuffers(ld, pool, 1, &cb);
}

static void s_CopyVulkanBuffer(VkDevice ld, VkCommandPool pool, VkQueue gq, VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
	VkCommandBuffer cb = s_BeginSingleUseCommandBuffer(ld, pool);

	VkBufferCopy copy{};
	copy.size = size;
	vkCmdCopyBuffer(cb, src, dst, 1, &copy);

	s_EndAndExecuteSingleUseCommandBuffer(ld, pool, gq, cb);
}

void VulkanApp::InitVertexBuffer()
{
	VkDeviceSize bsize = sizeof(Vertex) * s_vertices.size();

	VkBuffer staging_buffer;
	VkDeviceMemory staging_buffer_memory;

	s_CreateVulkanBuffer(physical_device, logical_device,
		bsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&staging_buffer, &staging_buffer_memory);

	void* data;
	vkMapMemory(logical_device, staging_buffer_memory, 0, bsize, 0, &data);
	std::memcpy(data, s_vertices.data(), (size_t)bsize);
	vkUnmapMemory(logical_device, staging_buffer_memory);

	s_CreateVulkanBuffer(physical_device, logical_device,
		bsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		&vertex_buffer, &vertex_buffer_memory);

	s_CopyVulkanBuffer(logical_device, command_pool, graphics_queue.queue, staging_buffer, vertex_buffer, bsize);

	vkDestroyBuffer(logical_device, staging_buffer, nullptr);
	vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
}

void VulkanApp::FreeVertexBuffer()
{
	vkDestroyBuffer(logical_device, vertex_buffer, nullptr);
	vkFreeMemory(logical_device, vertex_buffer_memory, nullptr);
}

void VulkanApp::InitIndexBuffer()
{
	VkDeviceSize bsize = sizeof(s_indices[0]) * s_indices.size();

	VkBuffer staging_buffer;
	VkDeviceMemory staging_buffer_memory;

	s_CreateVulkanBuffer(physical_device, logical_device,
		bsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&staging_buffer, &staging_buffer_memory);

	void* data;
	vkMapMemory(logical_device, staging_buffer_memory, 0, bsize, 0, &data);
	std::memcpy(data, s_indices.data(), (size_t)bsize);
	vkUnmapMemory(logical_device, staging_buffer_memory);

	s_CreateVulkanBuffer(physical_device, logical_device,
		bsize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		&index_buffer, &index_buffer_memory);

	s_CopyVulkanBuffer(logical_device, command_pool, graphics_queue.queue, staging_buffer, index_buffer, bsize);

	vkDestroyBuffer(logical_device, staging_buffer, nullptr);
	vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
}

void VulkanApp::FreeIndexBuffer()
{
	vkDestroyBuffer(logical_device, index_buffer, nullptr);
	vkFreeMemory(logical_device, index_buffer_memory, nullptr);
}

struct UniformBufferObject
{
	alignas(16) maya::Fmat4 model;
	alignas(16) maya::Fmat4 view;
	alignas(16) maya::Fmat4 projection;
};

void VulkanApp::InitUniformBuffers()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		s_CreateVulkanBuffer(physical_device, logical_device,
			sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniform_buffers[i], &uniform_buffers_memory[i]);

		vkMapMemory(logical_device, uniform_buffers_memory[i], 0, sizeof(UniformBufferObject), 0, &uniform_buffers_map[i]);
	}
}

void VulkanApp::FreeUniformBuffers()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyBuffer(logical_device, uniform_buffers[i], nullptr);
		vkFreeMemory(logical_device, uniform_buffers_memory[i], nullptr);
	}
}

void VulkanApp::UpdateUniformBuffer(uint32_t crnt)
{
	UniformBufferObject ubo;
	ubo.model = maya::RotateModel(static_cast<float>(glfwGetTime()) * 3.f, maya::Fvec3(0, 1, 0));
	ubo.view = maya::LookAtView(maya::Fvec3(0, 0.5f, -1), maya::Fvec3(0, -0.5, 1));
	ubo.projection = maya::PerspectiveProjection(static_cast<float>(maya::Pi) / 2.f, (float) swapchain_extent.width / swapchain_extent.height, 0.1f, 100.0f);
	ubo.projection[1][1] *= -1;

	std::memcpy(uniform_buffers_map[crnt], &ubo, sizeof(UniformBufferObject));
}

static void s_TransitionImageLayout(VkDevice ld, VkCommandPool pool, VkQueue q, VkImage image, VkFormat format, VkImageLayout oldl, VkImageLayout newl)
{
	VkCommandBuffer cb = s_BeginSingleUseCommandBuffer(ld, pool);

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldl;
	barrier.newLayout = newl;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags src_stage;
	VkPipelineStageFlags dst_stage;

	if (oldl == VK_IMAGE_LAYOUT_UNDEFINED && newl == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldl == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newl == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else {
		RequireOk(false, "Unsupporterd image layout transition.");
	}

	vkCmdPipelineBarrier(
		cb,
		src_stage, dst_stage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	s_EndAndExecuteSingleUseCommandBuffer(ld, pool, q, cb);
}

static void s_CopyVulkanBufferToImage(VkDevice ld, VkCommandPool pool, VkQueue gq, VkBuffer src, VkImage dst, uint32_t width, uint32_t height)
{
	VkCommandBuffer cb = s_BeginSingleUseCommandBuffer(ld, pool);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { width, height, 1 };

	vkCmdCopyBufferToImage(
		cb,
		src,
		dst,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);

	s_EndAndExecuteSingleUseCommandBuffer(ld, pool, gq, cb);
}

static void s_CreateVulkanImageTexture(VkPhysicalDevice pd, VkDevice ld,
	uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
	VkImageUsageFlags usage, VkMemoryPropertyFlags props,
	VkImage* image, VkDeviceMemory* mem)
{
	VkImageCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ci.imageType = VK_IMAGE_TYPE_2D;
	ci.extent.width = width;
	ci.extent.height = height;
	ci.extent.depth = 1;
	ci.mipLevels = 1;
	ci.arrayLayers = 1;
	ci.format = format;
	ci.tiling = tiling;
	ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ci.usage = usage;
	ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	ci.samples = VK_SAMPLE_COUNT_1_BIT;
	ci.flags = 0;

	VkResult r1 = vkCreateImage(ld, &ci, nullptr, image);
	RequireOk(r1 == VK_SUCCESS, "Cannot create texture image.");

	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(ld, *image, &req);

	VkMemoryAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = req.size;
	ai.memoryTypeIndex = s_FindMemoryTypeIndex(pd, req.memoryTypeBits, props);

	VkResult r2 = vkAllocateMemory(ld, &ai, nullptr, mem);
	RequireOk(r2 == VK_SUCCESS, "Cannot allocate texture image memory.");

	vkBindImageMemory(ld, *image, *mem, 0);
}

void VulkanApp::InitTextureImage()
{
	int width, height, channels;
	stbi_uc* pixels = stbi_load(TEXTURES_DIR "brick.png", &width, &height, &channels, STBI_rgb_alpha);
	VkDeviceSize imsize = width * height * 4;
	RequireOk(pixels, "Cannot load texture image.");

	VkBuffer staging_buffer;
	VkDeviceMemory staging_buffer_memory;

	s_CreateVulkanBuffer(physical_device, logical_device,
		imsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&staging_buffer, &staging_buffer_memory);

	void* data;
	vkMapMemory(logical_device, staging_buffer_memory, 0, imsize, 0, &data);
	std::memcpy(data, pixels, static_cast<size_t>(imsize));
	vkUnmapMemory(logical_device, staging_buffer_memory);

	stbi_image_free(pixels);

	s_CreateVulkanImageTexture(physical_device, logical_device, width, height,
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		&texture_image, &texture_image_memory);

	s_TransitionImageLayout(logical_device, command_pool, graphics_queue.queue, texture_image,
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	s_CopyVulkanBufferToImage(logical_device, command_pool, graphics_queue.queue, staging_buffer, texture_image, width, height);

	s_TransitionImageLayout(logical_device, command_pool, graphics_queue.queue, texture_image,
		VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(logical_device, staging_buffer, nullptr);
	vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
}

void VulkanApp::FreeTextureImage()
{
	vkDestroyImage(logical_device, texture_image, nullptr);
	vkFreeMemory(logical_device, texture_image_memory, nullptr);
}

void VulkanApp::InitTextureImageView()
{
	texture_image_view = s_CreateImageView(logical_device, texture_image, VK_FORMAT_R8G8B8A8_SRGB);
}

void VulkanApp::FreeTextureImageView()
{
	vkDestroyImageView(logical_device, texture_image_view, nullptr);
}

void VulkanApp::InitTextureSampler()
{
	VkSamplerCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	ci.magFilter = VK_FILTER_LINEAR;
	ci.minFilter = VK_FILTER_LINEAR;
	ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(physical_device, &props);
	ci.maxAnisotropy = props.limits.maxSamplerAnisotropy;

	ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	ci.unnormalizedCoordinates = VK_FALSE;
	ci.compareEnable = VK_FALSE;
	ci.compareOp = VK_COMPARE_OP_ALWAYS;
	ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	ci.mipLodBias = 0.0f;
	ci.minLod = 0.0f;
	ci.maxLod = 0.0f;

	VkResult r = vkCreateSampler(logical_device, &ci, nullptr, &texture_sampler);
	RequireOk(r == VK_SUCCESS, "Cannot create texture sampler.");
}

void VulkanApp::FreeTextureSampler()
{
	vkDestroySampler(logical_device, texture_sampler, nullptr);
}

void VulkanApp::InitDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 2> sizes{};
	sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	sizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
	sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	sizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

	VkDescriptorPoolCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
	ci.pPoolSizes = sizes.data();
	ci.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
	ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	VkResult r = vkCreateDescriptorPool(logical_device, &ci, nullptr, &descriptor_pool);
	RequireOk(r == VK_SUCCESS, "Cannot create descriptor pool.");
}

void VulkanApp::FreeDescriptorPool()
{
	vkDestroyDescriptorPool(logical_device, descriptor_pool, nullptr);
}

void VulkanApp::InitDescriptorSets()
{
	PER_FRAMES(VkDescriptorSetLayout) layouts;
	std::fill(layouts.begin(), layouts.end(), descriptor_set_layout);

	VkDescriptorSetAllocateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ci.descriptorPool = descriptor_pool;
	ci.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
	ci.pSetLayouts = layouts.data();

	VkResult r = vkAllocateDescriptorSets(logical_device, &ci, descriptor_sets.data());
	RequireOk(r == VK_SUCCESS, "Cannot allocate descriptor sets.");

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bi{};
		bi.buffer = uniform_buffers[i];
		bi.offset = 0;
		bi.range = sizeof(UniformBufferObject);

		VkDescriptorImageInfo ii{};
		ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		ii.imageView = texture_image_view;
		ii.sampler = texture_sampler;

		std::array<VkWriteDescriptorSet, 2> writes{};

		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = descriptor_sets[i];
		writes[0].dstBinding = 0;
		writes[0].dstArrayElement = 0;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writes[0].descriptorCount = 1;
		writes[0].pBufferInfo = &bi;

		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet = descriptor_sets[i];
		writes[1].dstBinding = 1;
		writes[1].dstArrayElement = 0;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[1].descriptorCount = 1;
		writes[1].pImageInfo = &ii;

		vkUpdateDescriptorSets(logical_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}
}

void VulkanApp::FreeDescriptorSets()
{
	vkFreeDescriptorSets(logical_device, descriptor_pool, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data());
}

