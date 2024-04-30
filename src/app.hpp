#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <optional>
#include <vector>

class VulkanApp
{
public:

	void Create();
	void Loop();
	void Destroy();

private:

	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physical_device;
	VkDevice logical_device;

	VkSwapchainKHR swapchain;
	std::vector<VkImageView> image_views;
	VkFormat swapchain_image_format;
	VkExtent2D swapchain_extent;
	std::vector<VkFramebuffer> swapchain_framebuffers;

	struct QueueFamily {
		std::optional<uint32_t> indices;
		VkQueue queue;
	};

	QueueFamily graphics_queue;
	QueueFamily present_queue;

	VkRenderPass render_pass;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphics_pipeline;

	VkCommandPool command_pool;
	VkCommandBuffer command_buffer;

	VkSemaphore image_available_semaphore;
	VkSemaphore render_finished_semaphore;
	VkFence in_flight_fence;

	VkBuffer vertex_buffer;
	VkDeviceMemory vertex_buffer_memory;
	VkBuffer index_buffer;
	VkDeviceMemory index_buffer_memory;

private:

	void InitWindow();
	void FreeWindow();
	void InitVulkan();
	void FreeVulkan();
	void InitDebugMessenger();
	void FreeDebugMessenger();
	void InitSurface();
	void FreeSurface();
	void InitDevice();
	void FreeDevice();
	void InitSwapchain();
	void FreeSwapchain();
	void InitImageViews();
	void FreeImageViews();
	void InitRenderPass();
	void FreeRenderPass();
	void InitGraphicsPipeline();
	void FreeGraphicsPipeline();
	void InitFrameBuffers();
	void FreeFrameBuffers();
	void InitCommandPool();
	void FreeCommandPool();
	void InitCommandBuffer();
	void FreeCommandBuffer();
	void InitSyncObjects();
	void FreeSyncObjects();
	void InitVertexBuffer();
	void FreeVertexBuffer();
	void InitIndexBuffer();
	void FreeIndexBuffer();

	void RecordCommandBuffer(uint32_t imageIndex);
	void Render();
};