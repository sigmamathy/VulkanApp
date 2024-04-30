#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <optional>
#include <vector>
#include <array>

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
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphics_pipeline;

	static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
#define PER_FRAMES(t) std::array<t, MAX_FRAMES_IN_FLIGHT>

	VkCommandPool command_pool;
	PER_FRAMES(VkCommandBuffer) command_buffers;
	PER_FRAMES(VkSemaphore) image_available_semaphores;
	PER_FRAMES(VkSemaphore) render_finished_semaphores;
	PER_FRAMES(VkFence) in_flight_fences;

	VkBuffer vertex_buffer;
	VkDeviceMemory vertex_buffer_memory;
	VkBuffer index_buffer;
	VkDeviceMemory index_buffer_memory;

	PER_FRAMES(VkBuffer) uniform_buffers;
	PER_FRAMES(VkDeviceMemory) uniform_buffers_memory;
	PER_FRAMES(void*) uniform_buffers_map;

	VkDescriptorPool descriptor_pool;
	PER_FRAMES(VkDescriptorSet) descriptor_sets;

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
	void InitDescriptorSetLayout();
	void FreeDescriptorSetLayout();
	void InitGraphicsPipeline();
	void FreeGraphicsPipeline();
	void InitFrameBuffers();
	void FreeFrameBuffers();

	void InitCommandPool();
	void FreeCommandPool();
	void InitCommandBuffers();
	void FreeCommandBuffers();
	void InitSyncObjects();
	void FreeSyncObjects();

	void InitVertexBuffer();
	void FreeVertexBuffer();
	void InitIndexBuffer();
	void FreeIndexBuffer();
	void InitUniformBuffers();
	void FreeUniformBuffers();

	void InitDescriptorPool();
	void FreeDescriptorPool();
	void InitDescriptorSets();
	void FreeDescriptorSets();

	void RecordCommandBuffer(uint32_t crnt, uint32_t imageIndex);
	void RenderFrame(uint32_t crnt);
	void UpdateUniformBuffer(uint32_t crnt);
};