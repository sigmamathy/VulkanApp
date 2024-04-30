#include "app.hpp"

int main() {
	VulkanApp app;
	app.Create();
	app.Loop();
	app.Destroy();
}