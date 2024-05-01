#version 450

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec2 i_position;
layout(location = 1) in vec3 i_color;
layout(location = 2) in vec2 i_texcoord;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_texcoord;

void main() {
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(i_position, 0.0, 1.0);
    v_color = i_color;
    v_texcoord = i_texcoord;
}