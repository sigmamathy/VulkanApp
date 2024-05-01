#version 450

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_texcoord;

layout(binding = 1) uniform sampler2D u_texsampler;

void main() {
    o_color = vec4(v_color * texture(u_texsampler, v_texcoord).rgb, 1.0);
}