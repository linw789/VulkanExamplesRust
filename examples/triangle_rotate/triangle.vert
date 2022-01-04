#version 450

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_color;

layout (binding = 0) uniform UBO
{
    mat4 projective;
    mat4 view;
    mat4 model;
} ubo;

layout (location = 0) out vec3 out_color;

void main() 
{
    out_color = in_color;
    gl_Position = vec4(in_pos, 1.0);
}
