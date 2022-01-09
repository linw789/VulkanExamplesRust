#version 450

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_color;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

layout (location = 0) out vec3 out_color;

void main()
{
    /*
    float val = ubo.model[3][3] / 255.0;
    out_color = vec3(val, val, val);
    gl_Position = vec4(in_pos, 1.0);
    */

    out_color = in_color;

    mat4 transform = ubo.projection * ubo.view * ubo.model;
    gl_Position = transform * vec4(in_pos, 1.0);
    gl_Position.y = -gl_Position.y;
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
}
