#version 450

in vec3 pos;
out vec2 tex_coords;

void main() {
  gl_Position = vec4(pos, 1.0);
  tex_coords = (vec2(pos) / 2.0) + 0.5;
}