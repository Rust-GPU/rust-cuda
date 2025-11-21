#version 450 

out vec4 color;
in vec2 tex_coords;

uniform sampler2D tex;

void main() {
  color = texture(tex, tex_coords);
}