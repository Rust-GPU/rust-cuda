use glam::{DVec2, USizeVec2, Vec2, Vec3};
use glium::glutin::event::{
    ElementState, Event, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use path_tracer_kernels::Viewport;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub origin: Vec3,
    pub lookat: Vec3,
    pub vup: Vec3,
    pub fov: f32,
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn as_viewport(&self, viewport: &mut Viewport) {
        let theta = self.fov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = self.aspect_ratio * viewport_height;

        let w = (self.origin - self.lookat).normalize();
        let u = self.vup.cross(w).normalize();
        let v = w.cross(u);

        viewport.origin = self.origin;
        viewport.horizontal = viewport_width * u;
        viewport.vertical = viewport_height * v;
        viewport.lower_left = self.origin - viewport.horizontal / 2.0 - viewport.vertical / 2.0 - w;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraController {
    pub sensitivity: f32,
    last_mouse_pos: Vec2,
    yaw: f32,
    pitch: f32,
    mousewheel_pressed: bool,
    shift_pressed: bool,
    first: bool,
}

impl CameraController {
    pub fn new(dimensions: USizeVec2) -> Self {
        CameraController {
            sensitivity: 0.1,
            last_mouse_pos: dimensions.as_vec2() / 2.0,
            yaw: -90.0,
            pitch: 0.0,
            mousewheel_pressed: false,
            shift_pressed: false,
            first: true,
        }
    }

    pub fn process_event(&mut self, event: &Event<()>, camera: &mut Camera) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CursorMoved { position, .. } => {
                    let mouse_pos = DVec2::new(position.x, position.y).as_vec2();
                    let delta = mouse_pos - self.last_mouse_pos;
                    self.last_mouse_pos = mouse_pos;

                    if !self.mousewheel_pressed {
                        return false;
                    }

                    if self.first {
                        self.last_mouse_pos = mouse_pos;
                        self.first = false;
                    }

                    if self.shift_pressed {
                        let change = Vec2::new(-delta.x, delta.y) * self.sensitivity * 0.05;
                        camera.lookat += change.extend(0.0);
                        camera.origin += change.extend(0.0);
                        return true;
                    }

                    self.yaw += delta.x * self.sensitivity;
                    self.pitch += delta.y * self.sensitivity;
                    self.pitch = self.pitch.min(89.0).max(-89.0);

                    let yaw_rad = self.yaw.to_radians();
                    let pitch_rad = self.pitch.to_radians();

                    let dir = Vec3::new(
                        yaw_rad.cos() * pitch_rad.cos(),
                        pitch_rad.sin(),
                        yaw_rad.sin() * pitch_rad.cos(),
                    );
                    camera.origin = camera.lookat + dir;
                    true
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let zoom = match delta {
                        MouseScrollDelta::LineDelta(x, y) => Vec2::new(*x, *y),
                        _ => panic!(),
                    };

                    camera.fov -= zoom.y;
                    camera.fov = camera.fov.clamp(1.0, 60.0);
                    true
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if *button == MouseButton::Middle {
                        self.mousewheel_pressed = *state == ElementState::Pressed;
                    }
                    false
                }
                WindowEvent::Resized(size) => {
                    camera.aspect_ratio = size.width as f32 / size.height as f32;
                    true
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if input.virtual_keycode == Some(VirtualKeyCode::LShift) {
                        self.shift_pressed = input.state == ElementState::Pressed;
                    }
                    false
                }
                _ => false,
            },
            _ => false,
        }
    }
}
