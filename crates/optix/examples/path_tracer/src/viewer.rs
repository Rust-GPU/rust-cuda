use glam::USizeVec2;
use glium::{
    Display, Program, Rect, Surface, VertexBuffer,
    glutin::{
        ContextBuilder,
        dpi::PhysicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    },
    implement_vertex,
    index::{NoIndices, PrimitiveType},
    texture::{RawImage2d, SrgbTexture2d},
    uniform,
};

use imgui::Condition;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use path_tracer_kernels::scene::Scene;
use std::time::Instant;

use crate::{HEIGHT, WIDTH, common::Camera, renderer::Renderer};

static IMAGE_VERT: &str = include_str!("../shaders/image.vert");
static IMAGE_FRAG: &str = include_str!("../shaders/image.frag");

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vertex {
    pos: [f32; 3],
}

implement_vertex!(Vertex, pos);

// fullscreen quad
const VERTICES: &[Vertex] = &[
    Vertex {
        pos: [1.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-1.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-1.0, -1.0, 0.0],
    },
    Vertex {
        pos: [1.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-1.0, -1.0, 0.0],
    },
    Vertex {
        pos: [1.0, -1.0, 0.0],
    },
];

pub fn run(camera: &Camera, scene: &Scene) -> ! {
    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new()
        .with_title("Render")
        .with_inner_size(PhysicalSize::new(WIDTH as f64, HEIGHT as f64));
    let cb = ContextBuilder::new().with_vsync(true);
    let display = Display::new(wb, cb, &event_loop).unwrap();
    let renderer = Renderer::new(USizeVec2::new(WIDTH, HEIGHT), camera, scene);
    let mut viewer = ViewerRenderer::new(display, renderer);

    let mut last_frame = Instant::now();

    event_loop.run(move |ev, _, control_flow| {
        viewer.process_event(ev, control_flow, &mut last_frame);
    });
}

struct ViewerRenderer {
    vertex_buffer: VertexBuffer<Vertex>,
    image_program: Program,
    image_size: USizeVec2,
    renderer: Renderer,
    imgui_ctx: imgui::Context,
    texture: SrgbTexture2d,
    display: Display,
    imgui_renderer: imgui_glium_renderer::Renderer,
    platform: WinitPlatform,
}

impl ViewerRenderer {
    pub fn new(display: Display, renderer: Renderer) -> Self {
        let vertex_buffer = VertexBuffer::new(&display, VERTICES).unwrap();
        let image_program = Program::from_source(&display, IMAGE_VERT, IMAGE_FRAG, None).unwrap();

        let size = display.gl_window().window().inner_size();
        let image_size = USizeVec2::new(size.width as usize, size.height as usize);
        let texture = SrgbTexture2d::empty(&display, size.width, size.height).unwrap();

        let mut imgui_ctx = imgui::Context::create();
        imgui_ctx.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui_ctx);
        {
            let gl_window = display.gl_window();
            let window = gl_window.window();
            platform.attach_window(imgui_ctx.io_mut(), window, HiDpiMode::Rounded);
        }

        let hidpi_factor = platform.hidpi_factor();
        imgui_ctx.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        let imgui_renderer =
            imgui_glium_renderer::Renderer::init(&mut imgui_ctx, &display).unwrap();

        Self {
            vertex_buffer,
            image_program,
            image_size,
            renderer,
            imgui_ctx,
            texture,
            display,
            imgui_renderer,
            platform,
        }
    }

    pub fn process_event(
        &mut self,
        ev: Event<()>,
        control_flow: &mut ControlFlow,
        last_frame: &mut Instant,
    ) {
        self.renderer.process_event(&ev, control_flow);
        {
            let gl_window = self.display.gl_window();
            self.platform
                .handle_event(self.imgui_ctx.io_mut(), gl_window.window(), &ev);
        }

        #[allow(clippy::single_match)]
        match ev {
            Event::NewEvents(_) => {
                let now = Instant::now();
                self.imgui_ctx.io_mut().update_delta_time(now - *last_frame);
                *last_frame = now;
            }
            Event::MainEventsCleared => {
                let gl_window = self.display.gl_window();
                self.platform
                    .prepare_frame(self.imgui_ctx.io_mut(), gl_window.window())
                    .expect("Failed to prepare frame");
                gl_window.window().request_redraw();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(new) => {
                    let image_size = USizeVec2::new(new.width as usize, new.height as usize);
                    self.image_size = image_size;
                    self.texture = SrgbTexture2d::empty(
                        &self.display,
                        image_size.x as u32,
                        image_size.y as u32,
                    )
                    .unwrap();
                    self.renderer.resize(image_size);
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                self.render();
            }
            _ => {}
        }
    }

    pub fn render(&mut self) {
        let ViewerRenderer {
            vertex_buffer,
            image_program,
            image_size,
            renderer,
            texture,
            display,
            imgui_renderer,
            platform,
            ..
        } = self;
        let ui = self.imgui_ctx.frame();
        let out = ui
            .window("crab")
            .size([300.0, 300.0], Condition::FirstUseEver)
            .build(|| renderer.render(ui))
            .unwrap();

        let raw =
            RawImage2d::from_raw_rgb(out.to_vec(), (image_size.x as u32, image_size.y as u32));

        texture.write(
            Rect {
                left: 0,
                bottom: 0,
                width: image_size.x as u32,
                height: image_size.y as u32,
            },
            raw,
        );

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        let uniforms = uniform! {
            tex: &*texture
        };

        target
            .draw(
                &*vertex_buffer,
                NoIndices(PrimitiveType::TrianglesList),
                image_program,
                &uniforms,
                &Default::default(),
            )
            .unwrap();

        let gl_window = display.gl_window();
        platform.prepare_render(ui, gl_window.window());

        let draw_data = self.imgui_ctx.render();
        imgui_renderer.render(&mut target, draw_data).unwrap();
        target.finish().unwrap();
    }
}
