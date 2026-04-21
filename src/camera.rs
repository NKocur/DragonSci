use glam::{Mat4, Vec2, Vec3};

pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    /// Horizontal rotation in radians
    pub yaw: f32,
    /// Vertical rotation in radians, clamped away from poles
    pub pitch: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    /// When true, use an orthographic projection instead of perspective.
    pub parallel: bool,
}

/// Snapshot of camera state returned to / accepted from Python.
#[derive(Clone, Copy)]
pub struct CameraState {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub parallel: bool,
}

impl Camera {
    pub fn fit(center: Vec3, radius: f32, aspect: f32) -> Self {
        Self {
            target: center,
            distance: radius * 2.5,
            yaw: 0.4,
            pitch: 0.4,
            fov_y: 45_f32.to_radians(),
            aspect,
            near: radius * 0.001,
            far: radius * 100.0,
            parallel: false,
        }
    }

    pub fn position(&self) -> Vec3 {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        self.target + Vec3::new(cos_p * sin_y, sin_p, cos_p * cos_y) * self.distance
    }

    pub fn view_matrix(&self) -> Mat4 {
        // For exactly top/front/side views the default up=Y can be degenerate.
        // Pick Z as up when looking straight down/up.
        let up = if self.pitch.abs() > 1.5 { Vec3::Z } else { Vec3::Y };
        Mat4::look_at_rh(self.position(), self.target, up)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        if self.parallel {
            // Half-height in world units: matches the perspective frustum at distance.
            let half_h = self.distance * (self.fov_y * 0.5).tan();
            let half_w = half_h * self.aspect;
            Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, self.near, self.far)
        } else {
            Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
        }
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw += delta.x * 0.008;
        self.pitch = (self.pitch - delta.y * 0.008).clamp(-1.55, 1.55);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.12)).max(self.near * 10.0);
    }

    /// Pan in the camera's local XY plane
    pub fn pan(&mut self, delta: Vec2) {
        let view = self.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.x_axis.y, view.x_axis.z);
        let up    = Vec3::new(view.y_axis.x, view.y_axis.y, view.y_axis.z);
        let scale = self.distance * 0.001;
        self.target -= right * delta.x * scale;
        self.target += up   * delta.y * scale;
    }

    pub fn state(&self) -> CameraState {
        CameraState {
            target: self.target.to_array(),
            distance: self.distance,
            yaw: self.yaw,
            pitch: self.pitch,
            parallel: self.parallel,
        }
    }

    pub fn apply_state(&mut self, s: CameraState) {
        self.target   = Vec3::from(s.target);
        self.distance = s.distance.max(self.near * 10.0);
        self.yaw      = s.yaw;
        self.pitch     = s.pitch.clamp(-1.55, 1.55);
        self.parallel  = s.parallel;
    }
}
