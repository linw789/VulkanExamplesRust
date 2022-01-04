use cgmath::{perspective, Deg, Matrix4, SquareMatrix, Vector3, Zero};

pub struct Camera {
    fovy: Deg<f32>, // vertical field of view
    znear: f32,
    zfar: f32,

    translation: Vector3<f32>,
    rotation: Vector3<f32>, // rotation angles around x, y and z axes respectively

    perspective: Matrix4<f32>,
    view: Matrix4<f32>,
}

impl Camera {
    /// Rotate `delta` (in radian) around x, y and z axes from the current orientation.
    pub fn rotate(&mut self, delta: Vector3<f32>) {
        self.rotation += delta;
    }

    pub fn translate(&mut self, delta: Vector3<f32>) {
        self.translation += delta;
    }

    /// Set the perspective projection matrix.
    /// `fovy` - Vertical field of view in radian.
    /// `aspect` - Aspect ratio (width / height) of the near clipping plane.
    pub fn set_perspective(&mut self, fovy: Deg<f32>, aspect: f32, znear: f32, zfar: f32) {
        self.fovy = fovy;
        self.znear = znear;
        self.zfar = zfar;

        self.perspective = perspective(fovy, aspect, znear, zfar);
    }

    pub fn update_view_matrix(&mut self) {
        type Mat4 = Matrix4<f32>;
        let rot = Mat4::from_angle_z(Deg(self.rotation.z))
            * Mat4::from_angle_y(Deg(self.rotation.y))
            * Mat4::from_angle_x(Deg(self.rotation.x));
        self.view = Mat4::from_translation(self.translation) * rot;
    }

    pub fn set_rotation(&mut self) {}
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fovy: Deg(60.0),
            znear: 1.0,
            zfar: 256.0,
            translation: Vector3::<f32>::zero(),
            rotation: Vector3::<f32>::zero(),
            perspective: Matrix4::<f32>::identity(),
            view: Matrix4::<f32>::identity(),
        }
    }
}
