use cgmath::{perspective, Deg, Matrix4, SquareMatrix, Vector3, Zero};
use crate::transform::Transform;

pub struct Camera {
    fovy: Deg<f32>, // vertical field of view
    znear: f32,
    zfar: f32,

    perspective: Matrix4<f32>,

    transform: Transform,
}

impl Camera {
    /// Set the perspective projection matrix.
    /// `fovy` - Vertical field of view in radian.
    /// `aspect` - Aspect ratio (width / height) of the near clipping plane.
    pub fn set_perspective(&mut self, fovy: Deg<f32>, aspect: f32, znear: f32, zfar: f32) {
        self.fovy = fovy;
        self.znear = znear;
        self.zfar = zfar;

        self.perspective = perspective(fovy, aspect, znear, zfar);
    }

    /// Get world-to-camera transformation matrix.
    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        let matrix = self.transform.get_transform_mat4();
        matrix.invert().unwrap()
    }

    pub fn set_rotation(&mut self) {}
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fovy: Deg(60.0),
            znear: 1.0,
            zfar: 256.0,
            perspective: Matrix4::<f32>::identity(),
            transform: Transform::default(),
        }
    }
}
