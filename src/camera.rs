use cgmath::{Matrix4, Vector3, perspective, Rad};

pub struct Camera {
    fovy: f32, // vertical field of view, in radian
    znear: f32,
    zfar: f32,

    translation: Vector3,
    rotation: Vector3, // rotation angles (in radian) around x, y and z axes respectively

    perspective: Matrix4,
    view: Matrix4,

    flip_y: f32, // either 1 or -1
}

impl Camera {
    pub fn new() -> Self {
        let mut camera = Camera {
            fovy: 0,
            znear: 0,
            zfar: 0,
            translation: Vector3::zero(),
            rotation: Vector3::zero(),
            perspective: Matrix4::indentity(),
            view: Matrix4::identity(),
            flip_y: 1,
        }
    }

    pub fn set_flip_y(&mut self, flip_y: bool) {
        if flip_y == true {
            self.flip_y = -1;
        } else {
            self.flip_y = 1;
        }
    }

    /// Rotate `delta` (in radian) around x, y and z axes from the current orientation.
    pub fn rotate(&mut self, delta: Vector3) {
        self.rotation += delta;
    }

    pub fn translate(&mut self, delta: Vector3) {
        self.position += delta;
    }

    /// Set the perspective projection matrix.
    /// `fovy` - Vertical field of view in radian.
    /// `aspect` - Aspect ratio of the near clipping plane.
    pub fn set_perspective(&mut self, fovy: f32, aspect: f32, znear: f32, zfar: f32) {
        self.fovy = fovy;
        self.znear = znear;
        self.zfar = zfar;

        self.perspective = perspective(fov, aspect, znear, zfar);
        self.perspective[1][1] *= flip_y;
    }

    pub fn update_view_matrix(&mut self) {
        let rot_x = Matrix4::from_angle_x(self.rotation.x * self.flip_y);
        let rot_y = Matrix4::from_angle_y(self.rotation.y);
        let rot_z = Matrix4::from_angle_z(self.rotation.z);
        let rot = rot_x * rot_y * rot_z;

        let trans = self.translation;
        trans.y *= self.flip_y;
        trans_mat = Matrix4::from_translation(trans);

        self.view = trans_mat * rot;
    }
}

