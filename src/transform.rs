use cgmath::{Deg, Matrix4, SquareMatrix, Vector3, Zero};

type Vec3 = Vector3<f32>;
type Mat4 = Matrix4<f32>;

pub struct Transform {
    translation: Vec3,
    rotation: Vec3,
    scale: Vec3,
}

impl Transform {
    pub fn add_translation(&mut self, delta: Vec3) {
        self.translation += delta;
    }

    pub fn add_rotation(&mut self, delta: Vec3) {
        self.rotation += delta;
    }

    pub fn add_scale(&mut self, delta: Vec3) {
        self.scale += delta;
    }

    pub fn get_transform_mat4(&self) -> Mat4 {
        let translation_mat4 = Mat4::from_translation(self.translation);
        let rotation_mat4 = Mat4::from_angle_z(Deg(self.rotation.z))
            * Mat4::from_angle_y(Deg(self.rotation.y))
            * Mat4::from_angle_x(Deg(self.rotation.x));
        let scale_mat4 = Mat4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);

        translation_mat4 * rotation_mat4 * scale_mat4
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::zero(),
            rotation: Vec3::zero(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}
