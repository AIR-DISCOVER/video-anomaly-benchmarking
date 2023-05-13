import os
import math
import argparse
import numpy as np
from PIL import Image

SELECT_THRESHOLD = 80  # Only evaluate near 80m
DEPTH_THRESHOLD = 0.8  # Relative error of 1%
WIDTH, HEIGHT = 1920, 1080

def get_intrinsic_matrix():
    # X: forward
    # Y: right
    # Z: up
    
    FOV = 90
    
    fy = WIDTH / 2 / np.tan(FOV / 2 * np.pi / 180)
    fz = HEIGHT / 2 / np.tan(FOV / 2 * np.pi / 180)
    
    cy = WIDTH / 2
    cz = HEIGHT / 2
    
    K = np.array([
        [1, 0, 0],
        [cy, fy, 0],
        [cz, 0, fz]
    ])
    
    inv_K = np.linalg.inv(K)

    return K, inv_K


def get_extrinsic_matrix(extrinsic_group):
    # CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system.
    # pitch (float – degrees) – Y-axis rotation angle.
    # yaw (float – degrees) – Z-axis rotation angle.
    # roll (float – degrees) – X-axis rotation angle.
    
    x, y, z, pitch, yaw, roll = extrinsic_group
    
    # Convert degrees to radians
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)
    
    Cp, Sp = math.cos(pitch_rad), math.sin(pitch_rad)
    Cy, Sy = math.cos(yaw_rad), math.sin(yaw_rad)
    Cr, Sr = math.cos(roll_rad), math.sin(roll_rad)
    
    # https://github.com/carla-simulator/carla/issues/2516#issuecomment-861761770
    R = np.array([
        [Cp * Cy, -Cr * Sy + Sr * Sp * Cy, Sr * Sy + Cr * Sp * Cy],
        [Cp * Sy, Cr * Cy + Sr * Sp * Sy, -Sr * Cy + Cr * Sp * Sy],
        [
            -Sp, Sr * Cp, Cr * Cp]
    ])
    
    return np.concatenate([R, np.array([[x], [y], [z]])], axis=1)


class Img2Camera:  # from image to camera space
    # Use UNREAL Engine coordinate system
    def __init__(self, K, inv_K):
        self.height = 1080
        self.width = 1920

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.ones = np.ones(self.height * self.width)[None, ...]
        self.pix_coords = np.stack([self.id_coords[0].reshape(-1), self.id_coords[1].reshape(-1)], 0)
        self.pix_coords = np.concatenate([self.ones, self.pix_coords], 0)
    
        self.K, self.inv_K = K, inv_K
    
    def apply(self, depth, attribute=None):
        cam_points = self.inv_K @ self.pix_coords
        cam_points2 = depth.reshape(1, -1) * cam_points
        
        # z *= -1
        cam_points2 = np.concatenate([cam_points2[:2, :], -cam_points2[2:, :]], 0)  # 3 x HW
        
        cam_points3 = np.concatenate([cam_points2, self.ones], 0)  # 3 x HW
        
        # attribute: C x H x W
        if attribute is not None:
            attribute = attribute.reshape(attribute.shape[0], -1)  # C x H x W -> C x HW
        
        return cam_points3, attribute


class Camera2World:  # from camera space to world space
    def __init__(self, extrinsic_matrix):
        self.extrinsic_matrix = extrinsic_matrix
    
    def apply(self, cam_points):
        # cam_points: 3 x N
        world_points = self.extrinsic_matrix @ cam_points  # 3 x N
        return world_points


class World2Camera:  # from world space to camera space
    def __init__(self, extrinsic_matrix):
        self.extrinsic_matrix = extrinsic_matrix  # 3 x 4
        self.extrinsic_matrix = np.concatenate([self.extrinsic_matrix, np.array([[0, 0, 0, 1]])], axis=0)  # 4 x 4
    
    def apply(self, world_points):
        # world_points: 3 x N
        world_points2 = np.concatenate([world_points, np.ones((1, world_points.shape[1]))], axis=0)  # 4 x N
        cam_points = np.linalg.inv(self.extrinsic_matrix) @ world_points2
        return cam_points[:3, :]  # 3 x N


class Camera2Img:  # from camera space to image space
    def __init__(self, K, inv_K):
        self.K = K
    
    def apply(self, cam_points):
        # z *= -1
        cam_points = np.concatenate([cam_points[:2, :], -cam_points[2:, :]], 0)  # 3 x N
        pix_coords = self.K @ cam_points
        pix_coords2 = pix_coords / pix_coords[0]
        pix_coords2 = pix_coords2[1:, :]
        return pix_coords2, pix_coords[:1, :]  # 2 x N, 1 x N


def project(extrinsic1, depth_path1, attribute1, extrinsic2, depth_path2):
    """Project extrinsic 1 to extrinsic 2.
    """

    img2camera = Img2Camera(*get_intrinsic_matrix())
    camera2img = Camera2Img(*get_intrinsic_matrix())

    camera2world1 = Camera2World(get_extrinsic_matrix(extrinsic1))
    world2camera2 = World2Camera(get_extrinsic_matrix(extrinsic2))
    
    assert os.path.exists(depth_path1), "Depth path does not exist!"
    assert os.path.exists(depth_path2), "Depth path does not exist!"

    depth_map = np.asarray(Image.open(depth_path1))
    depth_arr1 = (depth_map[..., 0] + 256 * depth_map[..., 1] + 65536 * depth_map[..., 2]) / (256 * 256 * 256 - 1) * 1000

    depth_map = np.asarray(Image.open(depth_path2))
    depth_arr2 = (depth_map[..., 0] + 256 * depth_map[..., 1] + 65536 * depth_map[..., 2]) / (256 * 256 * 256 - 1) * 1000
    
    # assert attribute1 to be (C, H, W)
    assert attribute1.ndim == 3, "Attribute should be 3D!"
    assert attribute1.shape[1] == depth_arr1.shape[0] and attribute1.shape[2] == depth_arr1.shape[1], "Attribute shape does not match depth shape!"
    
    camera_coords, attribute = img2camera.apply(depth_arr1, attribute1)
    
    camera_norm = np.linalg.norm(camera_coords[:3, :], axis=0)
    keep_indices = np.arange(HEIGHT * WIDTH)
    mask1 = camera_norm < SELECT_THRESHOLD
    keep_indices = keep_indices[mask1]
    viz_camera_coords = camera_coords[:, mask1]
    viz_attribute = attribute[:, mask1]
    
    
    viz_world_coords = camera2world1.apply(viz_camera_coords)
    another_camera_coords = world2camera2.apply(viz_world_coords)
    img_proj, depth_proj = camera2img.apply(another_camera_coords)

    mask2 = np.any(img_proj < 0, axis=0)
    mask3 = img_proj[0, :] >= WIDTH
    mask4 = img_proj[1, :] >= HEIGHT
    mask5 = depth_proj[0, :] <= 0
    
    keep_mask = ~(mask2 | mask3 | mask4 | mask5)
    keep_indices = keep_indices[keep_mask]
    
    img_proj_keep = img_proj[:, keep_mask]
    depth_proj_keep = depth_proj[:, keep_mask]
    attribute_keep = viz_attribute[:, keep_mask]
    
    reconstructed_depth = np.zeros((HEIGHT, WIDTH)) + 1000000
    reconstructed_attribute = np.zeros((attribute_keep.shape[0], HEIGHT, WIDTH))
    reconstructed_already_written = np.zeros((HEIGHT, WIDTH))
    for i in range(img_proj_keep.shape[1]):
        x, y = img_proj_keep[:, i]
        # use interpolation to fill in the depth map
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if x + dx < 0 or x + dx >= WIDTH or y + dy < 0 or y + dy >= HEIGHT:
                    continue
                candidate_x, candidate_y = int(x + dx), int(y + dy)
                
                if reconstructed_already_written[candidate_y, candidate_x] == 0:
                    reconstructed_depth[candidate_y, candidate_x] = depth_proj_keep[0, i]
                    reconstructed_attribute[:, candidate_y, candidate_x] = attribute_keep[:, i]
                else:
                    reconstructed_depth[candidate_y, candidate_x] = min(reconstructed_depth[candidate_y, candidate_x], depth_proj_keep[0, i])
                    reconstructed_attribute[:, candidate_y, candidate_x] = (reconstructed_attribute[:, candidate_y, candidate_x] + attribute_keep[:, i]) / 2
                
        
        x, y = int(x), int(y)
        reconstructed_depth[y, x] = depth_proj_keep[:, i]
        reconstructed_attribute[:, y, x] = attribute_keep[:, i]
    
    # Return a reconstructed depth map and attribute map, and a mask of shape (H, W)
    
    mask6 = np.abs(reconstructed_depth - depth_arr2) <= DEPTH_THRESHOLD
    mask7 = mask6.reshape(-1)[mask1][keep_mask]
    keep_indices = keep_indices[mask7]
    
    # convert keep_indices to (H, W) mask
    mask = np.zeros((HEIGHT * WIDTH), dtype=bool)
    mask[keep_indices] = True
    mask = mask.reshape(HEIGHT, WIDTH)
    
    return reconstructed_depth, reconstructed_attribute, mask



if __name__ == "__main__":
    # Unit test for consistency
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_path', type=str, required=True)
    parser.add_argument('--depth_path2', type=str, required=True)
    opt = parser.parse_args()
    
    depth_path = "/data21/tb5zhh/datasets/anomaly_dataset/v5_release/train/seq09-3/depth_v/237.png"
    depth_path2 = "/data21/tb5zhh/datasets/anomaly_dataset/v5_release/train/seq09-3/depth_v/243.png"
    assert os.path.exists(depth_path), "Depth path does not exist!"
    assert os.path.exists(depth_path2), "Depth path does not exist!"
    
    rgb_map = np.asarray(Image.open("/data21/tb5zhh/datasets/anomaly_dataset/v5_release/train/seq09-3/enhanced_rgb_v/cityscapes/reg-0.3-clip-1000/237.png")).transpose(2, 0, 1)
    
    extrinsic1 = [
                179.258392,
                -364.451996,
                1.801462,
                0.03788,
                0.764788,
                -0.004883
            ]
    
    extrinsic61 = [
                179.831955,
                -364.449554,
                1.800967,
                0.019186,
                0.656206,
                0.000105
            ]

    reconstructed_depth, reconstructed_attribute, mask = project(extrinsic1, depth_path, rgb_map, extrinsic61, depth_path2)

    Image.fromarray(reconstructed_attribute.transpose(1, 2, 0).astype(np.uint8)).save("reconstructed_rgb.png")
