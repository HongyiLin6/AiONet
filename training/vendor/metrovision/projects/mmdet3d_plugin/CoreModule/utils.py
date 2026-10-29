import torch
import torchvision.transforms.functional as TF


## bbox fusion
def transform_bbox(bbox_dec, R, T):
    """
    Batched coordinate transformation.
    bbox_dec: num_dec * Batch * num_query * box_dim
    """
    transformed = []
    for bbox in bbox_dec:
        B = len(R)
        # center in the bev
        cx = bbox[..., 0:1]
        cy = bbox[..., 1:2]
        cz = bbox[..., 4:5]

        centers = torch.cat([cx, cy, cz], dim=-1)

        # size
        w = bbox[..., 2:3]
        l = bbox[..., 3:4]
        h = bbox[..., 5:6]
        rot_sine = bbox[..., 6:7]
        rot_cosine = bbox[..., 7:8]
        rot = torch.atan2(rot_sine, rot_cosine)
        velocity = bbox[..., 8:10]
        if centers.dtype != R[0].dtype:
            R = [r.to(centers.dtype) for r in R]
            T = [t.to(centers.dtype) for t in T]
        translates = []
        for i in range(B):
            translate = centers[i] @ R[i].T + T[i]
            x = translate[..., 0:1]
            y = translate[..., 1:2]
            z = translate[..., 2:3]
            rot_sin = R[i][1, 0]
            rot_cos = R[i][0, 0]
            angle = torch.atan2(rot_sin, rot_cos)
            rot_new = rot[i] + angle
            R_2d = R[i][:2, :2]
            rotated_vel = velocity[i] @ R_2d.T
            transformed_boxes = torch.cat(
                (x, y, w[i], l[i], z, h[i], torch.sin(rot_new), torch.cos(rot_new), rotated_vel), dim=-1)
            translates.append(transformed_boxes)
        transformed.append(torch.stack(translates, dim=0))
    return torch.stack(transformed, dim=0)


## bev fusion

def transform_bev(bev_feat, R, T, point_cloud_range):
    """
    Transform the coordinate frame of BEV features.

    Args:
        bev_feat (torch.Tensor): Input BEV features, shape [C, H, W]
        inf2veh (torle.Tensor): Affine transformation matrix (3x3 or 4x4)
        point_cloud_range (list): Perception bounds [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        torch.Tensor: Transformed BEV features, shape [C, H, W]
    """

    # Extract rotation and translation parameters.
    angle = _extract_rotation_angle(R)  # Extract the rotation angle in degrees.
    translate = _extract_translation_pixels(T, point_cloud_range, bev_feat.shape[0], bev_feat.shape[1])  # Convert to pixel offsets.

    # Apply the affine rotation and translation.
    transformed = TF.affine(
        bev_feat,
        angle=angle,
        translate=translate,
        scale=1.0,  # Unit scale
        shear=[0.0, 0.0],  # Zero shear
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=0  # Fill outside the bounds with zeros.
    )

    # Optionally crop to the physical perception bounds.
    # mask = _generate_boundary_mask(transformed.shape, point_cloud_range)
    # transformed = transformed * mask

    return transformed


def _generate_boundary_mask(shape, point_cloud_range):
    """
    Create a BEV-sized mask that is zero outside the perception bounds.
    """
    h, w, _ = shape
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    # Generate grid coordinates in physical units.
    x_lin = torch.linspace(x_min, x_max, w)
    y_lin = torch.linspace(y_min, y_max, h)
    grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing='xy')

    # Check the ground-plane perception bounds.
    mask = (grid_x >= x_min) & (grid_x <= x_max) & (grid_y >= y_min) & (grid_y <= y_max)
    return mask.float().unsqueeze(-1)  # Expand to [H, W, 1] for broadcasting.


def _extract_rotation_angle(inf2veh):
    """
    Extract the angle of a rotation about the Z axis.
    The angle follows the affine rotation parameter convention.
    """
    # Extract the upper-left 2x2 rotation block.
    rotation_matrix = inf2veh[:2, :2]
    # Convert the rotation angle from radians to degrees.
    angle_rad = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    angle_deg = torch.rad2deg(angle_rad).item()
    return angle_deg


def _extract_translation_pixels(T, point_cloud_range, h_pixels, w_pixels):
    """
    Convert metric translation to pixel offsets using the perception bounds.
    Use the BEV grid resolution to map metric and pixel coordinates.
    """
    # Extract metric translation in meters.
    tx_m = T[0]  # Translation along x
    ty_m = T[1]  # Translation along y

    # Compute the resolution of a uniform BEV grid.
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    dx = (x_max - x_min) / w_pixels  # Meters per pixel along x
    dy = (y_max - y_min) / h_pixels  # Meters per pixel along y

    # Compute integer pixel offsets for the affine transform.
    tx_pixel = int(torch.round(tx_m / dx).item())
    ty_pixel = int(torch.round(ty_m / dy).item())

    return [tx_pixel, ty_pixel]  # The affine translate parameter takes a list.


def transform_query(query, R, T):
    for q in query:
        B = len(R)
        # center in the bev
        xyz = q[..., 0:3]
        yaw = q[..., 3:4]
        if R[0].dtype != xyz.dtype:
            R = [r.to(xyz.dtype) for r in R]
            T = [t.to(xyz.dtype) for t in T]
        for i in range(B):
            translate = xyz[i] @ R[i].T + T[i]
            rot_sin = R[i][1, 0]
            rot_cos = R[i][0, 0]
            angle = torch.atan2(rot_sin, rot_cos)
            new_yaw = yaw[i] + angle
            q[i][..., 0:3] = translate
            q[i][..., 3:4] = new_yaw
    return query
