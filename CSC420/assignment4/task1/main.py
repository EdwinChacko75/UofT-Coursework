import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

n_points = 8             # Number of points to select per image

#########################################
# DO NOT CHANGE THIS OR IT WILL OVERWRITE 
# MY FIGURES AND I WILL FAIL THIS 
# ASSIGNMENT AND THEN ILL BE VERY SAD
load_from_cache = True  # Set to True to load previously saved points
#########################################

def load_gray_image(path):
    """
    Load an image from disk and convert to gray level using the provided settings.
    """
    img = cv2.imread(path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return gray, img_rgb

def select_points_single(img_gray, title="Select points", n_points=6, color='r', case=None):
    """
    Display the gray image and allow manual selection of n_points. 
    Mark each point with a colored square.
    """
    plt.imshow(img_gray, cmap='gray')
    plt.title(title)
    print(f"Click {n_points} points on: {title}")

    points = plt.ginput(n_points, timeout=0)
    for pt in points:
        plt.plot(pt[0], pt[1], color + 's', markersize=8)
    plt.savefig(f'./assets/{case}.png')
    plt.show()
    return np.array(points)

def get_or_load_points(case_name, imgA_path, imgB_path, region_desc, colorA='r', colorB='b'):
    """
    If cached points exist and load_from_cache is True, load them. 
    Otherwise, select points manually.
    """
    cache_file = f"corresponding_points_{case_name}.npz"
    if load_from_cache and os.path.exists(cache_file):
        print(f"Loading cached points for case {case_name} from {cache_file}")
        data = np.load(cache_file)
        return data['ptsA'], data['ptsB']
    else:
        # Load images as gray
        imgA_gray, _ = load_gray_image(imgA_path)
        imgB_gray, _ = load_gray_image(imgB_path)
        ptsA = select_points_single(imgA_gray, f"Image I ({imgA_path}) - {region_desc}", n_points=n_points, color=colorA, case=f'{case_name}1')
        ptsB = select_points_single(imgB_gray, f"Image eI ({imgB_path}) - {region_desc}", n_points=n_points, color=colorB, case=f'{case_name}2')
        np.savez(cache_file, ptsA=ptsA, ptsB=ptsB)
        print(f"Saved selected points to {cache_file}")
        return ptsA, ptsB

def compute_homography(pts_src, pts_dst):
    """
    Compute the 3x3 homography matrix H mapping pts_src to pts_dst.
    """
    N = pts_src.shape[0]

    A = []
    for i in range(N):
        x, y = pts_src[i]
        xp, yp = pts_dst[i]
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])

    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1, :]
    H = h.reshape(3, 3)

    if H[2, 2] != 0:
        H = H / H[2, 2]
    return H

def plot_matched_points(eI_gray, pts_eI, pts_estimated, case=None):
    """Display the eI gray image with red squares for selected points and green squares for estimated points."""
    plt.figure()
    plt.imshow(eI_gray, cmap='gray')
    plt.plot(pts_eI[:, 0], pts_eI[:, 1], 'rs', markersize=8, label='Selected pts')
    plt.plot(pts_estimated[:, 0], pts_estimated[:, 1], 'gs', markersize=8, label='Estimated pts')
    plt.title("eI Image: Selected (red) vs. Estimated (green) points")
    plt.legend()
    plt.savefig(f'./assets/{case}_matches.png')

    plt.show()

def create_new_image(I_gray, eI_gray, H):
    """
    Create a new image (with 3 channels) large enough to contain the I image and
    all inverse-mapped pixels from the eI image.
    """
    hI, wI = I_gray.shape
    h_eI, w_eI = eI_gray.shape

    # I image corners in I coordinates
    corners_I = np.array([[0, 0],
                          [wI - 1, 0],
                          [wI - 1, hI - 1],
                          [0, hI - 1]], dtype=np.float32)
    # eI image corners in eI coordinates
    corners_eI = np.array([[0, 0],
                           [w_eI - 1, 0],
                           [w_eI - 1, h_eI - 1],
                           [0, h_eI - 1]], dtype=np.float32)

    H_inv = np.linalg.inv(H)
    ones = np.ones((4, 1), dtype=np.float32)
    corners_eI_homog = np.hstack((corners_eI, ones))
    corners_eI_mapped = (H_inv @ corners_eI_homog.T).T  # shape (4, 3)
    corners_eI_mapped = corners_eI_mapped[:, :2] / corners_eI_mapped[:, [2]]

    all_points = np.vstack((corners_I, corners_eI_mapped))
    min_xy = np.floor(all_points.min(axis=0)).astype(int)
    max_xy = np.ceil(all_points.max(axis=0)).astype(int)

    new_w = max_xy[0] - min_xy[0] + 1
    new_h = max_xy[1] - min_xy[1] + 1
    offset_x = -min_xy[0]
    offset_y = -min_xy[1]

    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    new_img[offset_y:offset_y+hI, offset_x:offset_x+wI, 0] = I_gray

    ys, xs = np.indices((new_h, new_w))  
    src_xs = xs - offset_x
    src_ys = ys - offset_y

    ones_arr = np.ones_like(src_xs)
    homog_coords = np.stack([src_xs, src_ys, ones_arr], axis=-1)  
    homog_coords_reshaped = homog_coords.reshape(-1, 3).T  

    mapped_coords = H @ homog_coords_reshaped  
    mapped_coords /= mapped_coords[2, :]  

    ex = mapped_coords[0, :].reshape(new_h, new_w)
    ey = mapped_coords[1, :].reshape(new_h, new_w)

    ex_round = np.rint(ex).astype(int)
    ey_round = np.rint(ey).astype(int)

    valid_mask = (ex_round >= 0) & (ex_round < w_eI) & (ey_round >= 0) & (ey_round < h_eI)

    valid_y, valid_x = np.nonzero(valid_mask)

    mapped_ex = ex_round[valid_mask]
    mapped_ey = ey_round[valid_mask]
    eI_values = eI_gray[mapped_ey, mapped_ex]

    new_img[valid_y, valid_x, 1] = eI_values  
    new_img[valid_y, valid_x, 2] = eI_values  

    return new_img


def main():
    case_choice = input("Enter case (A, B, or C): ").strip().upper()
    
    if case_choice == 'A':
        imgA_path = "hallway1.jpg"  
        imgB_path = "hallway2.jpg"  
        region_desc = "RIGHT WALL"
        colorA = 'r'
        colorB = 'b'
    elif case_choice == 'B':
        imgA_path = "hallway1.jpg"
        imgB_path = "hallway3.jpg"
        region_desc = "RIGHT WALL"
        colorA = 'r'
        colorB = 'b'
    elif case_choice == 'C':
        imgA_path = "hallway1.jpg"
        imgB_path = "hallway3.jpg"
        region_desc = "FLOOR"
        colorA = 'g'
        colorB = 'y'
    
    # (1) Select corresponding points 
    ptsA, ptsB = get_or_load_points(case_name=case_choice,
                                    imgA_path=imgA_path,
                                    imgB_path=imgB_path,
                                    region_desc=region_desc,
                                    colorA=colorA,
                                    colorB=colorB)
    ptsA = np.array(ptsA)
    ptsB = np.array(ptsB)
    
    # (2) Compute H manually
    H = compute_homography(ptsA, ptsB)
    print("Estimated Homography H:")
    print(H)
    
    # (3) Overlay visualization
    pts_estimated = []
    for pt in ptsA:
        vec = np.array([pt[0], pt[1], 1])
        mapped = H.dot(vec)
        mapped = mapped / mapped[2]
        pts_estimated.append(mapped[:2])
    pts_estimated = np.array(pts_estimated)
    
    eI_gray, _ = load_gray_image(imgB_path)
    plot_matched_points(eI_gray, ptsB, pts_estimated, case_choice)
    
    # (4) Create a new image as described.
    I_gray, _ = load_gray_image(imgA_path)
    new_img = create_new_image(I_gray, eI_gray, H)
    plt.figure()
    plt.imshow(new_img)
    plt.title("New Image: I (red) & Inverse-mapped eI (green/blue)")
    plt.savefig(f'./assets/{case_choice}_mapping.png')

    plt.show()

if __name__ == "__main__":
    main()
