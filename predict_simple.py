import cv2
from pathlib import Path
import time
import numpy as np
from datasets.transforms import PytorchHubNormalization
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights

# Download from https://vision.fe.uni-lj.si/public/mods_raw/
MODS_RAW_DIR = Path('C:\\') / 'Users' / 'Aresh' / 'Downloads' / 'MODS_raw'

# Download from https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth
PRETRAINED_PATH = Path('pretrained') /'ewasr_resnet18.pth'

CALIB_DIR = MODS_RAW_DIR / 'calibration'
SEQS_DIR = MODS_RAW_DIR / 'sequences'

def yaml_to_coeffs(calib_path: Path):
    """Load calibration data from a YAML file."""
    assert calib_path.exists()
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    M1 = fs.getNode('M1').mat()  # Left camera matrix
    D1 = fs.getNode('D1').mat()  # Left distortion coefficients
    M2 = fs.getNode('M2').mat()  # Right camera matrix
    D2 = fs.getNode('D2').mat()  # Right distortion coefficients
    R = fs.getNode('R').mat()    # Rotation matrix
    T = fs.getNode('T').mat()    # Translation vector
    img_node = fs.getNode('imageSize')
    img_size = (int(img_node.at(0).real()), int(img_node.at(1).real()))
    fs.release()
    return M1, D1, M2, D2, R, T, img_size

def coeffs_to_maps(M1, D1, M2, D2, R, T, img_size):
    """Construct rectification maps from calibration data."""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, D1, M2, D2, img_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, img_size, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, img_size, cv2.CV_32FC1)

    return mapx1, mapy1, mapx2, mapy2, Q

def yaml_to_maps(calib_path: Path):
    M1, D1, M2, D2, R, T, img_size = yaml_to_coeffs(calib_path)
    mapx1, mapy1, mapx2, mapy2, Q = coeffs_to_maps(M1, D1, M2, D2, R, T, img_size)
    return mapx1, mapy1, mapx2, mapy2, Q

def predict_single_image(predictor: Predictor, img, imu_mask):
    img = PytorchHubNormalization()(img).unsqueeze(0)
    imu_mask = imu_mask.unsqueeze(0)
    features = {
        "image": img,
        "imu_mask": imu_mask
    }
    _pred_masks = predictor.predict_batch(features)
    pred_mask = _pred_masks[0]
    return pred_mask

def get_sparse_matches(left_gray, right_gray, mask=None, max_disparity=64, y_tol=2, check_x_disp=True, fb_tol=4):

    half = max_disparity // 2

    # crop so disparity >= 0 and <= max_disparity
    L = left_gray[:, half:]
    R = right_gray[:, :-half]
    M = mask[:, half:] if mask is not None else None

    # corners
    feature_params = dict(
        maxCorners=300,              # more points; filter later
        qualityLevel=0.01,
        minDistance=10,
        blockSize=5,
        useHarrisDetector=False,
        mask=M
    )
    left_pts = cv2.goodFeaturesToTrack(L, **feature_params)
    if left_pts is None:
        return []

    # LK params: pyramids on, normal patch size
    lk = dict(
        winSize=(31, 31),
        maxLevel=6,  # allows ~512px motion once downsampled
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-4
    )

    right_pts, status, err = cv2.calcOpticalFlowPyrLK(L, R, left_pts, None, **lk)
    left_pts_2, status_2, err_2 = cv2.calcOpticalFlowPyrLK(R, L, right_pts, None, **lk)

    # flatten shapes
    left_pts = left_pts.reshape(-1, 2)
    right_pts = right_pts.reshape(-1, 2)
    status = status.reshape(-1)
    err = err.reshape(-1)
    left_pts_2 = left_pts_2.reshape(-1, 2)
    status_2 = status_2.reshape(-1)
    err_2 = err_2.reshape(-1)

    out = []

    def good_match(l, r, s, e):
        d = l[0] - r[0]
        if s == 0:
            return False
        if y_tol is not None and abs(l[1] - r[1]) > y_tol:
            return False
        if check_x_disp and (d < 0 or d > max_disparity):
            return False
        return True

    for l, r, s, e, l2, s2, e2 in zip(left_pts, right_pts, status, err, left_pts_2, status_2, err_2):
        if not good_match(l, r, s, e):
            continue
        if not good_match(l2, r, s2, e2):
            continue
        if fb_tol is not None and abs(l[0] - l2[0]) > fb_tol:
            continue

        # restore left x to uncropped coords
        l_full = l + np.array([half, 0], dtype=l.dtype)

        # optional quality filters:
        # if using GET_MIN_EIGENVALS: require e > thresh (e.g., 1e-4..1e-3)
        # else (SSD): require e < thresh (tune experimentally)
        out.append((l_full, r, e))
    return out

def rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right):
    rectified_left = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
    return rectified_left, rectified_right

def clahe_pair(rectified_left, rectified_right, clahe):
    left_gray = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY) 
    right_gray = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY) 
    return clahe.apply(left_gray), clahe.apply(right_gray)

def get_predictor():
    model = models.get_model(
        'ewasr_resnet18',
        num_classes=3, # Obstacle, water, sky
        pretrained=False,
        mixer="CCCCSS", # Token mixers in feature mixer(?)
        enricher="SS", # Token mixers in long-skip feature enricher(?)
        project=False # Project encoder features to fewer channels
    )
    state_dict = load_weights(PRETRAINED_PATH)
    model.load_state_dict(state_dict)
    fp16 = False # Half-precision
    return Predictor(model, fp16)

def main():
    predictor = get_predictor()

    # calib_path = CALIB_DIR / 'calibration-kope100.yaml'
    # seq_dir = SEQS_DIR / 'kope100-00006790-00007090'
    calib_path = CALIB_DIR / 'calibration-kope101.yaml'
    seq_dir = SEQS_DIR / 'kope101-00004130-00004650'
    # calib_path = CALIB_DIR / 'calibration-kope102.yaml'
    # seq_dir = SEQS_DIR / 'kope102-00007700-00008300'
    # seq_dir = SEQS_DIR / 'kope102-00000001-00000350'
    # seq_dir = SEQS_DIR / 'kope102-00022135-00022435'
    frames_dir = seq_dir / 'frames'
    
    assert calib_path.exists(), "Missing calibration file"
    assert frames_dir.exists(), "Missing frame directory"

    left_img_paths = list(frames_dir.glob("*L.jpg"))
    right_img_paths = list(frames_dir.glob("*R.jpg"))
    assert len(left_img_paths) == len(right_img_paths), "Unequal number of left and right images"

    # Exposure correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    M1, D1, M2, D2, R, T, img_size = yaml_to_coeffs(calib_path)
    print("Initial extrinsics:")
    print(R)
    print(T)
    map1_left, map2_left, map1_right, map2_right, Q = coeffs_to_maps(M1, D1, M2, D2, R, T, img_size)

    '''
    # Refine extrinsic params based on first image pair
    left_img_path = left_img_paths[0]
    right_img_path = right_img_paths[0]
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    rectified_left, rectified_right = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
    left_gray, right_gray = clahe_pair(rectified_left, rectified_right, clahe)
    matches = get_sparse_matches(left_gray, right_gray, mask=None, y_tol=None, check_x_disp=False, fb_tol=4)
    left_pts_rect = []
    right_pts_rect = []
    for l, r, _ in matches:
        left_pts_rect.append(l)
        right_pts_rect.append(r)
    left_pts_rect = np.array(left_pts_rect)
    right_pts_rect = np.array(right_pts_rect)
    E, inliers = cv2.findEssentialMat(left_pts_rect, right_pts_rect,
                                      method=cv2.RANSAC,
                                      prob=0.999, threshold=1.0)
    inliers = inliers.ravel().astype(bool)
    left_pts_rect = left_pts_rect[inliers]
    right_pts_rect = right_pts_rect[inliers]
    _, R_new, t_new, _ = cv2.recoverPose(E, left_pts_rect, right_pts_rect)
    R, T = R_new, t_new*np.linalg.norm(T)
    print("Refined extrinsics:")
    print(R)
    print(T)
    map1_left, map2_left, map1_right, map2_right, Q = coeffs_to_maps(M1, D1, M2, D2, R, T, img_size)
    '''

    times = [time.time()]

    for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
        # print("meow")
        # Assert same name except L/R
        assert left_img_path.name[:-5] == right_img_path.name[:-5], "Left and right images have different indices"
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        rectified_left, rectified_right = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
        left_gray, right_gray = clahe_pair(rectified_left, rectified_right, clahe)

        mask = (predict_single_image(predictor, rectified_left, torch.zeros((958, 1278))) == 0).astype(np.uint8)
        mask_img = mask*255
        # cv2.imshow("Obstacle mask", mask_img)

        pts = get_sparse_matches(left_gray, right_gray, mask=mask, y_tol=None, check_x_disp=False, fb_tol=4)

        canvas = cv2.hconcat([rectified_left * np.expand_dims(mask, -1), rectified_right])
        # canvas = cv2.hconcat([rectified_left, rectified_right])
        W = rectified_left.shape[1]
        rng = np.random.default_rng(0)  # fixed seed for repeatable colors

        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for left_pt, right_pt, err in pts:
            x0, y0 = left_pt
            x1, y1 = right_pt
            left_x.append(x0)
            left_y.append(y0)
            right_x.append(x1)
            right_y.append(y1)
            color = tuple(int(c) for c in rng.integers(64, 255, size=3))
            cv2.circle(canvas, (int(x0), int(y0)), 5, color, -1)
            cv2.circle(canvas, (int(x1 + W), int(y1)), 5, color, -1)
        left_x = np.array(left_x)
        left_y = np.array(left_y)
        right_x = np.array(right_x)
        right_y = np.array(right_y)
        disp_x = left_x - right_x
        disp_y = left_y - right_y
        # print(np.median(disp_y))

        canvas = cv2.resize(canvas, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("QDS sparse correspondences", canvas)
        '''
        fig, ax = plt.subplots()
        ax.scatter(disp_x, disp_y)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        plt.show()
        '''
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

