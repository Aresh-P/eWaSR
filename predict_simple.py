import cv2
from pathlib import Path
import time
import numpy as np
from datasets.transforms import PytorchHubNormalization
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ordered_set import OrderedSet

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, RandomWalk)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
# from stonesoup.dataassociator.probability import JPDA
from pyehm.plugins.stonesoup import JPDAWithEHM
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.gater.distance import DistanceGater
from stonesoup.measures import Mahalanobis
from stonesoup.tracker.simple import MultiTargetTracker   

from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights

# Download from https://vision.fe.uni-lj.si/public/mods_raw/
MODS_RAW_DIR = Path('C:\\') / 'Users' / 'Aresh' / 'Downloads' / 'MODS_raw'

# Download from https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth
PRETRAINED_PATH = Path('pretrained') /'ewasr_resnet18.pth'

# TASK = 'stereo'
# TASK = 'temporal'
TASK = 'stereotemporal'

CALIB_DIR = MODS_RAW_DIR / 'calibration'
SEQS_DIR = MODS_RAW_DIR / 'sequences'

IMG_HEIGHT = 958
IMG_WIDTH = 1278

YAW_ADJUST = 0

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
        maxCorners=100,              # more points; filter later
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

def imu_to_rotation(imu_path):
    global YAW_ADJUST
    with open(imu_path, 'r') as imu_file:
        imu_data = imu_file.read().split()
    roll, pitch, yaw = list(float(s) for s in imu_data)
    yaw += YAW_ADJUST # Sequence-dependent
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cyw, syw = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0,  0],
                [0, cp, -sp],
                [0, sp,  cp]], dtype=np.float64)

    Ry = np.array([[ cyw, 0, syw],
                [   0, 1,   0],
                [-syw, 0, cyw]], dtype=np.float64)

    Rz = np.array([[cr, sr, 0],
                [-sr,  cr, 0],
                [ 0,   0, 1]], dtype=np.float64)

    # R_cam = Rz @ Rx @ Ry
    R_cam = Ry @ Rx @ Rz
    return R_cam

def get_tracker(detector,
                clutter_spatial_density: float = 0.125):
    """
    Build a JPDA-based multi-target tracker that *discovers* new objects
    via a MultiMeasurementInitiator (no hard-coded start positions).

    Parameters
    ----------
    detector : iterable
        Any Stonesoup DetectionReader-like object that yields
        (timestamp, set_of_detections).
    prob_detect : float, optional
        P_D used in the JPDA hypothesiser.
    clutter_spatial_density : float, optional
        λ_c for JPDA gating.
    min_initiation_points : int, optional
        M in the “M-out-of-N” logic – how many detections are required
        before a holding track is promoted to a real track.

    Returns
    -------
    stonesoup.tracker.simple.MultiTargetMixtureTracker
    """

    x_meas_var, y_meas_var = (1.0, 1.0)
    x_step_var, y_step_var = (5, 5)
    # covar_trace_deletion_thresh = 100
    steps_deletion_thresh = 3
    min_initiation_points = 5
    gate_sigma = 3.0

    # --- 1.  Models ---------------------------------------------------------

    transition_model = CombinedLinearGaussianTransitionModel(
        [RandomWalk(x_step_var), RandomWalk(y_step_var)])

    measurement_model = LinearGaussian(
        ndim_state=2,
        mapping=(0, 1),                           # sensor measures x & y only
        noise_covar=np.array([[x_meas_var, 0],
                              [0,    y_meas_var]]))

    # --- 2.  Filtering components ------------------------------------------

    predictor = KalmanPredictor(transition_model)
    updater   = KalmanUpdater(measurement_model)

    # --- 3.  JPDA data-association -----------------------------------------

    # hypothesiser = PDAHypothesiser(
    #     predictor=predictor,
    #     updater=updater,
    #     clutter_spatial_density=clutter_spatial_density,
    #     prob_detect=prob_detect)

    # data_associator = JPDAWithEHM(hypothesiser=hypothesiser)

    hypothesiser = DistanceHypothesiser(
        predictor=predictor,
        updater=updater,
        measure=Mahalanobis(),
        missed_distance=gate_sigma**2)           # cost outside gate

    gater = DistanceGater(hypothesiser,
                          measure=Mahalanobis(),
                          gate_threshold=gate_sigma**2)       # Mahalanobis^2

    data_associator = GNNWith2DAssignment(gater)              # <-- CHANGE!    

    # --- 4.  Deleter for high-uncertainty tracks ---------------------------
    # deleter = CovarianceBasedDeleter(covar_trace_thresh=covar_trace_deletion_thresh)
    deleter = UpdateTimeStepsDeleter(time_steps_since_update=steps_deletion_thresh)

    # --- 5.  Initiator (2-hit logic) ---------------------------------------

    # *Holding* tracker inside the initiator: simple GNN + Mahalanobis gate
    init_hypo   = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=3)
    init_assoc  = GNNWith2DAssignment(init_hypo)

    prior = GaussianState([[IMG_WIDTH/2], [IMG_HEIGHT/2]],
                          np.diag([IMG_WIDTH*IMG_WIDTH/12, IMG_HEIGHT*IMG_HEIGHT/12]))

    initiator = MultiMeasurementInitiator(
        prior_state=prior,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=init_assoc,
        updater=updater,
        min_points=min_initiation_points)          # 2/2 by default :contentReference[oaicite:0]{index=0}

    # --- 6.  Build tracker --------------------------------------------------

    tracker = MultiTargetTracker(
        initiator=initiator,        # your MultiMeasurementInitiator
        deleter=deleter,
        detector=detector,
        data_associator=data_associator,
        updater=updater)

    return tracker

def id_to_color(i):
    rng = np.random.default_rng(i)
    return tuple(int(c) for c in rng.integers(64, 255, size=3))

def main():
    global YAW_ADJUST
    predictor = get_predictor()

    # calib_path = CALIB_DIR / 'calibration-kope100.yaml'
    # seq_dir = SEQS_DIR / 'kope100-00006790-00007090'
    # YAW_ADJUST = 0.5

    # calib_path = CALIB_DIR / 'calibration-kope101.yaml'
    # seq_dir = SEQS_DIR / 'kope101-00004130-00004650'
    # YAW_ADJUST = 2.75

    calib_path = CALIB_DIR / 'calibration-kope102.yaml'
    seq_dir = SEQS_DIR / 'kope102-00007700-00008300'
    YAW_ADJUST = -2.5

    # calib_path = CALIB_DIR / 'calibration-kope102.yaml'
    # seq_dir = SEQS_DIR / 'kope102-00000001-00000350'
    # YAW_ADJUST = 2 # Doesn't stay in one place

    # calib_path = CALIB_DIR / 'calibration-kope102.yaml'
    # seq_dir = SEQS_DIR / 'kope102-00022135-00022435'
    # YAW_ADJUST = 1.75

    frames_dir = seq_dir / 'frames'
    imu_dir = seq_dir / 'imu'
    
    assert calib_path.exists(), "Missing calibration file"
    assert frames_dir.exists(), "Missing frame directory"

    left_img_paths = list(frames_dir.glob("*L.jpg"))
    right_img_paths = list(frames_dir.glob("*R.jpg"))
    imu_paths = list(imu_dir.glob("*.txt"))
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

    if TASK == 'stereo':
        for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
            # Assert same name except L/R
            assert left_img_path.name[:-5] == right_img_path.name[:-5], "Left and right images have different indices"
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)

            rectified_left, rectified_right = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
            left_gray, right_gray = clahe_pair(rectified_left, rectified_right, clahe)

            mask = (predict_single_image(predictor, rectified_left, torch.zeros((IMG_HEIGHT, IMG_WIDTH))) == 0).astype(np.uint8)
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
            cv2.imshow("Sparse correspondences", canvas)
            '''
            fig, ax = plt.subplots()
            ax.scatter(disp_x, disp_y)
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            plt.show()
            '''
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    elif TASK == 'temporal':

        empty_detector = iter(())                       # any zero-length iterable works
        tracker = get_tracker(detector=empty_detector)

        fx = float(Q[2, 3])                 # focal length in pixels
        cx = float(-Q[0, 3])                # principal point x
        cy = float(-Q[1, 3])                # principal point y
        K  = np.array([[fx, 0,  cx],
                       [0,  fx, cy],
                       [0,  0,  1]], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        # Time (seconds)
        current_time = datetime.now()
        delta_t = timedelta(seconds=0.1)

        for left_img_path, imu_path in zip(left_img_paths, imu_paths):
            assert left_img_path.name[:-5] == imu_path.name[:-4], "Left and IMU names are different"
            left_img = cv2.imread(left_img_path)
            rectified_left = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
            mask = (predict_single_image(predictor, rectified_left, torch.zeros((IMG_HEIGHT, IMG_WIDTH))) == 0).astype(np.uint8)
            left_gray = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY) 
            # left_gray = clahe.apply(left_gray)
            left_pts = cv2.goodFeaturesToTrack(left_gray, 30, 0.01, 5, mask=mask)

            # This stabilization is not exactly correct because the camera and IMU are not aligned
            R_cam = imu_to_rotation(imu_path)
            H = (K @ R_cam.T @ K_inv).astype(np.float32)            

            left_pts_stabilized = cv2.perspectiveTransform(left_pts, H)

            # Update tracker
            detections = {Detection(state_vector=StateVector(m),
                                    timestamp=current_time)
                        for m in left_pts_stabilized}
            _, tracks = tracker.update_tracker(current_time, detections)

            # canvas = rectified_left * np.expand_dims(mask, -1)
            canvas = rectified_left * 1

            # Un-rotate the image (inverse of motion)
            h, w = rectified_left.shape[:2]
            canvas = cv2.warpPerspective(
                canvas, H, (w, h),
                flags=cv2.INTER_LINEAR,
            )
            
            for track in tracks:
                state_vector = track.state.state_vector
                x_hat, y_hat = float(state_vector[0]), float(state_vector[1])
                color = id_to_color(list(ord(c) for c in track.id))
                cv2.circle(canvas, (int(x_hat), int(y_hat)), 5, color, -1)

            
            cv2.imshow("Temporal tracking", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            current_time += delta_t

        cv2.destroyAllWindows()
    elif TASK == 'stereotemporal':

        ### Initialize tracker (same as "temporal" task, ignoring disparity)

        empty_detector = iter(())                       # any zero-length iterable works
        tracker = get_tracker(detector=empty_detector)

        fx = float(Q[2, 3])                 # focal length in pixels
        cx = float(-Q[0, 3])                # principal point x
        cy = float(-Q[1, 3])                # principal point y
        K  = np.array([[fx, 0,  cx],
                       [0,  fx, cy],
                       [0,  0,  1]], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        # Time (seconds)
        current_time = datetime.now()
        delta_t = timedelta(seconds=0.1)

        for left_img_path, right_img_path, imu_path in zip(left_img_paths, right_img_paths, imu_paths):

            ### Get stereo correspondences (same as "stereo" task)
            # Assert names match
            assert left_img_path.name[:-5] == right_img_path.name[:-5], "Left and right images have different indices"
            assert left_img_path.name[:-5] == imu_path.name[:-4], "Left and IMU names are different"
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)

            rectified_left, rectified_right = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
            left_gray, right_gray = clahe_pair(rectified_left, rectified_right, clahe)

            mask = (predict_single_image(predictor, rectified_left, torch.zeros((IMG_HEIGHT, IMG_WIDTH))) == 0).astype(np.uint8)
            mask_img = mask*255
            # cv2.imshow("Obstacle mask", mask_img)

            pts = get_sparse_matches(left_gray, right_gray, mask=mask, y_tol=None, check_x_disp=False, fb_tol=4)

            ### For now, just average positions (no disparity)
            avg_pts = []
            for left_pt, right_pt, err in pts:
                avg_pts.append([(left_pt+right_pt)/2])
            avg_pts =  np.array(avg_pts)

            ### Stabilize points
            R_cam = imu_to_rotation(imu_path)
            H = (K @ R_cam.T @ K_inv).astype(np.float32)            
            avg_pts_stabilized = cv2.perspectiveTransform(avg_pts, H)

            ### Track points
            detections = {Detection(state_vector=StateVector(m),
                                    timestamp=current_time)
                        for m in avg_pts_stabilized}
            _, tracks = tracker.update_tracker(current_time, detections)

            ### Stabilize images for display
            h, w = rectified_left.shape[:2]
            stabilized_left = cv2.warpPerspective(
                rectified_left, H, (w, h),
                flags=cv2.INTER_LINEAR
            )
            stabilized_right = cv2.warpPerspective(
                rectified_right, H, (w, h),
                flags=cv2.INTER_LINEAR
            )

            canvas = cv2.hconcat([stabilized_left, stabilized_right])

            # Just plot average position on both images
            # Getting original L/R points is a bit more work
            for track in tracks:
                state_vector = track.state.state_vector
                x_hat, y_hat = float(state_vector[0]), float(state_vector[1])
                color = id_to_color(list(ord(c) for c in track.id))
                cv2.circle(canvas, (int(x_hat), int(y_hat)), 5, color, -1)
                cv2.circle(canvas, (w+int(x_hat), int(y_hat)), 5, color, -1)

            canvas = cv2.resize(canvas, (0, 0), fx=0.7, fy=0.7)
            cv2.imshow("Stereo-temporal tracking", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            current_time += delta_t

        cv2.destroyAllWindows()
    else:
        print(f"Unrecognized task {TASK}")



if __name__ == "__main__":
    main()

