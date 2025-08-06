import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import cv2
import yaml

from datasets.mastr import MaSTr1325Dataset
from datasets.mods import MODSDataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights

# Usage: python .\predict.py images --model ewasr_resnet18 --weights .\pretrained\ewasr_resnet18.pth --output_dir .\output\mastr1325\ --dataset mastr --dataset_config .\configs\mastr1325_val.yaml

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL = 'wasr_resnet18_imu'


def add_common_model_args(parser: argparse.ArgumentParser):
    """Arguments shared by both subcommands."""
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--model", type=str, choices=models.model_list, default=DEFAULT_MODEL,
                        help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory (used for saving masks or video).")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--mixer", type=str, default="CCCCSS",
                        help="Token mixers in feature mixer.")
    parser.add_argument("--project", action='store_true',
                        help="Project encoder features to fewer channels.")
    parser.add_argument("--enricher", type=str, default="SS",
                        help="Token mixers in long-skip feature enricher.")


def get_arguments():
    parser = argparse.ArgumentParser(description="WaSR Inference (video-only or image-dataset)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # images subcommand
    img_p = subparsers.add_parser("images", help="Run inference on an image dataset (MaSTr/MODS).")
    add_common_model_args(img_p)
    img_p.add_argument("--dataset", type=str, choices=["mastr", "mods"], required=True)
    img_p.add_argument("--dataset_config", type=str, required=True,
                       help="Path to the dataset mapping/config file.")
    img_p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Minibatch size used on each device.")

    # video subcommand
    vid_p = subparsers.add_parser("video", help="Run inference on a single video file only.")
    add_common_model_args(vid_p)
    vid_p.add_argument("--video", type=str, required=True, help="Path to input video file.")
    vid_p.add_argument("--display", action='store_true', help="Display prediction in real-time.")
    vid_p.add_argument("--save", action='store_true', help="Write output side-by-side video.")
    vid_p.add_argument("--save_path", type=str,
                       help="Optional explicit path for saved video. Defaults to output_dir/prediction_<name>.mp4")
    vid_p.add_argument("--downscale", type=float, default=1.0, help="Scale factor for resizing video frames before inference.")

    return parser.parse_args()


def build_predictor(args) -> Predictor:
    model = models.get_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=False,
        mixer=args.mixer,
        enricher=args.enricher,
        project=args.project
    )
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    return Predictor(model, args.fp16)


def process_images(args, predictor: Predictor):
    if args.dataset == "mods":
        dataset = MODSDataset(args.dataset_config, normalize_t=PytorchHubNormalization())
    else:
        dataset = MaSTr1325Dataset(args.dataset_config, normalize_t=PytorchHubNormalization(), include_original=False)

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = [time.time()]
    for features, _labels in dl:
        _pred_masks = predictor.predict_batch(features)
        # Save predictions
        for i, (pred_mask, img_name) in enumerate(zip(_pred_masks, _labels['img_name'])):
            output_path = output_dir / f"{img_name}_pred.png"
            pred_mask_rgb = SEGMENTATION_COLORS[pred_mask]  # Convert class indices to RGB
            Image.fromarray(pred_mask_rgb).save(output_path)
            print(f"Saved prediction for {img_name} to {output_path}")
        times.append(time.time())

    for i in range(len(times) - 1):
        print(f"Batch {i}: {times[i + 1] - times[i]:.3f} seconds")

def maps_from_yaml(calib_path: Path):
    """Load calibration data from a YAML file and constuct rectification maps."""
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    M1 = fs.getNode('M1').mat()  # Left camera matrix
    D1 = fs.getNode('D1').mat()  # Left distortion coefficients
    M2 = fs.getNode('M2').mat()  # Right camera matrix
    D2 = fs.getNode('D2').mat()  # Right distortion coefficients
    R = fs.getNode('R').mat()    # Rotation matrix
    T = fs.getNode('T').mat()    # Translation vector
    img_size = (1278, 958)  # TODO: load properly from calibration file
    fs.release()

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, D1, M2, D2, img_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, img_size, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, img_size, cv2.CV_32FC1)

    return mapx1, mapy1, mapx2, mapy2

def process_mods(args, predictor: Predictor):
    mods_dir = Path('data') / 'MODS'
    # session, timestamp = ('kope100', '00011830-00012500')
    # session, timestamp = ('kope100', '00006790-00007090')
    session, timestamp = ('stru02', '00097380-00097740')
    calib_path = mods_dir / 'calibration' / f'calibration-{session}.yaml'

    mapx1, mapy1, mapx2, mapy2 = maps_from_yaml(calib_path)

    seq_name = f"{session}-{timestamp}"
    process_mods_seq(args, predictor, mods_dir, seq_name, mapx1, mapy1, mapx2, mapy2)

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

def img_and_masked(img, pred_mask, pred_mask_rgb):
    """Return the original image and the image masked by the prediction mask."""
    class0 = np.expand_dims(pred_mask == 0, -1)
    class1 = np.expand_dims(pred_mask == 1, -1)
    class2 = np.expand_dims(pred_mask == 2, -1)
    img_class0 = ((img * class0) + (pred_mask_rgb * (1-class0))).astype(np.uint8)
    img_class1 = ((img * class1) + (pred_mask_rgb * (1-class1))).astype(np.uint8)
    img_class2 = ((img * class2) + (pred_mask_rgb * (1-class2))).astype(np.uint8)
    return np.vstack((np.hstack((img, img_class0)), np.hstack((img_class1, img_class2))))

def masked_only(img, pred_mask):
    class0 = np.expand_dims(pred_mask == 0, -1)
    return img * class0

def process_mods_seq(args, predictor: Predictor, mods_dir: Path, seq_name: str, mapx1, mapy1, mapx2, mapy2, save_imgs=True, show_imgs=True):

    seq_dir = mods_dir / 'sequences' / seq_name
    frame_dir = seq_dir / 'frames'
    imu_dir = seq_dir / 'imus'
    output_subdir = Path(args.output_dir) / seq_name
    if save_imgs:
        output_subdir.mkdir(parents=True, exist_ok=True)

    mods_names = [f.stem for f in imu_dir.glob('*.png')]
    print("Names:", mods_names)
    for img_name in mods_names:
        img_path = frame_dir / f"{img_name}.jpg"
        img = np.array(Image.open(img_path))

        right_img_name = img_name[:-1] + 'R' # Replace 'L' with 'R'
        right_img_path = frame_dir / f"{right_img_name}.jpg"
        right_img = np.array(Image.open(right_img_path))

        # Apply rectification
        img = cv2.remap(img, mapx1, mapy1, cv2.INTER_LINEAR)
        right_img = cv2.remap(right_img, mapx2, mapy2, cv2.INTER_LINEAR)

        # Save rectified images
        if save_imgs:
            rect_dir = output_subdir / 'framesRectified'
            rect_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(rect_dir / f"{img_name}.jpg")
            Image.fromarray(right_img).save(rect_dir / f"{right_img_name}.jpg")

        imu_path = imu_dir / f"{img_name}.png"
        imu_mask = np.array(Image.open(imu_path))
        imu_mask = torch.from_numpy(imu_mask.astype(bool))

        pred_mask = predict_single_image(predictor, img, imu_mask)

        if save_imgs:
            mask_dir = output_subdir / 'masks'
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred_mask).save(mask_dir / f"{img_name}.png")

        pred_mask_rgb = SEGMENTATION_COLORS[pred_mask]
        if save_imgs:
            mask_rgb_dir = output_subdir / 'masks_rgb'
            mask_rgb_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred_mask_rgb).save(mask_rgb_dir / f"{img_name}.png")

        if show_imgs:
            result = img_and_masked(img, pred_mask, pred_mask_rgb)
            result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow('MODS Prediction', result)
            cv2.waitKey(1)

def process_video(args, predictor: Predictor):
    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = None
    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = out_dir / f"prediction_{Path(args.video).stem}.mp4"
        out_path = Path(args.save_path) if args.save_path else default_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width * 2, frame_height))
        print(f"Saving to: {out_path}")

    normalize_t = PytorchHubNormalization()
    times = [time.time()]

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_og = frame.copy()

        if args.downscale != 1.0:
            new_size = (int(frame_width / args.downscale), int(frame_height / args.downscale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        times.append(time.time())
        print("Read time:", times[-1] - times[-2])

        # Prepare frame for prediction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = normalize_t(frame_pil).unsqueeze(0)

        # Build features dict compatible with the model (dummy IMU for *_imu models)
        features = {
            "image": frame_tensor,
            "imu": torch.zeros(1, 6),
            "imu_mask": torch.ones(1),
        }

        times.append(time.time())
        print("Preprocessing time:", times[-1] - times[-2])

        # Predict
        pred_probs = predictor.predict_batch_probs(features)[0]

        # Convert classes to RGB mask
        pred_probs = np.permute_dims(pred_probs, (1, 2, 0)) # (height, width, num_classes)
        pred_probs = cv2.resize(pred_probs, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        print("Shape of resized prediction probabilities:", pred_probs.shape)
        print(np.min(pred_probs), np.max(pred_probs))
        '''
        pred_mask = predict_single_image(predictor, frame_rgb, torch.ones(1))

        times.append(time.time())
        print("Prediction time:", times[-1] - times[-2])

        '''
        # Compute colored_mask as a weighted combination of SEGMENTATION_COLORS by pred_probs
        # pred_probs: (H, W, num_classes)
        print(pred_probs.shape, SEGMENTATION_COLORS.shape)
        colored_mask = np.tensordot(pred_probs, SEGMENTATION_COLORS, axes=(2, 0))
        colored_mask = np.clip(colored_mask, 0, 255).astype(np.uint8)

        side_by_side = np.hstack((frame_og, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)))
        '''
        result = masked_only(frame_rgb, pred_mask)

        if args.display:
            # Lighter-weight preview
            preview = cv2.resize(result, (0, 0), fx=0.3, fy=0.3)
            cv2.imshow('Video Prediction', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out is not None:
            # out.write(result)
            out_dir = Path(args.output_dir)
            img_name = f"{frame_idx:04d}.jpg"
            Image.fromarray(result).save(out_dir / img_name)

        times.append(time.time())
        print("Display/Write time:", times[-1] - times[-2])
        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def main():
    print("OpenCV version:", cv2.__version__)
    args = get_arguments()
    print(args)

    predictor = build_predictor(args)

    # Usage:
    # python .\predict.py images --model ewasr_resnet18 --weights .\pretrained\ewasr_resnet18.pth --output_dir .\output\mods\ --dataset mastr --dataset_config .\configs\mastr1325_val.yaml
    process_mods(args, predictor)
    exit()

    if args.mode == "video":
        # Video-only mode: Skip any dataset work.
        # Usage:
        # python .\predict.py video --model ewasr_resnet18 --weights .\pretrained\ewasr_resnet18.pth --video .\data\VIS_Onshore\Videos\MVI_1448_VIS_Haze.avi --display
        process_video(args, predictor)
    elif args.mode == "images":
        process_images(args, predictor)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
