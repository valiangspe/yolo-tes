import hydra
import torch
import cv2
from random import randint
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


# Helper function to calculate Euclidean distance
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def draw_boxes_with_speed(
    img,
    bbox,
    identities=None,
    categories=None,
    names=None,
    previous_positions=None,
    fps=30,
    offset=(0, 0),
    distance_to_street=10,  # Distance from camera to street in meters
    correction_factor=5,  # Correction factor for the camera angle and perspective
):
    speeds = {}
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        category = names[int(categories[i])]
        box_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

        # Speed calculation: track position change between frames
        if id in previous_positions:
            prev_center = previous_positions[id]
            distance = euclidean_distance(box_center, prev_center)

            speed_px_per_sec = distance * fps
            # Adjust speed with correction factor and distance to street
            speed_real_world = speed_px_per_sec / (
                correction_factor * (distance_to_street / 10)
            )
            speeds[id] = speed_real_world
        else:
            speed_real_world = 0  # First frame, no speed

        # Save current position as previous for next frame
        previous_positions[id] = box_center

        # label = f"{category} #{id} | {speed:.0f} px/s"
        label = f"{category} #{id} | {speed_real_world:.0f} km/h"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        bounding_color = (0, 255, 253)

        if category != "car" and category != "truck":
            bounding_color = (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), bounding_color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1
        )

    return img, speeds


# Initialize tracker for previous positions
previous_positions = {}


tracker = None


def init_tracker():
    global tracker

    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(
        max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh
    )


rand_color_list = []


def draw_boxes(
    img,
    bbox,
    identities=None,
    categories=None,
    names=None,
    offset=(0, 0),
):

    for i, box in enumerate(bbox):
        # print(categories[i])

        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        box_center = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id)
        category = names[int(categories[i])]
        full_label_name = f"{category} #{label}"
        (w, h), _ = cv2.getTextSize(full_label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        bounding_color = (0, 255, 253)

        if category != "car" and category != "truck":
            bounding_color = (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), bounding_color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(
            img,
            full_label_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            [255, 255, 255],
            1,
        )

    return img


def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    # ......................................


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(
            img, line_width=self.args.line_thickness, example=str(self.model.names)
        )

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):

        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f"{idx}: "
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, "frame", 0)
        # tracker
        self.data_path = p

        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / "labels" / p.stem) + (
            "" if self.dataset.mode == "image" else f"_{frame}"
        )
        log_string += "%gx%g " % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # #..................USE TRACK FUNCTION....................
        dets_to_sort = np.empty((0, 6))

        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack(
                (dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass]))
            )

        tracked_dets = tracker.update(dets_to_sort)
        tracks = tracker.getTrackers()

        for track in tracks:
            [
                cv2.line(
                    im0,
                    (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                    (
                        int(track.centroidarr[i + 1][0]),
                        int(track.centroidarr[i + 1][1]),
                    ),
                    rand_color_list[track.id],
                    thickness=3,
                )
                for i, _ in enumerate(track.centroidarr)
                if i < len(track.centroidarr) - 1
            ]

        if len(tracked_dets) > 0:
            # print(tracked_dets)
            bbox_xyxy = tracked_dets[:, :4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]

            # draw_boxes(
            #     im0,
            #     bbox_xyxy,
            #     identities,
            #     categories,
            #     self.model.names,
            # )

            # Pass the previous positions and FPS to draw_boxes_with_speed
            im0, speeds = draw_boxes_with_speed(
                im0,
                bbox_xyxy,
                identities,
                categories,
                self.model.names,
                previous_positions,
                fps=30,
                distance_to_street=10,  # You can adjust this distance based on the camera setup
                correction_factor=5,  # Adjust based on camera angle and perspective
            )

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        print(log_string)
        return log_string


@hydra.main(
    version_base=None,
    config_path=str(DEFAULT_CONFIG.parent),
    config_name=DEFAULT_CONFIG.name,
)
def predict(cfg):
    init_tracker()
    random_color_list()

    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
