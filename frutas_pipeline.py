from pathlib import Path
import argparse
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
import kagglehub

SEED = 42
DATASET_SLUG = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
DATA_DIR = Path.cwd() / "fruit_binary_yolo_cls"
BASE_MODEL = "yolo11n-cls.pt"
DEFAULT_RUN_NAME = "fruit_hs_vs_rt_cls_simple_ft5"
DETECTOR_WEIGHTS = "yolo11n.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRODUCE_CLASS_NAMES = {"apple", "banana", "orange", "broccoli", "carrot"}


def choose_device():
    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
        return 0
    print("Usando CPU")
    return "cpu"


def find_dataset_root(download_path):
    candidates = [download_path] + [path for path in download_path.iterdir() if path.is_dir()]
    for candidate in candidates:
        subdirs = [path.name for path in candidate.iterdir() if path.is_dir()]
        has_healthy = any(name.endswith("_Healthy") for name in subdirs)
        has_rotten = any(name.endswith("_Rotten") for name in subdirs)
        if has_healthy and has_rotten:
            return candidate
    raise FileNotFoundError("No se encontro una carpeta con estructura *_Healthy / *_Rotten")


def collect_images(dataset_root):
    grouped_paths = defaultdict(list)
    for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        name = class_dir.name
        if name.endswith("_Healthy"):
            label = "Healthy"
        elif name.endswith("_Rotten"):
            label = "Rotten"
        else:
            continue

        files = [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
        grouped_paths[label].extend(files)
    return grouped_paths


def stratified_split(items_by_label, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    rng = random.Random(seed)
    split = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for label, items in items_by_label.items():
        items = items.copy()
        rng.shuffle(items)

        count = len(items)
        count_train = int(count * train_ratio)
        count_val = int(count * val_ratio)

        split["train"][label] = items[:count_train]
        split["val"][label] = items[count_train:count_train + count_val]
        split["test"][label] = items[count_train + count_val:]

    return split


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_dataset(output_root=DATA_DIR):
    random.seed(SEED)
    raw_download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    dataset_root = find_dataset_root(raw_download_path)
    grouped_paths = collect_images(dataset_root)
    splits = stratified_split(grouped_paths)

    if output_root.exists():
        shutil.rmtree(output_root)

    for split_name in ["train", "val", "test"]:
        for label in ["Healthy", "Rotten"]:
            (output_root / split_name / label).mkdir(parents=True, exist_ok=True)

    jobs = []
    for split_name, split_labels in splits.items():
        for label, paths in split_labels.items():
            out_dir = output_root / split_name / label
            for index, src in enumerate(paths):
                dst = out_dir / f"{label.lower()}_{index:07d}{src.suffix.lower()}"
                jobs.append((src, dst))

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as executor:
        list(tqdm(executor.map(lambda item: link_or_copy(*item), jobs), total=len(jobs), desc="Preparando dataset"))

    print("Dataset listo en:", output_root)


def choose_sample_image(test_root=DATA_DIR / "test"):
    for label in ["Healthy", "Rotten"]:
        label_dir = test_root / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                return path
    raise FileNotFoundError("No se encontro una imagen de prueba en DATA_DIR/test")


def metrics_to_dataframe(split_name, metrics):
    rows = []
    results_dict = getattr(metrics, "results_dict", {}) or {}
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            rows.append({
                "split": split_name,
                "metric": key,
                "value": float(value),
            })
    return pd.DataFrame(rows)


def save_metrics_artifacts(run_name, val_metrics, test_metrics):
    metrics_df = pd.concat(
        [metrics_to_dataframe("val", val_metrics), metrics_to_dataframe("test", test_metrics)],
        ignore_index=True,
    )

    run_dir = Path("runs/classify") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "metrics_summary.csv"
    png_path = run_dir / "metrics_summary.png"
    metrics_df.to_csv(csv_path, index=False)

    if not metrics_df.empty:
        pivot_df = metrics_df.pivot(index="metric", columns="split", values="value")
        ax = pivot_df.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Metricas de validacion y test")
        ax.set_ylabel("Valor")
        ax.set_xlabel("Metrica")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()

    print("Metricas guardadas en:", csv_path)
    if png_path.exists():
        print("Grafico guardado en:", png_path)
    return metrics_df, csv_path, png_path


def train_classifier(run_name=DEFAULT_RUN_NAME, epochs=5, batch=32):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No existe dataset: {DATA_DIR}")

    device = choose_device()
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_DIR),
        task="classify",
        imgsz=640,
        epochs=epochs,
        batch=batch,
        device=device,
        project="runs/classify",
        name=run_name,
        pretrained=True,
        seed=SEED,
        deterministic=True,
        verbose=True,
        workers=0,
        exist_ok=True,
    )

    best_weights = Path("runs/classify") / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"No existe best.pt en {best_weights}")

    model = YOLO(str(best_weights))
    val_metrics = model.val(data=str(DATA_DIR), split="val", imgsz=640, device=device)
    test_metrics = model.val(data=str(DATA_DIR), split="test", imgsz=640, device=device)
    metrics_df, csv_path, png_path = save_metrics_artifacts(run_name, val_metrics, test_metrics)

    print("Best weights:", best_weights)
    return best_weights, val_metrics, test_metrics, metrics_df, csv_path, png_path


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def detect_and_count(image_path, detector, classifier, device, det_conf=0.20, det_iou=0.60, min_box_area=32 * 32, only_produce=True):
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer imagen: {image_path}")

    height, width = bgr.shape[:2]
    det_result = detector.predict(
        source=bgr,
        conf=det_conf,
        iou=det_iou,
        device=device,
        verbose=False,
    )[0]

    instances = []
    healthy_count = 0
    rotten_count = 0
    names = det_result.names

    for box in det_result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)

        area = (x2 - x1) * (y2 - y1)
        if area < min_box_area:
            continue

        class_index = int(box.cls[0].item())
        det_name = names[class_index]
        det_score = float(box.conf[0].item())

        if only_produce and PRODUCE_CLASS_NAMES is not None and det_name not in PRODUCE_CLASS_NAMES:
            continue

        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_result = classifier.predict(source=crop, imgsz=640, device=device, verbose=False)[0]
        top_index = int(cls_result.probs.top1)
        health_label = cls_result.names[top_index]
        health_conf = float(cls_result.probs.top1conf.item())

        label_lower = health_label.lower()
        if "healthy" in label_lower:
            healthy_count += 1
            box_color = (60, 180, 75)
        elif "rotten" in label_lower:
            rotten_count += 1
            box_color = (230, 25, 75)
        else:
            box_color = (255, 165, 0)

        text = f"{health_label} {health_conf:.2f} | {det_name} {det_score:.2f}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(bgr, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        instances.append({
            "bbox_xyxy": (x1, y1, x2, y2),
            "det_class": det_name,
            "det_conf": det_score,
            "health_label": health_label,
            "health_conf": health_conf,
        })

    summary = {
        "healthy": healthy_count,
        "rotten": rotten_count,
        "total": healthy_count + rotten_count,
    }
    return bgr, instances, summary


def run_single_inference(image_path, run_name=DEFAULT_RUN_NAME, output_path=None):
    device = choose_device()
    best_weights = Path("runs/classify") / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"No existe best.pt en {best_weights}")

    detector = YOLO(DETECTOR_WEIGHTS)
    classifier = YOLO(str(best_weights))
    annotated, instances, summary = detect_and_count(image_path, detector, classifier, device)

    if output_path is not None:
        cv2.imwrite(str(output_path), annotated)
        print("Imagen guardada en:", output_path)

    print(f"Conteo final -> {summary['rotten']} podridas, {summary['healthy']} sanas")
    print(f"Instancias analizadas: {summary['total']}")
    return annotated, instances, summary


def run_demo_inference(run_name=DEFAULT_RUN_NAME, output_path="sample_result.jpg"):
    sample_image = choose_sample_image()
    print("Imagen de prueba:", sample_image)
    output_image = Path(output_path) if output_path else None
    return run_single_inference(sample_image, run_name=run_name, output_path=output_image)


def run_batch_inference(input_dir, output_dir, run_name=DEFAULT_RUN_NAME):
    device = choose_device()
    best_weights = Path("runs/classify") / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"No existe best.pt en {best_weights}")

    detector = YOLO(DETECTOR_WEIGHTS)
    classifier = YOLO(str(best_weights))

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in sorted(input_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        annotated, instances, summary = detect_and_count(image_path, detector, classifier, device)
        out_image = output_dir / image_path.name
        cv2.imwrite(str(out_image), annotated)

        rows.append({
            "image": image_path.name,
            "healthy": summary["healthy"],
            "rotten": summary["rotten"],
            "total": summary["total"],
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "counts.csv"
    df.to_csv(csv_path, index=False)
    print("CSV generado en:", csv_path)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Pipeline de frutas sano vs podrido")
    parser.add_argument("--prepare-dataset", action="store_true", help="Descargar y preparar el dataset")
    parser.add_argument("--train", action="store_true", help="Entrenar el clasificador")
    parser.add_argument("--all", action="store_true", help="Preparar dataset y entrenar")
    parser.add_argument("--image", type=str, help="Ruta de una imagen para inferencia")
    parser.add_argument("--demo-image", action="store_true", help="Usar automaticamente una imagen del split test")
    parser.add_argument("--input-dir", type=str, help="Carpeta de imagenes para inferencia batch")
    parser.add_argument("--output-dir", type=str, default="inference_outputs", help="Carpeta de salida para inferencia batch")
    parser.add_argument("--output-image", type=str, help="Ruta de salida para una sola imagen")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME, help="Nombre del run de entrenamiento")
    parser.add_argument("--epochs", type=int, default=5, help="Numero de epocas")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    if args.all or args.prepare_dataset:
        prepare_dataset()

    if args.all or args.train:
        train_classifier(run_name=args.run_name, epochs=args.epochs, batch=args.batch)

    if args.image:
        output_image = Path(args.output_image) if args.output_image else None
        run_single_inference(args.image, run_name=args.run_name, output_path=output_image)

    if args.demo_image:
        output_image = args.output_image if args.output_image else "sample_result.jpg"
        run_demo_inference(run_name=args.run_name, output_path=output_image)

    if args.input_dir:
        run_batch_inference(args.input_dir, args.output_dir, run_name=args.run_name)

    if not any([args.all, args.prepare_dataset, args.train, args.image, args.demo_image, args.input_dir]):
        parser.print_help()


if __name__ == "__main__":
    main()
