#!/usr/bin/env python3
from ultralytics import YOLO
import os
import argparse


class LicensePlateModel:
    def __init__(self, weights='yolov8n.pt', data_yaml='data.yaml'):
        """
        Initialize the license plate detection model.

        Args:
            weights: Initial weights file path, use pretrained YOLOv8 model by default
            data_yaml: Path to the data configuration file
        """
        self.weights = weights
        self.data_yaml = data_yaml
        self.model = None

    def train(self, epochs=100, img_size=640, batch_size=16, device='0', project='runs/train'):
        """
        Train the YOLOv8 model on license plate data.

        Args:
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Batch size
            device: Device to train on ('0', '0,1,2,3', 'cpu')
            project: Save directory
        """
        # Load model
        self.model = YOLO(self.weights)

        # Start training
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project,
            name='license_plates',
            patience=20,  # Early stopping patience
            save=True,  # Save checkpoints
            save_period=10,  # Save checkpoint every 10 epochs
            verbose=True  # Display training progress
        )

        # The model automatically saves checkpoints at every save_period and the best model
        print(f"Training complete. Model checkpoints saved every 10 epochs.")
        print(f"Best model saved at: {project}/license_plates/weights/best.pt")

        return results

    def evaluate(self, model_path=None):
        """
        Evaluate the model on validation set.

        Args:
            model_path: Path to the trained model, use the object's model if None
        """
        if model_path:
            model = YOLO(model_path)
        else:
            if self.model is None:
                raise ValueError("No model loaded. Please train or load a model first.")
            model = self.model

        # Run validation
        metrics = model.val(data=self.data_yaml)
        return metrics

    def predict(self, img_path, model_path=None, conf=0.25):
        """
        Run inference on an image.

        Args:
            img_path: Path to the image
            model_path: Path to the trained model, use the object's model if None
            conf: Confidence threshold
        """
        if model_path:
            model = YOLO(model_path)
        else:
            if self.model is None:
                raise ValueError("No model loaded. Please train or load a model first.")
            model = self.model

        # Run inference
        results = model(img_path, conf=conf)
        return results

    def save_model(self, path='best_model.pt'):
        """
        Save the model to a specified path.

        Args:
            path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")

        self.model.export(format='pt', save_dir=os.path.dirname(path))
        print(f"Model exported to {path}")


def main():
    # Get the project root directory (one level up from the script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Set default data path to the project root data.yaml
    default_data_path = os.path.join(project_dir, 'data.yaml')

    parser = argparse.ArgumentParser(description="Train YOLOv8 for License Plate Detection")
    parser.add_argument('--data', type=str, default=default_data_path, help='dataset yaml file')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='device (cuda device or cpu)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'predict'],
                        help='mode: train, validate or predict')
    parser.add_argument('--image', type=str, help='image path for prediction')
    args = parser.parse_args()

    print(f"Using data file: {args.data}")

    # Initialize the model
    lp_model = LicensePlateModel(weights=args.weights, data_yaml=args.data)

    # Execute requested mode
    if args.mode == 'train':
        print(f"Training YOLOv8 model with {args.data}...")
        results = lp_model.train(
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        )
        # Models are automatically saved every 10 epochs during training
        print(f"Training complete. Results saved to {results}")

    elif args.mode == 'val':
        print("Evaluating model...")
        metrics = lp_model.evaluate()
        print(f"Evaluation metrics: {metrics}")

    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image argument required for prediction mode")
            return

        print(f"Running inference on {args.image}...")
        results = lp_model.predict(args.image)
        results[0].show()
        print(f"Predictions: {results[0].boxes}")


if __name__ == "__main__":
    main()