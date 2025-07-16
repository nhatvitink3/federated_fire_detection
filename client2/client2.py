import flwr as fl
import torch
import argparse
from ultralytics import YOLO
import cv2


class YOLOv8Client(fl.client.NumPyClient):
    def __init__(self, model_path, data_yaml):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.model.parameters(), parameters):
            p.data.copy_(torch.from_numpy(new_p))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train(data=self.data_yaml, epochs=config.get("epochs", 1), imgsz=320, batch=8)
        num_params = sum(p.numel() for p in self.model.model.parameters())
        return self.get_parameters(config), num_params, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.model.val(data=self.data_yaml, imgsz=320, batch=8)
        return 0.0, sum(p.numel() for p in self.model.model.parameters()), {"accuracy": metrics.box.map50}


def demo_webcam(model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Fire Detection Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to data.yaml file")
    parser.add_argument('--model', type=str, default="../model/yolov8s.pt", help="Path to YOLOv8 model")
    parser.add_argument('--demo_webcam', action='store_true', help="Run webcam demo")
    args = parser.parse_args()

    if args.demo_webcam:
        demo_webcam(args.model)
    else:
        client = YOLOv8Client(args.model, args.data)
        fl.client.start_client(server_address="localhost:8080", client=client.to_client())

