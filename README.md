# Federated Fire and Smoke Detection

This project demonstrates a simple federated learning setup using Flower and YOLOv8 for fire and smoke detection.

## Structure
- `server.py`: Flower server script.
- `client.py`,`client1/client1.py`,`client2/client2.py`: Flower client script using YOLOv8.
- `model/yolov8_fire.pt`: Placeholder for pretrained YOLOv8 fire detection model.
- `client1/dataset` and `client2/dataset`: Separate datasets for different clients.
- `requirements.txt`: Required Python packages.

## Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `python server.py`
3. Run the clients (in separate terminals): 
   ```
   cd client1
   python client1.py --data data/client1/data.yaml
   cd client2
   python client2.py --data data/client2/data.yaml
   ```
