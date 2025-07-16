import flwr as fl
import pickle
import os

# Cấu hình huấn luyện mỗi client
def fit_config(rnd):
    return {"epochs": 1}

# Hàm tính trung bình trọng số cho đánh giá mô hình
def weighted_average(metrics):
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_acc = sum([num_examples * m["accuracy"] for num_examples, m in metrics]) / total_examples
    return {"accuracy": weighted_acc}

# Tạo strategy kế thừa FedAvg để thêm cơ chế lưu mô hình toàn cục
class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            # Lưu mô hình toàn cục bằng pickle
            path = f"model/global_model_round_{rnd}.pkl"
            with open(path, "wb") as f:
                pickle.dump(aggregated_parameters, f)
            print(f"[Server] ✅ Đã lưu mô hình toàn cục: {path}")

        return aggregated_parameters, aggregated_metrics

# Sử dụng strategy đã mở rộng
strategy = SaveModelFedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Khởi chạy server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy
)
