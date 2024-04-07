from locust import HttpUser, task, between
import json


class APITestUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_internal_status(self):
        payload = {"externalStatus": "TERMINAL IN"}
        self.client.post("/predict", json=payload)


if __name__ == "__main__":
    import os

    os.system("locust -f locustfile.py")
