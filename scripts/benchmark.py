from locust import HttpUser, between, task
import random
import socket
import struct


class APIUser(HttpUser):
    wait_time = between(2, 4)

    @task
    def lookup(self):
        self.client.get("/lookup?ip=" + socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff))),
                        name="/lookup")
