import unittest

from utils.credentials import load_postgres_credentials, load_rabbitmq_credentials


class TestCredentialLoadings(unittest.TestCase):
    def test_postgresql_envs_check_type(self):
        postgres_creds = load_postgres_credentials()

        self.assertIsInstance(postgres_creds, dict)

    def test_postgresql_envs_values(self):
        postgres_creds = load_postgres_credentials()

        self.assertNotEqual(postgres_creds["user"], None)
        self.assertNotEqual(postgres_creds["password"], None)
        self.assertNotEqual(postgres_creds["host"], None)
        self.assertNotEqual(postgres_creds["port"], None)

        self.assertIsInstance(postgres_creds["user"], str)
        self.assertIsInstance(postgres_creds["password"], str)
        self.assertIsInstance(postgres_creds["host"], str)
        self.assertIsInstance(postgres_creds["port"], str)

    def test_rabbitmq_envs_check_type(self):
        rabbitmq_creds = load_rabbitmq_credentials()

        self.assertIsInstance(rabbitmq_creds, dict)

    def test_rabbitmq_envs_values(self):
        rabbitmq_creds = load_postgres_credentials()

        self.assertNotEqual(rabbitmq_creds["user"], None)
        self.assertNotEqual(rabbitmq_creds["password"], None)
        self.assertNotEqual(rabbitmq_creds["host"], None)
        self.assertNotEqual(rabbitmq_creds["port"], None)

        self.assertIsInstance(rabbitmq_creds["user"], str)
        self.assertIsInstance(rabbitmq_creds["password"], str)
        self.assertIsInstance(rabbitmq_creds["host"], str)
        self.assertIsInstance(rabbitmq_creds["port"], str)
