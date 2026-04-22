"""
Shared boto3 S3 client factory.

Centralizes connection settings so services reuse one place for credentials,
region, and retry policy.
"""

from __future__ import annotations
from typing import Optional, Any
import boto3
from botocore.config import Config
from app.core.config import S3Config


class S3ClientFactory:
    """
    Centralized S3 client manager.
    Ensures a single reusable client instance.
    """

    _client: Optional[Any] = None

    def __init__(self, s3_config: Optional[S3Config] = None):
        self.config = s3_config or S3Config()

    def get_client(self) -> Any:
        """
        Returns a singleton S3 client.
        """
        if self.__class__._client is None:
            self.__class__._client = boto3.client(
                "s3",
                region_name=self.config.REGION_NAME or None,
                aws_access_key_id=self.config.ACCESS_KEY_ID or None,
                aws_secret_access_key=self.config.SECRET_ACCESS_KEY or None,
                aws_session_token=self.config.SESSION_TOKEN or None,
                config=Config(retries={"max_attempts": 5, "mode": "standard"}),
            )
        return self.__class__._client
