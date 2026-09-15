"""
Kafka Topic Configuration for Sentinel Gujarat.
Metadata-only messaging architecture with in-memory asyncio fallback for local deployments.
"""

from typing import Dict, List, Any

KAFKA_BROKERS = [
    "localhost:9092",
]

KAFKA_TOPICS = {
    "raw.detections": {
        "num_partitions": 30,
        "replication_factor": 1,
        "config": {"retention.ms": "86400000", "compression.type": "lz4"}
    },
    "reid.jobs": {
        "num_partitions": 10,
        "replication_factor": 1,
        "config": {"retention.ms": "21600000", "compression.type": "lz4"}
    },
    "danger.alerts": {
        "num_partitions": 5,
        "replication_factor": 1,
        "config": {"retention.ms": "604800000", "compression.type": "gzip"}
    },
    "officer.dispatch": {
        "num_partitions": 5,
        "replication_factor": 1,
        "config": {"retention.ms": "604800000", "compression.type": "gzip"}
    },
    "evidence.seal": {
        "num_partitions": 5,
        "replication_factor": 1,
        "config": {"retention.ms": "86400000", "compression.type": "gzip"}
    },
    "journey.events": {
        "num_partitions": 10,
        "replication_factor": 1,
        "config": {"retention.ms": "172800000", "compression.type": "lz4"}
    },
}

CONSUMER_GROUPS = {
    "reid-vehicle-workers": "raw.detections",
    "reid-person-workers": "raw.detections",
    "danger-score-workers": "reid.jobs",
    "dispatch-workers": "danger.alerts",
    "evidence-seal-workers": "evidence.seal",
    "journey-builder-workers": "journey.events",
}


def create_all_topics():
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKERS, client_id="sentinel-admin")
        topics = [
            NewTopic(name=k, num_partitions=v["num_partitions"], replication_factor=v["replication_factor"])
            for k, v in KAFKA_TOPICS.items()
        ]
        admin.create_topics(topics, validate_only=False)
        print(f"Created {len(topics)} Kafka topics")
    except Exception as e:
        print(f"Kafka topics notice: {e}")


if __name__ == "__main__":
    create_all_topics()
