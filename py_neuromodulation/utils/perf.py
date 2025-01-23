from typing import Any
import time
import threading
import logging
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class MetricPoint:
    timestamp: float
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricBuffer:
    """Maintains a rolling buffer of metric values with timestamps."""

    def __init__(self, max_size: int = 1000):
        self.buffer: deque[MetricPoint] = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, value: float, metadata: dict[str, Any] | None = None):
        with self.lock:
            self.buffer.append(
                MetricPoint(timestamp=time.time(), value=value, metadata=metadata or {})
            )

    def get_stats(self, window_seconds: float | None = None) -> dict[str, float]:
        with self.lock:
            if not self.buffer:
                return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std_dev": 0.0}

            current_time = time.time()
            values = [
                point.value
                for point in self.buffer
                if window_seconds is None
                or (current_time - point.timestamp) <= window_seconds
            ]

            if not values:
                return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std_dev": 0.0}

            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }


class PerformanceMonitor:
    """Centralized system for tracking performance metrics across the application."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger("PyNM.Performance")
            self.metrics: dict[str, MetricBuffer] = {}
            self.timers: dict[str, float] = {}
            self.counters: dict[str, int] = {}
            self.metrics_lock = threading.Lock()
            self.initialized = True

    def record_metric(
        self, name: str, value: float, metadata: dict[str, Any] | None = None
    ):
        """Record a metric value with optional metadata."""
        with self.metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = MetricBuffer()
            self.metrics[name].add(value, metadata)

    def start_timer(self, name: str):
        """Start a timer for measuring operation duration."""
        self.timers[name] = time.time()

    def stop_timer(self, name: str, record: bool = True) -> float:
        """Stop a timer and optionally record its duration as a metric."""
        if name not in self.timers:
            raise KeyError(f"Timer '{name}' was never started")

        duration = time.time() - self.timers[name]
        if record:
            self.record_metric(f"{name}_duration", duration)

        del self.timers[name]
        return duration

    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter by the specified amount."""
        with self.metrics_lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def get_counter(self, name: str) -> int:
        """Get the current value of a counter."""
        return self.counters.get(name, 0)

    def get_metric_stats(
        self, name: str, window_seconds: float | None = None
    ) -> dict[str, float]:
        """Get statistics for a metric over the specified time window."""
        if name not in self.metrics:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std_dev": 0.0}
        return self.metrics[name].get_stats(window_seconds)

    def get_all_metrics(
        self, window_seconds: float | None = None
    ) -> dict[str, dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            name: self.get_metric_stats(name, window_seconds) for name in self.metrics
        }

    def log_summary(self, window_seconds: float | None = None):
        """Log a summary of all metrics and counters."""
        stats = self.get_all_metrics(window_seconds)
        self.logger.info("Performance Summary:")

        for name, metric_stats in stats.items():
            self.logger.info(f"{name}:")
            for stat_name, value in metric_stats.items():
                self.logger.info(f"  {stat_name}: {value:.3f}")

        self.logger.info("Counters:")
        for name, value in self.counters.items():
            self.logger.info(f"  {name}: {value}")


# Example usage:
# monitor = PerformanceMonitor()
#
# # Record individual metrics
# monitor.record_metric("queue_size", queue.qsize())
#
# # Time operations
# monitor.start_timer("websocket_send")
# await websocket.send_bytes(data)
# duration = monitor.stop_timer("websocket_send")
#
# # Track message counts
# monitor.increment_counter("messages_received")
#
# # Get stats for the last 5 minutes
# stats = monitor.get_metric_stats("websocket_send", window_seconds=300)
