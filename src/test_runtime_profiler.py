# test_runtime_profiler.py
import time
import threading
from runtime_profiler import RuntimeProfiler

def test_profiler_mock():
    profiler = RuntimeProfiler(mock=True)
    profiler.start()

    # Simulate tokens arriving from inference requests
    def fake_inference():
        for _ in range(10):
            time.sleep(0.5)
            profiler.notify_tokens_generated(50)

    t = threading.Thread(target=fake_inference)
    t.start()
    t.join()

    # Let the profiler take a few samples
    time.sleep(3)

    estimate = profiler.current_estimate()
    print("Current estimate:", estimate)
    assert estimate["status"] in ("ready", "warming_up")

    summary = profiler.stop()
    print("Frontier points:", summary["frontier"])
    print("Total snapshots:", len(summary["snapshots"]))
    assert len(summary["snapshots"]) > 0, "Should have collected at least one snapshot"
    print("Runtime profiler test passed ✓")

if __name__ == "__main__":
    test_profiler_mock()