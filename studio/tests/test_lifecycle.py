import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from studio.lifecycle import ServiceLease, process_alive, process_identity


class LifecycleTests(unittest.TestCase):
    def test_one_service_per_checkout_and_lease_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.lock"
            lease = ServiceLease(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "already"):
                    ServiceLease(path)
            finally:
                lease.close()
            reopened = ServiceLease(path)
            reopened.close()

    def test_live_process_identity_and_pid_reuse(self):
        own = process_identity(os.getpid())
        self.assertTrue(process_alive(own))
        self.assertFalse(process_alive({**own, "created": own["created"] - 10}))
        with patch("studio.lifecycle.psutil.Process", side_effect=psutil.AccessDenied(os.getpid())):
            self.assertIsNone(process_alive(own))

    def test_child_completion_is_observed_not_inferred_from_status_file(self):
        # This child imports only the standard library; the test suite itself is
        # launched through the authoritative CUDA launcher.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.3)"],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        identity = process_identity(child.pid)
        self.assertTrue(process_alive(identity))
        self.assertEqual(child.wait(timeout=5), 0)
        self.assertFalse(process_alive(identity))
