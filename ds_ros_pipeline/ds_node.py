# Import convention (pinned for every module in this package): ds_ros_pipeline/
# is NOT a Python package — there is no __init__.py and no relative imports.
# run.sh execs this file with the package directory as the script dir, so it is
# first on sys.path; every module imports siblings flat ("import frames",
# "import disk"). tests.py and any other external entrypoint must insert the
# package dir on sys.path before importing. Existing repo code is reached as
# "deepstream_yolo.*" via PYTHONPATH=$PWD/src (run.sh sets it).
"""Entrypoint (DESIGN.md Sec 2 thread model, Sec 9 wiring).

Owns process lifecycle: Gst.init, config resolution, building both pipelines,
wiring probes and collaborators to the rclpy node, thread startup/shutdown,
and bus handling for both pipelines — including the loop=false EOS ->
'ended' transition of Sec 5 (mark lifecycle ended, wake capture waiters,
finalize the recorder, keep publishing /ds/status; the process runs until
SIGINT).

Threads created here (Sec 2): the GLib.MainLoop runs on the main thread with
bus watches for both pipelines; "ros" runs MultiThreadedExecutor
(num_threads=8, Sec 4); "feeder" is started via source.start(); "batch" via
BatchWorker.start(); "disk" via DiskWorker.start().
"""

from __future__ import annotations

import signal
import sys
import threading
import traceback

import batch_pipeline
import config as config_mod
import disk
import frames
import infer_configs
import live_pipeline
import source as source_mod
import timestamps

EXECUTOR_THREADS = 8            # Sec 4 thread-budget analysis
SHUTDOWN_JOIN_TIMEOUT_S = 10.0  # bound every shutdown wait (slowest: 5 s drain)
PREBUILD_STATE_TIMEOUT_S = 600.0  # engine build is ~1-3 min (Sec 11 risk 10)
PREBUILD_WIDTH = 2560           # engine cache is dimension-independent; these
PREBUILD_HEIGHT = 1440          # just satisfy mux/appsrc caps for the warmup


class App:
    """Composition root; one instance per process.

    Build order (main thread): config <- rclpy.init + node params;
    infer_configs.write_batch_yolo_config; registry, lifecycle, grab state;
    source.create_source; live_pipeline.build + install_ingest_stamp_probe +
    grab probe (frames.make_grab_probe on caps_grab src pad) +
    connect_preview; batch_pipeline.build + collect probes + BatchWorker;
    disk.Recorder + DiskWorker; ros_io.DsRosNode wired with everything;
    bus watches; states to PLAYING; source.start(); workers; executor thread;
    GLib.MainLoop.run().
    """

    def __init__(self, argv: list[str]) -> None:
        self.argv = list(argv)
        self.exit_code = 0
        self.config: config_mod.PipelineConfig | None = None
        self.registry: timestamps.TimestampRegistry | None = None
        self.lifecycle: frames.Lifecycle | None = None
        self.grab: frames.GrabState | None = None
        self.source: source_mod.SourceBin | None = None
        self.live: live_pipeline.LiveParts | None = None
        self.batch: batch_pipeline.BatchParts | None = None
        self.collector: batch_pipeline.ResultCollector | None = None
        self.worker: batch_pipeline.BatchWorker | None = None
        self.recorder: disk.Recorder | None = None
        self.disk_worker: disk.DiskWorker | None = None
        self._node = None
        self._executor = None
        self._ros_thread: threading.Thread | None = None
        self._loop = None
        self._live_bus = None
        self._batch_bus = None
        self._shutdown_done = False

    def build(self) -> None:
        """Construct and wire everything (order above). Main thread, once."""
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        import rclpy

        # rclpy must not own SIGINT/SIGTERM: the GLib handlers installed in
        # run() quit the MainLoop, which drives the one orderly shutdown path.
        try:
            from rclpy.signals import SignalHandlerOptions

            rclpy.init(args=self.argv,
                       signal_handler_options=SignalHandlerOptions.NO)
        except ImportError:
            rclpy.init(args=self.argv)

        # Resolve --ros-args -p overrides on a short-lived bootstrap node:
        # the config must exist before any pipeline is built, but DsRosNode
        # takes the finished collaborators (built FROM the config) — so the
        # parameters are read first on a throwaway node of the same name.
        # This MUST all happen before Gst.init(None): Gst.init dlopens
        # libunwind, and creating the FIRST rmw context/node after that (with
        # libfastrtps already loaded via import rclpy) aborts the process in
        # Fast DDS's internally-caught SHM-setup exception (DSN-1). Nodes
        # created after Gst.init are fine once rclpy.init preceded it.
        from rclpy.node import Node

        bootstrap = Node("ds_pipeline")
        try:
            self.config = config_mod.declare_parameters(bootstrap)
        finally:
            bootstrap.destroy_node()
        cfg = self.config

        Gst.init(None)
        self._loop = GLib.MainLoop()

        pgie_config = infer_configs.write_batch_yolo_config(cfg.batch_engine_batch)
        sgie_config = infer_configs.sgie_config_path()

        self.registry = timestamps.TimestampRegistry()
        self.lifecycle = frames.Lifecycle()
        self.grab = frames.GrabState(cfg, self.lifecycle)

        self.source = source_mod.create_source(cfg)
        self.live = live_pipeline.build_live_pipeline(cfg, self.source)
        live_pipeline.install_ingest_stamp_probe(self.live.pace, self.registry)
        grab_probe = frames.make_grab_probe(self.grab, self.registry,
                                            timestamps.resolve)
        self.live.caps_grab.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, grab_probe)

        self.batch = batch_pipeline.build_batch_pipeline(
            cfg, self.source.width, self.source.height,
            str(pgie_config), str(sgie_config))
        self.collector = batch_pipeline.ResultCollector()
        batch_pipeline.install_collect_probes(self.batch, self.collector)
        # The publish callbacks close over self._node (assigned below, before
        # any run can execute — runs only start once services are up).
        self.worker = batch_pipeline.BatchWorker(
            cfg, self.grab, self.batch, self.collector,
            publish_detections=lambda result:
                self._node.publish_detections(result),
            publish_assessment=lambda result, object_id:
                self._node.publish_assessment(result, object_id))

        self.recorder = disk.Recorder(cfg, self.live, self.registry)
        self.disk_worker = disk.DiskWorker()

        import ros_io

        self._node = ros_io.DsRosNode(
            cfg, self.lifecycle, self.grab, self.recorder, self.worker,
            self.registry, self.disk_worker,
            loop_count_fn=lambda: self.source.n_loops)
        # Re-declare on the live node purely for introspection (ros2 param
        # list/get); values were already resolved on the bootstrap node.
        config_mod.declare_parameters(self._node)

        live_pipeline.connect_preview(
            self.live.sink_preview, self._node.publish_preview, self.registry)

        self._live_bus = self.live.pipeline.get_bus()
        self._batch_bus = self.batch.pipeline.get_bus()
        self._live_bus.add_watch(GLib.PRIORITY_DEFAULT, self.on_bus_message)
        self._batch_bus.add_watch(GLib.PRIORITY_DEFAULT, self.on_bus_message)

    def run(self) -> int:
        """Set both pipelines PLAYING, start threads, run the GLib MainLoop
        until SIGINT/fatal bus error; then shutdown(). Returns exit code.
        Blocks the main thread for the process lifetime."""
        from gi.repository import GLib, Gst

        from rclpy.executors import MultiThreadedExecutor

        for signum in (signal.SIGINT, signal.SIGTERM):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signum,
                                 self._on_signal, signum)

        # Batch first (idles until fed), then live (starts real dataflow).
        for name, pipeline in (("batch", self.batch.pipeline),
                               ("live", self.live.pipeline)):
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                print(f"FATAL: {name} pipeline refused to go PLAYING",
                      file=sys.stderr, flush=True)
                self.exit_code = 1
                self.shutdown()
                return self.exit_code

        self.disk_worker.start()
        self.worker.start()
        self.source.start()

        self._executor = MultiThreadedExecutor(num_threads=EXECUTOR_THREADS)
        self._executor.add_node(self._node)
        self._ros_thread = threading.Thread(
            target=self._executor.spin, name="ros", daemon=True)
        self._ros_thread.start()

        print("ds_node: pipelines PLAYING, services up", flush=True)
        try:
            self._loop.run()
        finally:
            self.shutdown()
        return self.exit_code

    def _on_signal(self, signum: int) -> bool:
        """GLib unix-signal handler (main-loop context): quit the loop; run()
        then performs the one orderly shutdown. Handler stays installed."""
        print(f"ds_node: received signal {signum}, shutting down", flush=True)
        self._loop.quit()
        return True

    def on_bus_message(self, bus, message) -> bool:
        """Bus watch for both pipelines (GLib main context, main thread).

        EOS from the live pipeline (only possible with source.loop=false,
        Sec 5): lifecycle.mark_ended(); grab.notify_ended();
        recorder.on_source_eos(); do NOT quit. ERROR: log and initiate
        shutdown. Always returns True — a raising (hence falsy-returning)
        watch callback would be removed by GLib permanently, silently losing
        ERROR/EOS handling for that pipeline (DSN-2).
        """
        try:
            self._handle_bus_message(bus, message)
        except Exception:
            traceback.print_exc()
        return True

    def _handle_bus_message(self, bus, message) -> None:
        from gi.repository import Gst

        origin = "live" if bus == self._live_bus else "batch"
        mtype = message.type
        if mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"ERROR: {origin} pipeline: {err.message}"
                  f"\n{debug or '(no debug info)'}",
                  file=sys.stderr, flush=True)
            self.exit_code = 1
            self._loop.quit()
        elif mtype == Gst.MessageType.WARNING:
            warn, _debug = message.parse_warning()
            print(f"WARNING: {origin} pipeline: {warn.message}",
                  file=sys.stderr, flush=True)
        elif mtype == Gst.MessageType.EOS:
            if origin == "live":
                # Sec 5 ended transition: keep publishing /ds/status, stay up.
                print("ds_node: live pipeline EOS -> state 'ended'"
                      " (process stays alive; /ds/status keeps publishing)",
                      flush=True)
                self.lifecycle.mark_ended()
                self.grab.notify_ended()
                self.recorder.on_source_eos()
            else:
                print("WARNING: unexpected EOS on the batch pipeline"
                      " (its appsrc never sends EOS)",
                      file=sys.stderr, flush=True)

    def shutdown(self) -> None:
        """Orderly teardown (main thread): stop workers (batch, disk), stop
        source (join feeder), executor.shutdown + join "ros" thread, both
        pipelines to NULL, rclpy.shutdown. Idempotent."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._finalize_recording()
        self._stop_batch_worker()
        if self.source is not None:
            self.source.stop()
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=SHUTDOWN_JOIN_TIMEOUT_S)
            except Exception:
                traceback.print_exc()
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            if self._ros_thread.is_alive():
                print("WARNING: ros executor thread did not exit"
                      f" within {SHUTDOWN_JOIN_TIMEOUT_S:.0f}s",
                      file=sys.stderr, flush=True)
        if self.disk_worker is not None:
            self.disk_worker.stop()
        self._teardown_gst()
        self._teardown_rclpy()

    def _stop_batch_worker(self) -> None:
        """Stop the batch worker with a bounded wait (DSN-3).

        worker.stop() joins without a timeout, and a mid-flight appsrc push
        (block=true) is only released by the batch pipeline leaving PLAYING —
        which used to happen after the join, so a wedged pipeline could hang
        shutdown forever. Run stop() on a helper thread; if it does not
        finish in time, NULL the batch pipeline (unblocks the push, pending
        collect waits end within their own 10 s timeout) and wait again.
        """
        if self.worker is None:
            return
        stopper = threading.Thread(target=self.worker.stop,
                                   name="batch-stop", daemon=True)
        stopper.start()
        stopper.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
        if not stopper.is_alive():
            return
        print("WARNING: batch worker did not stop within"
              f" {SHUTDOWN_JOIN_TIMEOUT_S:.0f}s; forcing the batch pipeline"
              " to NULL to unblock it", file=sys.stderr, flush=True)
        try:
            from gi.repository import Gst

            if self.batch is not None:
                self.batch.pipeline.set_state(Gst.State.NULL)
        except Exception:
            traceback.print_exc()
        stopper.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S + 1.0)
        if stopper.is_alive():
            print("WARNING: batch worker still running after pipeline NULL"
                  " (collect timeout pending); continuing shutdown",
                  file=sys.stderr, flush=True)

    def _finalize_recording(self) -> None:
        """Finalize an active recording via the normal Sec 7 stop path
        (momentary-block detach + drain) while the live pipeline is still
        PLAYING — the drain needs a running graph."""
        if self.recorder is None or self.recorder.state != "recording":
            return
        try:
            success, _message, stats = self.recorder.stop()
            if stats is not None:
                print(f"ds_node: finalized recording {stats.path}"
                      f" (frames_written={stats.frames_written},"
                      f" drained={stats.drained})", flush=True)
            elif not success:
                print("WARNING: could not finalize the active recording",
                      file=sys.stderr, flush=True)
        except Exception:
            traceback.print_exc()

    def _teardown_gst(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception:
            return
        for bus in (self._live_bus, self._batch_bus):
            if bus is not None:
                bus.remove_watch()
        for parts in (self.live, self.batch):
            if parts is not None:
                parts.pipeline.set_state(Gst.State.NULL)

    def _teardown_rclpy(self) -> None:
        try:
            import rclpy
        except Exception:
            return
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if rclpy.ok():
            rclpy.shutdown()


def prebuild(engine_batch: int | None = None) -> int:
    """run.sh --prebuild path (Sec 11 risk 10): write the batch nvinfer
    config and warm the engine cache by cycling the batch pipeline through
    PLAYING (nvinfer builds and caches the b{N} engine during element start,
    ~1-3 min on first run), then exit. Uses config defaults except
    ``--engine-batch N`` (DSN-4: deployments running the Sec 11 risk 1
    fallback batch.engine_batch=4 warm the b4 engine, not an unused b8);
    no ROS, no live pipeline, no services."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    if engine_batch is None:
        cfg = config_mod.PipelineConfig()
    else:
        cfg = config_mod.PipelineConfig(batch_engine_batch=engine_batch)
    pgie_config = infer_configs.write_batch_yolo_config(cfg.batch_engine_batch)
    sgie_config = infer_configs.sgie_config_path()
    print(f"prebuild: wrote {pgie_config}", flush=True)

    Gst.init(None)
    parts = batch_pipeline.build_batch_pipeline(
        cfg, PREBUILD_WIDTH, PREBUILD_HEIGHT,
        str(pgie_config), str(sgie_config))
    print("prebuild: starting the batch pipeline"
          " (first run builds the b%d engine, ~1-3 min)"
          % cfg.batch_engine_batch, flush=True)
    try:
        if parts.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            _print_bus_errors(parts.pipeline)
            return 1
        ret, _state, _pending = parts.pipeline.get_state(
            int(PREBUILD_STATE_TIMEOUT_S) * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            _print_bus_errors(parts.pipeline)
            return 1
    finally:
        parts.pipeline.set_state(Gst.State.NULL)
    print("prebuild: engine cache ready", flush=True)
    return 0


def _print_bus_errors(pipeline) -> None:
    from gi.repository import Gst

    bus = pipeline.get_bus()
    while True:
        message = bus.pop_filtered(Gst.MessageType.ERROR)
        if message is None:
            break
        err, debug = message.parse_error()
        print(f"ERROR: prebuild: {err.message}\n{debug or '(no debug info)'}",
              file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Build and run the App; installs the SIGINT handler that quits the
    MainLoop. run.sh execs this."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--prebuild" in args:
        args.remove("--prebuild")
        engine_batch: int | None = None
        if "--engine-batch" in args:
            i = args.index("--engine-batch")
            try:
                engine_batch = int(args[i + 1])
            except (IndexError, ValueError):
                print("usage: --prebuild [--engine-batch N]",
                      file=sys.stderr, flush=True)
                return 2
        try:
            return prebuild(engine_batch)
        except Exception:
            traceback.print_exc()
            return 1
    app = App(args)
    try:
        app.build()
    except KeyboardInterrupt:
        app.shutdown()
        return 130
    except Exception:
        traceback.print_exc()
        app.shutdown()
        return 1
    return app.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
