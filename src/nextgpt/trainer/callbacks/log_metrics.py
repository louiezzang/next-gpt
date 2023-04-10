from .base import Callback


class LogMetrics(Callback):
    """ 
    The callback for logging metrics.
    """

    def __init__(self, logger, verbose: bool = False) -> None:
        super().__init__()
        self.logger = logger
        self.verbose = verbose
        if not hasattr(logger, "log_metrics"):
            raise ValueError(f"{logger} should have 'log_metrics(metrics, step)' function!")

    def on_log_metrics(self, metrics: dict, **kwargs) -> None:
        step = kwargs.get("step", -1)
        if step < 0:
            step = kwargs.get("epoch", -1)
        if metrics is None or step < 0:
            if self.verbose:
                if step < 0:
                    print(f"[WARN] Cannot log metrics: undefined 'step' or 'epoch' in kwargs")
                if metrics is None:
                    print(f"[WARN] Cannot log metrics: metrics={metrics}, {kwargs}")
            return
        
        self.logger.log_metrics(metrics, step)
        if self.verbose:
            print(f"*** on_log_metrics: metrics={metrics}, {kwargs}")
