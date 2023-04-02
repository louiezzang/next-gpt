from transformers import (
    TrainingArguments,
    TrainerState, TrainerControl, 
    TrainerCallback, ProgressCallback
)   

class LoggingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the custom logging by such as model logger.
    """

    def __init__(self, logger):
        self.logger = logger
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        #print(f"*** on_epoch_end: {state}")
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        #print(f"*** on_step_end: {state}")
        pass
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        print(f"*** on_evaluate: {state}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        print(f"*** on_save: {state}")
        if self.logger:
            step = state.global_step
            log_history = state.log_history

            log = None
            for x in log_history:
                if x["step"] == step:
                    log = x
                    break
            if hasattr(self.logger, "log_metrics") and log:
                print(f"*** log metrics by model logger: {log}")
                metrics = {
                    "train_loss": log["loss"]
                }
                self.logger.log_metrics(metrics, step)

