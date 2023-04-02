from dataclasses import asdict, dataclass, field, fields

from transformers import (
    Trainer, TrainingArguments,
    TrainerState, TrainerControl, 
    TrainerCallback, ProgressCallback
) 


class SupervisedTrainer(Trainer):
    """
    Trainer for supervised funtuning.
    """
    
    def safe_save_model(self, output_dir: str, logger=None):
        """ 
        Collects the state dict and dump to disk.
        """
        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in list(state_dict.items())}
            del state_dict
            # Save into local.
            self._save(output_dir, state_dict=cpu_state_dict)

            if logger and hasattr(logger, "release_model"):
                args_as_dict = asdict(self.args)
                state_dict = {
                    "model_state_dict": cpu_state_dict,
                    "model_args": args_as_dict
                }
                logger.release_model(state_dict=state_dict)


