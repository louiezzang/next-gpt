# nextGPT
> 📢 Open source implementation for ChatGPT replica to build the end-to-end pipeline from SFT to RLHF.

- 🔥 Step 1) SFT: Surpervised Fine-tuning
- 🔥 Step 2) RM: Reward Model
- 🔥 Step 3) PPO: Proximal Policy Optimization

# Installation
```
$ pip install nextgpt
```
or
```
$ git clone https://github.com/louiezzang/next-gpt.git
$ cd next-gpt/
$ pip install .
$ cd ../
```

# Examples
See [chatGPT example](https://github.com/louiezzang/next-gpt/blob/main/examples/chatgpt_example.ipynb)

# RLHF
What is [RLHF](https://gist.github.com/JoaoLages/c6f2dfd13d2484aa8bb0b2d567fbf093)?

Implementation of RLHF (Reinforcement Learning with Human Feedback) was powered by Colossal-AI. More details can be found in the [blog](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt).

The RLHF was forked and modified from these git repos.
- https://github.com/airobotlab/KoChatGPT/tree/main/colossalai_ChatGPT_230319
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat

# References
- https://github.com/airobotlab/KoChatGPT
- https://github.com/airobotlab/KoChatGPT/tree/main/colossalai_ChatGPT_230319
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat
- https://github.com/huggingface/peft
- https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py
- https://github.com/databrickslabs/dolly
