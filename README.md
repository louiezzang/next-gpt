# next-gpt
---

# Installation
```
$ git clone https://github.com/louiezzang/next-gpt.git
$ cd next-gpt/
$ pip install .
$ cd ../
```
or you can build the wheel file.
```
$ git clone https://github.com/louiezzang/next-gpt.git
$ ./next-gpt/build.sh build_wheel
$ ls ./next-gpt/dist/
```

# Development Environment (Optional)
This is only for setting up your IDE development environment.
```
$ pipenv install --skip-lock -r requirements.txt 
```
To activate this project's virtualenv
```
$ pipenv shell
```

# Packages
## `nextgpt.nano`
This package is a minimal implemention of GPT-2 model to understand the basic concept of GPT-2 model architecture.
This module was forked and modified from [nanoGPT](https://github.com/karpathy/nanoGPT) git repo of Andrej karpathy.

## `nextgpt.rlhf`
What is [RLHF](https://gist.github.com/JoaoLages/c6f2dfd13d2484aa8bb0b2d567fbf093)?

Implementation of RLHF (Reinforcement Learning with Human Feedback) was powered by Colossal-AI. More details can be found in the [blog](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt).

The RLHF was forked and modified from these git repos.
- https://github.com/airobotlab/KoChatGPT/tree/main/colossalai_ChatGPT_230319
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat

# References
- https://github.com/karpathy/nanoGPT
- https://github.com/airobotlab/KoChatGPT
- https://github.com/airobotlab/KoChatGPT/tree/main/colossalai_ChatGPT_230319
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat
- https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py
- https://github.com/databrickslabs/dolly
