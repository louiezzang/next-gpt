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

# Requirements (Optional)
Make sure to install PyTorch >= 2.0 for nanogpt module.

$ pip3 install numpy --pre torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu116


# Development Environment (Optional)
This is only for setting up your IDE development environment.
```
$ pipenv install --skip-lock -r requirements.txt 
```
To activate this project's virtualenv
```
$ pipenv shell
```

# References
- https://github.com/karpathy/nanoGPT
- https://github.com/airobotlab/KoChatGPT
- https://github.com/airobotlab/KoChatGPT/tree/main/colossalai_ChatGPT_230319
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat
- https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py
- https://github.com/databrickslabs/dolly
