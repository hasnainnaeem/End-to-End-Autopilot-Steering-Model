# End-to-end Learning for Self-driving Cars
A TensorFlow implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes. Please check project report for more details.
# How to Use
- Training: `python train.py`
- Testing on Live Video Stream: `python run.py`
- Testing on Dataset: `python run_dataset.py`
- Testing on Vide: `python run_video.py`
- Visualize model and plotting loss: run `tensorboard --logdir=./logs`, then open http://localhost:6006/ in browser.