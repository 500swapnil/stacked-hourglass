# TFKeras Implementation of Stacked Hourglass Network
A Tensorflow+Keras implementation of [Stacked Hourglass Network for Keypoint Detection](https://arxiv.org/abs/1603.06937) with several residual layers and can be trained to detect keypoints of chairs and cars.


## Dependencies
1. Python 3.6+
2. Tensorflow 2.2.0+ (For GPU, use Tensorflow-GPU 2.2.0+)
3. Pandas

To install these dependencies for CPU, run
```bash
pip install -r requirements.txt
```
(For GPU support, change line `tensorflow>=2.2.0` to `tensorflow-gpu>=2.2.0` in requirements.txt)


## Train Model
To train from scratch, run
```bash
python train.py
```
_Note_: The train, data, evaluation parameters are present in `params.py`

## Data Format Expected
The dataloader expects a csv file indicating (assuming `N` keypoints)
```
<image_name>, <keypointX_1>, <keypointY_1>, <keypointX_2>, <keypointY_2>,.... <keypointX_N>, <keypointY_N>
```
The image directory (`DATA_DIR`) and annotation csv (`anno_file`) need to be specified in `params.py`.

---
## References
1. https://github.com/bearpaw/pytorch-pose

_(Visualization is under progress)_