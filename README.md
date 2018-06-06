# text-detection-ctpn
text detection mainly based on ctpn (connectionist text proposal network). It is implemented in tensorflow. I use it to detect id card includes Chinese and European,I also use it to dectect printed transfer orders, but it should be noticing that this model can be used in almost every horizontal scene text detection task. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog][1]

[1]:http://pancakeawesome.ink/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%E4%B9%8BCTPN(Detecting%20Text%20in%20Natural%20Image%20with%20Connectionist%20Text%20Proposal%20Network).html
***
# setup
- requirements: tensorflow1.3, opencv-python
- if you have a gpu device, build the library by
```shell
cd lib/utils
chmod +x make.sh
./make.sh
```
# demo
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ./ctpn/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. 
- Second, prepare the training data as referred in paper, or you can download the data I prepared from previous link.You can use my label tool to label datas.P.S:This tool can label different types object at the same time.I add 2k id cards and printed transfer orders in the train data.Especialy,for printed transfer records,I use my [Java programmer](https://github.com/PancakeAwesome/datagenerator_for_printed_transfers) to generate standard form orders at high speed.
```shell
cd prepare_training_data
python bbox_label_tool.py
```
- Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run
```shell
python split_label.py
```
- it will generate the prepared data in current folder, and then run
```shell
python ToVoc.py
```
- to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/ and then run
```shell
cd ../data
ln -s TEXTVOC VOCdevkit2007
```
## train 
Simplely run
```shell
python ./ctpn/train_net.py
```
- you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
# here are some results
<img src="/data/results/001.jpg" width=320 height=240 /><img src="/data/results/002.jpg" width=320 height=240 />
<img src="/data/results/003.jpg" width=320 height=240 /><img src="/data/results/004.jpg" width=320 height=240 />
<img src="/data/results/009.jpg" width=320 height=480 /><img src="/data/results/010.png" width=320 height=320 />

