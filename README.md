


## Data preparation
1. Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)
2. Download image_attribute_segment_outputs [BaiduYun(code: vbss)]() which are the masks of image attributes segmented with the Grounded SAM, and save it in (e.g. ~/datasets/outputs/).
3. For more attribute categories, use the provided [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to generate the attribute segmentations. We will provide the modifed code.

'''
cd ".../Grounded-Segment-Anything-main"; python grounded_sam_a2ha.py
'''

4. Download text_attribute_annotation [BaiduYun(code: vbss)]() which are the parsed description of human attributes from the original annotations in CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets.

Your `datasets` directory should look like this:
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
        |-- attribute.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- attribute.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- attribute.json
```


## Training
```
python train.py --img_aug --batch_size 64 --MLM --loss_names 'sdm+mlm+id+itc+att'  --dataset_name ' '  --root_dir 'your_root_dir' --num_epoch 60



## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

