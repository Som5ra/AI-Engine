
### IMPORTANT NOTES:
- EXPORT_CFG determine the input batch: "detection_onnxruntime_static.py" and "detection_onnxruntime_dynamic.py"

``` # without nms
EXPORT_CFG="/media/sombrali/HDD1/mmlib/mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py" && \
MODEL_CFG="mmdetection/work_dirs/mbnv3_20241203/mbnv3_20241203.py" && \
CHECKPOINT="mmdetection/work_dirs/mbnv3_20241203/epoch_120.pth" && \
WORKING_DIR=$(echo "$CHECKPOINT" | cut -f 1 -d '.') && \
python mmdeploy/tools/deploy.py \
    ${EXPORT_CFG} \
    ${MODEL_CFG} \
    ${CHECKPOINT} \
    mmdetection/demo/demo.jpg \
    --work-dir ${WORKING_DIR} \
    --device cpu \
&& \
ONNX_MODEL="$WORKING_DIR/end2end.onnx" && \
ONNX_NONMS_MODEL="$WORKING_DIR/end2end_nonms.onnx" && \
python onnx_remove_nms.py \
    --input ${ONNX_MODEL}
    --output ${ONNX_NONMS_MODEL} \
&& \ 
python symbolic_shape_infer.py \
    --input ${ONNX_NONMS_MODEL} \
    --output ${ONNX_NONMS_MODEL} \
    --auto_merge \
&& \
python convert_to_fp16_int8.py \
    --input ${ONNX_NONMS_MODEL}
```

### Option: (Add a preprocessor for unity sentis inference engine)
```
python add_preprocessor_sentis.py
```


``` # with nms
EXPORT_CFG="/media/sombrali/HDD1/mmlib/mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py" && \
MODEL_CFG="/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/retinanet_mbnv2-1x_coco/retinanet_mbnv2-1x_coco.py" && \
CHECKPOINT="/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/retinanet_mbnv2-1x_coco/epoch_12.pth" && \
WORKING_DIR=$(echo "$CHECKPOINT" | cut -f 1 -d '.') && \
python mmdeploy/tools/deploy.py \
    ${EXPORT_CFG} \
    ${MODEL_CFG} \
    ${CHECKPOINT} \
    mmdetection/demo/demo.jpg \
    --work-dir ${WORKING_DIR} \
    --device cpu \
&& \
ONNX_MODEL="$WORKING_DIR/end2end.onnx" && \
python symbolic_shape_infer.py \
    --input ${ONNX_MODEL} \
    --output ${ONNX_MODEL} \
    --auto_merge \
&& \
python convert_to_fp16_int8.py \
    --input ${ONNX_MODEL}
```



### Use Opset 12 to support some operator in ort1.19.2
```YOLO EXPORT
MODEL_CFG="/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/rtmdet_tiny_disney_headband_v7_largesyn_hsv_20241101/rtmdet_tiny_disney_headband_v7_largesyn_hsv_20241101.py" && \
CHECKPOINT="/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/rtmdet_tiny_disney_headband_v7_largesyn_hsv_20241101/epoch_150.pth" && \
WORKING_DIR=$(echo "$CHECKPOINT" | cut -f 1 -d '.') && \

python mmyolo/projects/easydeploy/tools/export_onnx.py  \
${MODEL_CFG}       \
${CHECKPOINT}        \
--work-dir ${WORKING_DIR}     \
--img-size 320 320     \
--batch 1     \
--device cpu     \
--simplify       \
--opset 12      \
--pre-topk 1000         \
--keep-topk 100         \
--iou-threshold 0.65    \
--score-threshold 0.2
&& \
ONNX_MODEL="$WORKING_DIR/epoch_150.onnx" && \
ONNX_NONMS_MODEL="$WORKING_DIR/epoch_150_nonms.onnx" && \
python onnx_remove_nms.py \
    --input ${ONNX_MODEL} \
    --output ${ONNX_NONMS_MODEL} \
    --opset 12 \
&& \ 
python symbolic_shape_infer.py \
    --input ${ONNX_NONMS_MODEL} \
    --output ${ONNX_NONMS_MODEL} \
    --auto_merge \
&& \
python convert_to_fp16_int8.py \
    --input ${ONNX_NONMS_MODEL} \
    --opset 12
```