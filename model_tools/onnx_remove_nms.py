# python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

import onnxruntime as ort
import onnx
import numpy as np
import onnx_graphsurgeon as gs
from onnx_graphsurgeon import Tensor, Constant
from onnx import version_converter
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# def _remove_children(graph, node, level = 1):
#     logging.info("Attempting to remove node ", node.name)

#     if node not in graph.nodes:
#         return
#     graph.nodes.remove(node)

#     for child in list(node.outputs):
#         t = dict()
#         for n in list(graph.nodes):
#             if isinstance(n, Constant):
#                 continue
#             if child in list(n.inputs):
#                 t[n.name] = n
                
        
#         for _, n in t.items():
#             _remove_children(graph, n, level = level + 1)
    
#     node.outputs = []


# def _get_possible_mask_sink_nodes(graph):
#     sinks = []
#     for node in list(graph.nodes):
#         if len(node.inputs) > 0 and len(node.outputs) > 0:
#             is_connected = False
#             for o in list(node.outputs):
                
                
#                 for n in list(graph.nodes):
#                     if isinstance(n, Constant):
#                         continue
#                     if o in list(n.inputs):
#                         is_connected = True
#                         break
                
#                 if is_connected:
#                     break

#             if not is_connected:
#                 sinks.append(node)
#                 logging.info("Found sink: ", node.name)
    
#     return sinks


def remove_nms(src_onnx_file, out_file=None, ops_ver = 11):
    logging.info(f"[convert_remove_nms] Loading {src_onnx_file}")    
    graph = gs.import_onnx(onnx.load(src_onnx_file))
    has_mask = any(map(lambda x: x.name == 'masks', graph.outputs))
    if has_mask:
        raise Exception("Mask model is not supported")
    
    found = False
   
    graph.outputs.clear()
    for node in list(graph.nodes):
        # if (node.name in ['/TRTBatchedNMS', '/NonMaxSuppression']):

        # targeting ort ops NonMaxSuppression
        if (node.name in ['/NonMaxSuppression']):
            logging.info("Replacing: ")
            bbox_tensor = node.inputs[0]
            cls_tensor = node.inputs[1]
            
            logging.info(node)

            bbox_tensor.name = 'dets'
            cls_tensor.name = 'labels'

            graph.outputs.append(bbox_tensor.to_variable(dtype=np.float32, shape=None))
            graph.outputs.append(cls_tensor.to_variable(dtype=np.float32, shape=None))

            found = True
            
            
            # delete all nodes after this
            # if has_mask:
            #     _remove_children(graph=graph, node=node)
            #     sinks = _get_possible_mask_sink_nodes(graph)

            #     has_mask_outs = False
            #     for sink in sinks:
            #         for sink_out in sink.outputs:
            #             if sink_out not in graph.outputs:
            #                 # has shape like [B, Dets, Mask], and last dim != 2 (may be used in resizing)
            #                 if sink_out.shape is not None and len(sink_out.shape) == 3 and sink_out.shape[2] != 2:
            #                     logging.info("Adding possible mask output ", sink_out)
            #                     graph.outputs.append(sink_out)
            #                     has_mask_outs = True
                
            #     if not has_mask_outs:
            #         raise Exception("Unable to determine mask node")

            break

    if not found:
        raise Exception("No NMS layer is found.")

    logging.info("[convert_remove_nms] Cleaning up...")
    graph.cleanup()
    graph = gs.export_onnx(graph)

    if out_file is None:
        out_file = src_onnx_file.replace(".onnx", "_nonms.onnx")
    
    logging.info(f"[convert_remove_nms] Writing {out_file}...")
    onnx.save( version_converter.convert_version(graph, ops_ver), out_file)

    logging.info("[convert_remove_nms] Done")


if __name__ == '__main__':

    # remove_nms(
    #     '/ssd1/xavier_onnx/mmdeploy_cpu_workdir/rtmdet/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b/end2end_nonms.onnx'
    # )

    # remove_nms(
    #     '/ssd1/xavier_onnx/mmdeploy_cpu_workdir/pretrain_maskrcnn_test/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5/end2end_nonms.onnx'
    # )

    # exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The input model file")
    parser.add_argument("--output", help="The output model file")
    parser.add_argument("--opset", type=int, default=12, help="The output model file")
    args = parser.parse_args()
    
    remove_nms(args.input, out_file=args.output, ops_ver=args.opset)