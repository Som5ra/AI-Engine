import onnx_graphsurgeon as gs
import onnxruntime as ort
import onnx
import numpy as np

# Register the custom operator (not needed for simple operators like Add and Divide)
def register_custom_op():
    so = ort.SessionOptions()
    return so

# Register the custom operator
so = register_custom_op()

# in_path = '/media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Weights/rtmdet_t_v7_20241028.onnx'
# out_path = '/media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Weights/rtmdet_t_v7_20241028_preprocessor.onnx'
# first_node_name = "images"

in_path = '/media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Weights/end2end_nonms_fp16.onnx'
out_path = '/media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Weights/end2end_nonms_fp16_preprocessor.onnx'
first_node_name = "input"

# Load the ONNX model
model = gs.import_onnx(onnx.load(in_path))

# Find the node with the input called "images" or "input"
input_node = [node for node in model.nodes if node.inputs and (node.inputs[0].name == first_node_name)][0]

# Create constants for mean, std, and refactor
refactor = np.array([255.0], dtype=np.float32).reshape((1, 1, 1, 1))
mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape((1, 3, 1, 1))
std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape((1, 3, 1, 1))

refactor_node = gs.Constant(name="refactor", values=refactor)
mean_node = gs.Constant(name="mean", values=mean)
std_node = gs.Constant(name="std", values=std)

# Create the Mul node for refactoring
mul_node = gs.Node(
    op="Mul",
    inputs=[input_node.inputs[0], refactor_node],
    outputs=[gs.Variable(name="mul_output", dtype=np.float32, shape=(1, 3, 320, 320))]
)

# Create the Sub node for mean subtraction
sub_node = gs.Node(
    op="Sub",
    inputs=[mul_node.outputs[0], mean_node],
    outputs=[gs.Variable(name="sub_output", dtype=np.float32, shape=(1, 3, 320, 320))]
)

# Create the Div node for standard deviation division
div_node = gs.Node(
    op="Div",
    inputs=[sub_node.outputs[0], std_node],
    outputs=[gs.Variable(name="preprocessed_image", dtype=np.float32, shape=(1, 3, 320, 320))]
)

# Find the first "cast" node
cast_node = [node for node in model.nodes if node.op == "Cast"][0]

# Update the input of the first "cast" node to be the output of the div_node
cast_node.inputs[0] = div_node.outputs[0]

# Insert the new nodes before the first "cast" node
model.nodes.insert(model.nodes.index(cast_node), mul_node)
model.nodes.insert(model.nodes.index(cast_node), sub_node)
model.nodes.insert(model.nodes.index(cast_node), div_node)

# Save the modified model
onnx.save(gs.export_onnx(model), out_path)

### test:
# Run the model with ONNX Runtime
session = ort.InferenceSession(out_path, so)

# Example input data
input_data = np.random.rand(1, 3, 320, 320).astype(np.float32)

# Run inference
outputs = session.run(None, {first_node_name: input_data})
print(outputs)