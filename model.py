import cv2 as cv
import sys

#loads the using weights and correspding configuration rule
def load_model(model_weights="",model_config=""):
	
	if model_weights == "" and model_config == "":
		print("Model config and weights not found!");
		sys.exit(0);

	net = cv.dnn.readNet(model_weights,model_config);
	return net;

def get_output_layers(net):
    layer_names = net.getLayerNames();
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()];
    return output_layers;
