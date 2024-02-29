from model.fcos import FCOSDetector
from model.tfcos import TFCOSDetector
from ptflops import get_model_complexity_info

if __name__ == "__main__":
    model = FCOSDetector(mode="train").cuda()
    # model = TFCOSDetector(phase="train").cuda()

    flops, params = get_model_complexity_info(model, (3, 320, 320), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)


