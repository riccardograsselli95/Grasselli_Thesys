
"""
##############################################################################
##############################################################################
########                                                       ###############
########                   RICCARDO GRASSELLI                   ###############
########                                                       ###############
##############################################################################
##############################################################################



"""

import matplotlib.pyplot as plt
import json

file_path = 'Results/results.json'
with open(file_path, 'r') as file:
     results_dict = json.loads(file.read())

#print(json.dumps(results_dict, indent=4, sort_keys=False))


LR = "0.0001"
metrics = results_dict.keys()
print('Avalaible metrics: ', metrics)
# ['mean_absolute_error', 'val_mean_absolute_error', 'loss', 'val_loss', 'DSSIM', 'SSIM', 'val_DSSIM', 'val_SSIM']

for metric in metrics:
    for data in list(results_dict[metric].keys()):
        y = []
        for key in results_dict[metric][data][LR].keys():
            y.append(round(float(results_dict[metric][data][LR][key]),2))
        plt.plot(list(results_dict[metric][data][LR].keys()), y)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    # https://stackoverflow.com/questions/50128668/how-to-adjust-tick-frequency-for-string-x-axis
    plt.xticks(list(results_dict[metric][data][LR].keys())[::5], rotation=0)
    #plt.legend(list(results_dict[metric].keys()))
    plt.legend(['Original subsampling', 'Radial subsampling with 8 spokes', 'Radial subsampling with 16 spokes', 'Radial subsampling with 32 spokes'])
    plt.show()












