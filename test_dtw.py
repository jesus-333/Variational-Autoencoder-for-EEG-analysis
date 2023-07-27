import library.analysis.fake_signal as fake_signal
import library.analysis.dtw_analysis as dtw_analysis

import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% 

def main():
    config = dict(
        t_start = 0,
        t_end = 4,
        f_sampling = 250,
        f_sine = 13,
        phase = 0,
        amplitude = 1
    )

    x, t = fake_signal.generate_sinusoide(config)

    config['phase'] = 0.25
    x_shift, t = fake_signal.generate_sinusoide(config)
    config['phase'] = 0

    config['amplitude'] = 7
    x_amplitude, t = fake_signal.generate_sinusoide(config)
    config['amplitude'] = 1
    
    radius = 20

    plt.figure(figsize = (15, 10))
    plt.plot(t, x, label = 'Original')
    plt.plot(t, x_shift, label = 'Shifted')
    plt.plot(t, x_amplitude, label = 'Amplitude')
    plt.legend()
    plt.xlim([config['t_start'], config['t_end']])
    plt.xlim([config['t_start'], 1])
    plt.grid(True)
    plt.show()
    
    dtw_fast_orig_orig_abs = dtw_analysis.compute_dtw_fastdtw(x, x, radius = radius) / len(x)
    dtw_python_orig_orig = dtw_analysis.compute_dtw_dtwpython(x, x) / len(x)
    dtw_soft_orig_orig = dtw_analysis.compute_dtw_softDTWCuda(x, x) / len(x)

    dtw_fast_orig_shift_abs = dtw_analysis.compute_dtw_fastdtw(x, x_shift, radius = radius) / len(x)
    dtw_python_orig_shift = dtw_analysis.compute_dtw_dtwpython(x, x_shift) / len(x)
    dtw_soft_orig_shift = dtw_analysis.compute_dtw_softDTWCuda(x, x_shift) / len(x)

    dtw_fast_orig_amplitude_abs = dtw_analysis.compute_dtw_fastdtw(x, x_amplitude, radius = radius) / len(x)
    dtw_python_orig_amplitude = dtw_analysis.compute_dtw_dtwpython(x, x_amplitude)/ len(x)
    dtw_soft_orig_amplitude = dtw_analysis.compute_dtw_softDTWCuda(x, x_amplitude)/ len(x)
    
    print("DTW Original vs Original")
    print("\t FastDTW  : {:.2f} (Absolute value)".format(dtw_fast_orig_orig_abs))
    print("\t PythonDTW: {:.2f}".format(dtw_python_orig_orig))
    print("\t SoftDTW  : {:.2f}".format(dtw_soft_orig_orig))
    
    print("DTW Original vs shift")
    print("\t FastDTW  : {:.2f} (Absolute value)".format(dtw_fast_orig_shift_abs))
    print("\t PythonDTW: {:.2f} ".format(dtw_python_orig_shift))
    print("\t SoftDTW  : {:.2f}".format(dtw_soft_orig_shift))

    print("DTW Original vs Amplitude")
    print("\t FastDTW  : {:.2f} (Absolute value)".format(dtw_fast_orig_amplitude_abs))
    print("\t PythonDTW: {:.2f}".format(dtw_python_orig_amplitude))
    print("\t SoftDTW  : {:.2f}".format(dtw_soft_orig_amplitude))

if __name__ == '__main__':
    main()
