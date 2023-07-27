import library.analysis.fake_signal as fake_signal
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main():
    config = dict(
        t_start = 0,
        t_end = 4,
        f_sampling = 250,
        f_sine = 1,
        phase = 0,
        amplitude = 1
    )

    x, t = fake_signal.generate_sinusoide(config)

    config['phase'] = 10
    x_shift, t = fake_signal.generate_sinusoide(config)
    config['phase'] = 0

    config['amplitude'] = 4
    x_amplitude, t = fake_signal.generate_sinusoide(config)
    config['amplitude'] = 1

    plt.figure(figsize = (15, 10))
    plt.plot(t, x, label = 'Original')
    plt.plot(t, x_shift, label = 'Shifted')
    plt.plot(t, x, label = 'Amplitude')
    plt.legend()
    plt.xlim([config['t_start'], config['t_end']])
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
