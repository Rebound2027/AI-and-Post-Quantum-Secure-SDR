import adi
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
from scipy.signal import butter, lfilter, resample  # ✅ Import resample

# ✅ Initialize ADALM-Pluto SDR
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(5e6)  # ✅ High sample rate for better interference learning
sdr.rx_rf_bandwidth = int(1e6)
sdr.rx_lo = int(915e6)  # ✅ Tune to correct frequency
sdr.tx_hardwaregain = 50  # ✅ Max TX Power
sdr.rx_hardwaregain = 30  # ✅ Adjust RX Gain
sdr.rx_buffer_size = 4096  # ✅ Ensure RX buffer size matches expected signal length

# ✅ Function to Apply Low-Pass Filter
def lowpass_filter(signal, cutoff=1e6, fs=5e6, order=5):
    """ Apply a Butterworth Low-Pass Filter """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

# ✅ Capture TX & RX Signals for Training
def capture_signals(num_samples=5000):
    """ Captures TX and RX signals for neural network training """
    tx_data = []
    rx_data = []

    for _ in range(num_samples):
        tx_signal = np.random.randn(100)  # Simulated TX signal
        tx_signal = resample(tx_signal, 4096)  # ✅ Resample TX to match RX size
        sdr.tx(tx_signal)  # Transmit the signal
        rx_signal = sdr.rx()  # Capture the received signal
        filtered_rx = lowpass_filter(rx_signal)  # ✅ Apply Low-Pass Filter
        tx_data.append(tx_signal)
        rx_data.append(filtered_rx)

    tx_data = np.array(tx_data)
    rx_data = np.array(rx_data)
    return tx_data, rx_data

# ✅ Train Neural Network for Self-Interference Cancellation
def train_nnsic_model():
    """ Trains a neural network to cancel self-interference """
    tx_data, rx_data = capture_signals()

    # ✅ Define NN Model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(4096,)),  # ✅ Match input to resampled TX size
        Dropout(0.2),  
        Dense(256, activation='relu'),
        Dense(4096, activation='linear')  # ✅ Output must match RX signal length
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    
    # ✅ Train the model
    model.fit(tx_data, rx_data - tx_data, epochs=10, batch_size=32)  # ✅ Now TX & RX are the same size
    model.save("nnsic_model.h5")  # ✅ Save trained model

    print("✅ NNSIC Model Trained & Saved!")

# ✅ Load and Apply the Trained Model in Real-Time
def apply_nnsic():
    """ Uses the trained model to remove interference in real-time """
    model = tf.keras.models.load_model("nnsic_model.h5")

    while True:
        tx_signal = np.random.randn(100)  # Simulated TX signal
        tx_signal = resample(tx_signal, 4096)  # ✅ Resample TX to 4096 samples
        sdr.tx(tx_signal)  # Transmit while listening
        rx_signal = sdr.rx()  # Capture received signal
        filtered_rx = lowpass_filter(rx_signal)  # ✅ Apply Low-Pass Filter
        predicted_interference = model.predict(tx_signal.reshape(1, -1))  # ✅ Predict interference
        corrected_signal = filtered_rx - predicted_interference  # ✅ Remove interference
        print(f"📡 Corrected SNR: {10 * np.log10(np.mean(np.abs(corrected_signal) ** 2)):.2f} dB")
        time.sleep(0.5)  # ✅ Prevent excessive CPU usage

# ✅ Train the NNSIC Model
train_nnsic_model()

# ✅ Start Real-Time NNSIC Processing
apply_nnsic()