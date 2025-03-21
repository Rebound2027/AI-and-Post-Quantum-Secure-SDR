import adi
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import time
from scipy.signal import butter, lfilter  # ✅ Import Low-Pass Filter

# ✅ Initialize ADALM-Pluto SDR
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(5e6)  # ✅ Increased Sample Rate for Better Filtering
sdr.rx_rf_bandwidth = int(1e6)
sdr.rx_lo = int(915e6)
sdr.rx_hardwaregain = 30  # Adjust RX Gain to filter noise
sdr.tx_hardwaregain = 20  # Initial TX power level
sdr.rx_buffer_size = 4096  # Optimize for continuous streaming

# ✅ Define RL Actions: Decrease (-5 dB), Maintain (0 dB), Increase (+5 dB)
actions = [-5, 0, +5]
state_size = 1  # SNR as input
action_size = len(actions)

# ✅ Deep Q-Network (DQN) Model
model = Sequential([
    Dense(24, activation='relu', input_shape=(state_size,)),  
    Dense(24, activation='relu'),
    Dense(action_size, activation='linear')  
])
model.compile(optimizer='adam', loss='mse')

# ✅ Initialize Q-Table (for faster lookup)
Q_table = np.zeros((50, len(actions)))

# ✅ Reinforcement Learning Parameters
alpha, gamma, epsilon = 0.1, 0.9, 0.1

# ✅ Global SNR Variable (Shared Across Threads)
snr = 20  # Default value

# ✅ Function to Apply Low-Pass Filter
def lowpass_filter(signal, cutoff=1e6, fs=5e6, order=5):
    """ Apply a Butterworth Low-Pass Filter """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

# ✅ Function to Monitor SNR in Real-Time (With Low-Pass Filter)
def monitor_snr():
    """ Continuously measure SNR from ADALM-Pluto """
    global snr
    while True:
        samples = sdr.rx()
        filtered_signal = lowpass_filter(samples)  # ✅ Apply Low-Pass Filter
        signal_power = np.mean(np.abs(filtered_signal) ** 2)
        noise_power = np.var(filtered_signal)
        snr = max(0, min(49, 10 * np.log10(signal_power / noise_power)))  # Clip SNR
        time.sleep(0.5)  # ✅ Small delay to avoid overloading CPU

# ✅ Start Parallel Thread for Real-Time SNR Monitoring
snr_thread = threading.Thread(target=monitor_snr, daemon=True)
snr_thread.start()

# ✅ Function to Choose the Best Power Adjustment Action
def choose_action(snr):
    """ Select best power adjustment action using RL """
    snr = max(0, min(49, int(snr)))  # Ensure SNR is within valid range
    return actions[np.argmax(Q_table[snr])]  # Use learned Q-values

# ✅ Function to Update Q-Table After Each Action
def update_q_table(state, action, reward, next_state):
    """ Update Q-values using the Bellman equation """
    best_next_action = np.argmax(Q_table[next_state])
    Q_table[state, actions.index(action)] += alpha * (reward + gamma * Q_table[next_state, best_next_action] - Q_table[state, actions.index(action)])

# ✅ Train the RL Model Using Simulated SNR Conditions
def train_rl_model():
    """ Train RL model for adaptive power control """
    for episode in range(1000):  # Training episodes
        snr_val = random.randint(0, 49)  # Ensure initial SNR is within bounds
        for _ in range(10):  # Limit power adjustments per episode
            action = choose_action(snr_val)  # Select action
            new_snr = max(0, min(49, snr_val + action))  # Ensure new SNR is valid
            reward = 10 if new_snr > 20 else -10  # Reward for good SNR
            update_q_table(snr_val, action, reward, new_snr)  # Update Q-table
            snr_val = new_snr  # Move to next state

    print("✅ RL Training Complete!")

# ✅ Train the RL Model Before Real-Time Testing
train_rl_model()

# ✅ Set Time Limit (Run for 3 Minutes)
start_time = time.time()
run_duration = 180  # ✅ Run for 3 minutes (180 seconds)

# ✅ Real-Time Adaptive Power Control Loop (With Low-Pass Filter)
while time.time() - start_time < run_duration:  # ✅ Stop after 3 minutes
    optimal_power_adjustment = choose_action(int(snr))  # Select best action
    sdr.tx_hardwaregain = max(0, min(50, sdr.tx_hardwaregain + optimal_power_adjustment))  # Adjust power
    print(f"📡 Live SNR: {snr:.2f} dB | Adjusting Power by {optimal_power_adjustment} dB")
    time.sleep(2)  # ✅ Slowing down updates for better readability

print("✅ Power control loop completed. Exiting program.") 