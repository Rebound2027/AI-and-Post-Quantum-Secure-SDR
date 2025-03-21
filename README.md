# Optimized-SDR-using-AI-and-ML-techniques

AI-Optimized Full-Duplex SDR Communication System

An intelligent IoT communication system built on *ADALM-Pluto SDR* using AI/ML techniques to enhance reliability, optimize power usage, and ensure secure data transmission in challenging wireless environments.


Project Overview

This project focuses on optimizing full-duplex SDR communication using AI-driven approaches for dynamic spectrum management, jamming resistance, and post-quantum cryptography. Built around *ADALM-Pluto SDR*, it integrates machine learning and cryptographic solutions to create a robust, secure, and efficient IoT communication network.

Features

- *NNSIC (Neural Network-based Self-Interference Cancellation)*
- *Deep Q-Learning for Adaptive Power Control*
- *CNN-based Spectrum Sensing* for predicting free frequency bands
- *Q-Learning* for Anti-Jamming and Frequency Hopping
- *ChaCha20 and Falcon-512* implementation for Post-Quantum Cryptography (PQC)
- Real-time optimization for full-duplex communication
- Secure and dynamic frequency allocation and interference mitigation

Technologies & Components Used

| Component / Technology       | Purpose                                                  |
|------------------------------|----------------------------------------------------------|
| *ADALM-Pluto SDR*          | Full-duplex software-defined radio hardware              |
| *NNSIC*                    | AI-driven self-interference cancellation                 |
| *Deep Q-Learning*          | Adaptive power control based on dynamic RF conditions    |
| *CNN (Convolutional NN)*   | Predicts free frequencies from spectrum data             |
| *Q-Learning*               | Adaptive frequency hopping and anti-jamming strategy     |
| *ChaCha20*                 | Symmetric stream cipher for data encryption              |
| *Falcon-512*               | Post-quantum signature scheme for authentication         |


Current Progress

Completed:
- Implementation of *Post-Quantum Cryptography* (ChaCha20 & Falcon-512)
- Integration of *NNSIC* model for self-interference cancellation
- *Adaptive Power Control* using Deep Q-Learning

In Progress / Next Steps:

- Training and deployment of *CNN* for free frequency prediction
- Integration of *Q-Learning* agent for anti-jamming and frequency hopping
- Real-time testing and hardware integration on ADALM-Pluto
- Latency and performance benchmarking

System Architecture

            +---------------------------+
            |     IoT Edge Device       |
            |  (ADALM-Pluto + AI Core)  |
            +------------+--------------+
            
                         |
         +---------------+----------------+
         |     Spectrum Sensing (CNN)     |
         | Self-Interference Cancellation |
         | Power Control (Deep Q-Learn)   |
         |  Frequency Hopping (Q-Learn)   |
         |     Secure Layer (PQC)         |
         +---------------+----------------+
         
                         |
                  +------+--------+
                  | Full-Duplex   |
                  | Communication |
                  +---------------+

Applications

- Secure IoT networks in hostile RF environments
- Defense-grade communication systems
- Industrial automation with real-time control
- Disaster-resilient and adaptive wireless networks

Contributors

- *Nitish Dandu*
- *Sai Harsha*
- *Rahul Chowdary*
- *Salla Shivesh*

