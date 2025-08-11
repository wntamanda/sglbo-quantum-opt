# SGLBO vs Other Optimizers for Quantum Neural Networks

This repo contains my Python implementation of **Stochastic Gradient Line Bayesian Optimization** (SGLBO) and a comparison against other common optimizers (e.g., SPSA, COBYLA, Adam) when training **Quantum Neural Networks** (QNNs) under different noise conditions.

Based on the method from:  
📄 [Stochastic Gradient Line Bayesian Optimization for Efficient Noise-Robust Optimization of Parameterized Quantum Circuits](https://arxiv.org/abs/2111.07952)  
Shiro Tamiya, Hayata Yamasaki (2021)

## What this project does

- Implements SGLBO from scratch for variational quantum circuits.
- Runs experiments on QNNs (via Qiskit Machine Learning).
- Compares SGLBO against other optimizers across **noiseless** and **noisy** simulation levels.
- Includes a reusable experiment framework so datasets, optimizers, and noise profiles can be swapped in without rewriting code.
- Prepares for eventual runs on **IBM Quantum hardware**.

## Why

Training quantum neural networks is hard because quantum hardware is noisy and measurement results are uncertain. Most classical optimizers struggle in this setting and often get stuck. SGLBO is a Bayesian optimization method that adapts more intelligently to noise. This project tests whether it performs better than other common optimizers when training QNNs.

## Status

- SGLBO implemented and running.
- SPSA comparison implemented.
- Adding more optimizers (Adam, COBYLA).
- Preparing for real-hardware tests.

## Notes

If anything breaks… pretend you didn’t see it (or open an issue 😔).