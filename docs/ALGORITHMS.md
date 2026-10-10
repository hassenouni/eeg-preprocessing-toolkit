# Offline and causal processing

`bandpass` uses Butterworth second-order sections. `notch` uses a second-order
IIR notch. Their defaults apply forward-backward filtering and require adequate
signal length; SciPy raises a clear length error when padding is impossible.
`causal=True` chooses a forward-only filter with zero initial state, suitable for a
whole block, not a state-preserving streaming implementation.

`resample` uses rational-ratio polyphase filtering with anti-aliasing.
`average_reference` subtracts the across-channel mean; select EEG channels before
referencing and consider montage differences. Channel quality flags use volts.
`ChannelStandardizer.fit(train)` stores per-channel means and standard deviations.
`transform(test)` reuses them and never fits on test data. Constant scales become 1.
Zero-phase filters use future samples and must not be presented as causal detectors.

## References

- [Method or implementation reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)
- [CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/)
- [Siena 1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
