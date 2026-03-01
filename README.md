# Audio-Utilities

A collection of tools for audio measurement and analysis.

## Projects

### [audio_gen_test](audio_gen_test/)
Measures cumulative audio degradation through repeated DAC/ADC conversion cycles. Plays audio through your interface, records it back, and repeats for N generations.

### [null_tests](null_tests/)
Empirical demonstration that FLAC decodes bit-identically to WAV through a real DAC/ADC signal chain. Plays WAV and FLAC through hardware, captures the loopback, and nulls the two captures.

### [korf_compliance_calc](korf_compliance_calc/)
Phono cartridge compliance and effective mass calculator. Computes and plots the frequency response of a cartridge/tonearm system.
