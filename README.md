# SonarCore

**Advanced Hydroacoustic Sonar Simulation and Optimization Platform**

SonarCore is a comprehensive software platform for simulating and optimizing hydroacoustic sonar systems. It enables engineers and researchers to model underwater acoustic propagation, optimize CHIRP signal parameters, and analyze receiver chain performance before hardware implementation.

**🌊 Part of [Open Deep Water](https://github.com/ipmgroup/Open_Deep_Water) initiative**

## 🎯 Key Features

- **Physical Modeling**: Accurate simulation of underwater acoustic propagation using established models (Mackenzie, Francois-Garrison)
- **Signal Processing**: CHIRP signal generation and matched filtering with multiple window functions
- **Receiver Chain Simulation**: Detailed modeling of LNA, VGA, and ADC components with real hardware parameters
- **Parameter Optimization**: Automated optimization of system parameters to meet performance targets
- **Interactive GUI**: User-friendly interface for parameter input and result visualization
- **Component Library**: Extensive database of real transducers, amplifiers, and ADCs
- **Transducer Modeling**: Create BVD, MBVD, EBVD, Mason, and KLM models from Conductance and Susceptance data

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 📋 Supported Systems

**Note:** These simulators are currently under active development.

- **Sub-Bottom Profiler (SBP)**: Ground profile and penetration depth analysis
- **Side-Scan Sonar**: Seafloor imaging and backscattering analysis
- **USBL**: Underwater positioning systems
- **Underwater Acoustic Communication**: Wideband LFM chirp signals with correlation reception (18-34 kHz)
- **Harbor Defense**: Passive underwater target detection and localization system for port security using distributed hydrophone arrays and TDOA (Time Difference of Arrival) methods

## 🔬 Transducer Modeling

**✅ Completed:** Transducer model creation from Conductance and Susceptance data.

SonarCore now supports creating comprehensive transducer models from experimental impedance data:

- **BVD (Butterworth-Van Dyke)**: Basic 4-parameter model
- **MBVD (Modified BVD)**: Enhanced model with dielectric losses (5 parameters)
- **EBVD (Extended BVD)**: Multi-resonance model with harmonics (7+ parameters)
- **Mason Model**: Physical wave-based model for ultrasonic transducers
- **KLM Model**: Equivalent circuit model for hydroacoustic transducers

### Features

- **Data Import**: Load Conductance and Susceptance from CSV files
- **Model Fitting**: Automatic parameter optimization with multiple error metrics
- **Model Comparison**: Compare different models and select the best fit
- **Export**: Save fitted models with all parameters for use in SonarCore simulations
- **Visualization**: Interactive plots showing model fit quality

### Usage

```bash
# Run BVD model fitting tool
python scripts/bvd.py
```

For detailed information, see [TRANSDUCER_PARAMETERS_REQUIRED.md](TRANSDUCER_PARAMETERS_REQUIRED.md) and [scripts/transducer_models.md](scripts/transducer_models.md).

## 🔍 Sub-Bottom Profiler (SBP)

The SBP simulator enables detailed modeling of sub-bottom sediment layers and acoustic penetration analysis.

### Key Capabilities

- **Multi-Layer Sediment Modeling**: Configure complex sediment profiles with up to 10 different layer types
  - Natural sediments: Clay, Silt, Sand, Mud, Gravel, Soft Clay
  - Artificial materials: Concrete, Steel, Wood (underwater)
  - Biological materials: Biological tissue
- **CHIRP Signal Processing**: Linear CHIRP generation with configurable frequency range (3-12 kHz typical)
- **Matched Filtering**: Cross-correlation for echo detection and interface identification
- **Visualization Modes**:
  - **RAW Mode**: Full signal from surface including water column
  - **Ground Mode**: Sediment layers only (from first echo to end)
- **Interface Detection**: Automatic identification of layer boundaries with red markers
- **Signal Path Analysis**: Complete signal propagation path with transmission parameters
- **Layer Management**: Interactive GUI with layer reordering (Move Up/Down)

### Running SBP Simulator

```bash
# Run SBP GUI
python main_sbp.py
```

### Sediment Layer Properties

Each layer is characterized by:
- **Density** (ρ): kg/m³
- **Sound Speed** (c): m/s
- **Attenuation** (α): dB/m at 1 kHz (frequency-dependent)
- **Thickness**: Layer depth, m
- **Acoustic Impedance**: Z = ρ × c (calculated automatically)

### Output

The simulator provides:
- **Profile Visualization**: A-scan with raw signal and correlation
- **Detected Interfaces**: Marked with red dots showing layer boundaries
- **Signal Path Diagram**: Complete transmission chain analysis
- **Results Summary**: Detected interfaces, depths, amplitudes, and transmission parameters

For detailed technical specifications, see [SBP.md](SBP.md).

## 🏗️ Architecture

SonarCore consists of three main modules:

- **CORE**: Computational engine with physical models and optimization algorithms
- **DATA**: Component library with real hardware parameters (transducers, LNA, VGA, ADC)
- **GUI**: Interactive interface for simulation and visualization

## 📚 Documentation

Complete technical specifications and documentation:

- **[Sonar.md](Sonar.md)** - Complete technical documentation for the SonarCore platform (in Russian)
- **[OSO.md](OSO.md)** - Output Stage Optimization (OSO) technical specification for underwater acoustic communication systems with CHIRP modulation. Includes multi-parameter optimization of Class D amplifier, transformer, matching inductance, and transducer parameters
- **[connection.md](connection.md)** - Underwater acoustic communication simulator specification (wideband LFM chirp, correlation reception). Includes signal encoding (up/down chirp for bits), channel modeling, receiver chain, and BER analysis. Status of implementation in SonarCore is documented
- **[harbor_defense.md](harbor_defense.md)** - Harbor defense system specification for passive underwater target detection and localization in port areas. Includes distributed hydrophone array modeling, TDOA (Time Difference of Arrival) estimation, multipath propagation, and target tracking. Covers mathematical models for signal propagation, noise, and environmental effects in shallow water
- **[SBP.md](SBP.md)** - Sub-Bottom Profiler (SBP) technical specification with ground profile and penetration depth modeling
- **[side_scan.md](side_scan.md)** - Side-Scan Sonar technical specification (60-900 kHz) with digital IQ processing
- **[USBL.md](USBL.md)** - Ultra-Short Baseline positioning system technical specification
- **[INSTALL.md](INSTALL.md)** - Installation and setup guide

## 🔬 Scientific Models

- **Water Propagation**: Mackenzie (1981) sound speed, Francois-Garrison (1982) attenuation
- **Signal Processing**: CHIRP generation, matched filtering, envelope detection
- **Receiver Modeling**: Thermal noise, quantization, clipping, ENOB calculation
- **Optimization**: Multi-parameter optimization with constraint handling

## 💻 Requirements

- Python 3.8+
- NumPy, SciPy
- PyQt5 (for GUI)

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please refer to the technical documentation for implementation details.

---

**SonarCore** - Bringing precision to underwater acoustic system design.

