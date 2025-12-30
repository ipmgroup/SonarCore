# SonarCore

**Advanced Hydroacoustic Sonar Simulation and Optimization Platform**

SonarCore is a comprehensive software platform for simulating and optimizing hydroacoustic sonar systems. It enables engineers and researchers to model underwater acoustic propagation, optimize CHIRP signal parameters, and analyze receiver chain performance before hardware implementation.

## 🎯 Key Features

- **Physical Modeling**: Accurate simulation of underwater acoustic propagation using established models (Mackenzie, Francois-Garrison)
- **Signal Processing**: CHIRP signal generation and matched filtering with multiple window functions
- **Receiver Chain Simulation**: Detailed modeling of LNA, VGA, and ADC components with real hardware parameters
- **Parameter Optimization**: Automated optimization of system parameters to meet performance targets
- **Interactive GUI**: User-friendly interface for parameter input and result visualization
- **Component Library**: Extensive database of real transducers, amplifiers, and ADCs

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 📋 Supported Systems

- **Sub-Bottom Profiler (SBP)**: Ground profile and penetration depth analysis
- **Side-Scan Sonar**: Seafloor imaging and backscattering analysis
- **USBL**: Underwater positioning systems

## 🏗️ Architecture

SonarCore consists of three main modules:

- **CORE**: Computational engine with physical models and optimization algorithms
- **DATA**: Component library with real hardware parameters (transducers, LNA, VGA, ADC)
- **GUI**: Interactive interface for simulation and visualization

## 📚 Documentation

Complete technical specifications and documentation:

- **[Sonar.md](Sonar.md)** - Complete technical documentation for the SonarCore platform (in Russian)
- **[SBP.md](SBP.md)** - Sub-Bottom Profiler (SBP) technical specification with ground profile and penetration depth modeling
- **[side_scan.md](side_scan.md)** - Side-Scan Sonar technical specification (300-900 kHz) with digital IQ processing
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

