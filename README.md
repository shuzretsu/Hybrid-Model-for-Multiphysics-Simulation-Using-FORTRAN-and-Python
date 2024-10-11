
## **1. Prerequisites**

### **A. Required Software**

To run the multiphysics simulation, we will need software such as :

1. **Fortran Compiler**: it is required to compile the Fortran code.
    - Install with: 
      ```bash
      sudo apt install gfortran
      ```
   
2. **CUDA Toolkit**: required to run the CUDA files for GPU acceleration.
    - Install CUDA following the official guide: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

3. **Python**: Python 3.x with the necessary libraries.
    - Just install Python and required packages:
      ```bash
      sudo apt install python3 python3-pip
      pip3 install numpy matplotlib
      ```

### **B. System Requirements**

- **OS**: Linux/Ubuntu recommended for better compatibility with CUDA and Fortran. 👍
- **GPU**: A CUDA-compatible GPU for running GPU-accelerated code.
- **Compiler**: Make sure `gcc`, `gfortran`, and `nvcc` (from CUDA) are available.

---

## **2. Installation and Setup**

### **A. Compiling Fortran Code**

Fortran code need to handles the core simulation calculations such as (heat conduction, fluid flow, and molecular dynamics). To compile the Fortran code we need to do this step:

1. **Navigate to the Fortran directory**:
   ```bash
   cd multiphysics_simulation/src/fortran
   ```

2. **Compile using Makefile**:
   The `makefile` is already provided in the folder, so simply run:
   ```bash
   make
   ```
   This will generate executables for the Fortran programs:
   - `heat_conduction`
   - `fluid_flow`
   - `molecular_dynamics`

### **B. CUDA Acceleration**

now we get to CUDA acceleration, it is used for solving heat conduction and fluid flow faster on the GPU.<br>
compile the CUDA files:

1. **Navigate to the CUDA directory**:
   ```bash
   cd multiphysics_simulation/cuda
   ```

2. **Compile CUDA programs**:
   - Fluid Solver:
     ```bash
     nvcc fluid_solver.cu -o fluid_solver
     ```
   - Heat Solver:
     ```bash
     nvcc heat_solver.cu -o heat_solver
     ```

   These will generate GPU-accelerated executables for the heat and fluid solvers.


### **C. Python Setup**

Now for the Python scripts we do handle visualization, parameter estimation, and orchestration between different components.

1. **Install the Python packages with pip**:
   ```bash
   pip3 install numpy matplotlib
   ```

2. **Running the main simulation**:
   The `main.py` script orchestrates all of the components such as Fortran, CUDA, and Python.
   - Run the main script:
     ```bash
     python3 multiphysics_simulation/src/python/main.py
     ```

   What would the script do?
   - Load initial conditions from `data/initial_conditions.txt`
   - Call the Fortran and CUDA solvers
   - Generate output files and visualizations

---

## **3. Running Tests**

And now, to verify that the system is working correctly, there are a series of tests implemented in Python:

1. **Navigate to the tests directory**:
   ```bash
   cd multiphysics_simulation/tests
   ```

2. **Running the heat conduction test**:
   ```bash
   python3 test_heat_conduction.py
   ```

3. **Running the fluid flow test**:
   ```bash
   python3 test_fluid_flow.py
   ```

4. **Running the molecular dynamics test**:
   ```bash
   python3 test_molecular_dynamics.py
   ```

Each test will verify the output from the corresponding Fortran or CUDA simulation.

---

## **4. Output Data**

The results of the simulations will be stored in the `data/results/` directory, with output files for each simulation.

- **Heat Map Output**: `heat_map_output.png`
- **Velocity Field Output**: `velocity_field_output.png`
- **Molecular Dynamics Results**: `molecular_dynamics_results.txt`

---

## **5. Additional Information**

### **A. Visualization**

The `visualize.py` script in the Python folder is responsible for generating visual representations of the simulation results.

- You can use the following command to visualize specific outputs:
  ```bash
  python3 visualize.py
  ```

### **B. Parameter Estimation**

The `parameter_estimation.py` script is designed for estimating key parameters from the simulation. It can be extended to adjust parameters dynamically based on observed data from the simulation.

---

## **6. IF THERE IS TROUBLE?**

Simply just do some troubleshoot on :

- **Compiler Issues**: Ensure if `gfortran`, `gcc`, and `nvcc`  correctly installed and accessible from the terminal.
- **CUDA Issues**: Check if your GPU is CUDA-compatible and with the correct version of CUDA being installed.
- **Python Issues**: Make sure all of required Python libraries stuff are installed via `pip3` and you're using Python 3.x.

# NOTES
The code still in building and Im still confuse, but ok, since it is just for fun, i will update it.
