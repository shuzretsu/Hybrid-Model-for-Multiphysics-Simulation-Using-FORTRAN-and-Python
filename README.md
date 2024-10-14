# Tutorial on How to Install it
## **1. Prerequisites**

### **A. Required Software**

To run the multiphysics simulation, we will need software such as :

1. **Fortran Compiler**: to compile the Fortran code.
    - Install with: 
      ```bash
      sudo apt install gfortran
      ```
   
2. **CUDA Toolkit**: to run the CUDA files for GPU acceleration.
    - Install CUDA following the official guide: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

3. **Python**: Python 3.x
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
<br>
