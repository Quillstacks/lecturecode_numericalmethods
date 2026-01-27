# Numerical Methods Lecture Code

Numerical Methods are essential for solving mathematical problems that cannot be addressed analytically. This course will cover fundamental concepts, error analysis, problem conditioning, stability, and a variety of numerical techniques to equip you with the skills needed to implement reliable and efficient algorithms in scientific computing and engineering applications.

This course will teach you everything you need to know to become proficient with numerical methods, and how to put them to good use in machine learning, data science, and engineering contexts.



## Virtual Machine Setup

Virtual machines share resources and network interfaces with the host machine. Be aware of this when allocating physical resources. We recommend allocating at least 2 CPU cores, 20GB of disk space, and 4096MB of RAM to your VM for a smooth experience. If you have limited resources you can try with less—it will still get you far.

To get started with the provided virtual machine:

1. **Download and install VirtualBox** from [https://www.virtualbox.org/](https://www.virtualbox.org/). The version used for this course was *VirtualBox-7.2.4-170995-Win*.
2. **Load the provided virtual machine**. The VM uses [Ubuntu 22.04 LTS](https://releases.ubuntu.com/22.04/) as a base. The `.ora` file can be found [here](https://www.tobedone.com).
3. **Login** with username `user` and password `qwertz`. Be aware of possible keyboard layout differences.
4. **Pull this Repository** into the VM. It is precloned there, but you need to make sure to have the most up-to-date version.

VMs are charming because they enable us to be on the same setup, configurations and starting point. This does not mean you need to stick around there, rather see it as a fallback, but feel free to run the code in your setup and environment.




## Python Version

This project is tested with **Python 3.10**. It is recommended to use a matching Python version for best results. You can check your Python version with:

```sh
python --version
```


To run the code examples on your own machine, follow these steps:

1. **Create a virtual environment (venv):**
	```sh
	python -m venv .venv
	```
2. **Activate the virtual environment:**
	- On Windows:
	  ```sh
	  .venv\Scripts\activate
	  ```
	- On macOS/Linux:
	  ```sh
	  source .venv/bin/activate
	  ```
3. **Install the required packages:**
	```sh
	pip install -r requirements.txt
	```

You are now ready to run the scripts in this repository!

---

## Course Structure

- Each lecture will be accompanied by Python scripts that demonstrate the key concepts and algorithms discussed in class.
- The code examples are designed to be practical and directly applicable to real-world problems.

Topics covered
- Lecture 1: [Tapping into Computational Power](./1_TappingIntoComputationalPower/) - Discretization 
- Lecture 2: [Getting used to Errors everywhere](./2_GettingUsedToErrorsEverywhere/) - Floating Point Representation
- Lecture 3: [Some call it Error. I call it Character](./3_WeCallItCharacter/) - Error Analysis of Numerical Solutions
