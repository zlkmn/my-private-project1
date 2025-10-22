# DRL-based Partial Offloading with D2D in WPC-MEC Networks

## 1. Overview

This project simulates a wireless powered mobile-edge computing (WPC-MEC) network where multiple wireless devices (WDs) can partially offload their computation tasks. The offloading strategy is learned by a Deep Reinforcement Learning (DRL) agent.

This work extends the concepts from the paper "Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks" by introducing two key innovations:
1.  **Partial Offloading**: WDs can split a task and offload fractions to different destinations.
2.  **Device-to-Device (D2D) Collaboration**: WDs can offload tasks not only to the Access Point (AP) but also to other peer WDs, creating a collaborative computing environment.

The system operates based on a robust three-phase time model:
1.  **Energy Harvesting Phase**: All WDs harvest energy from the AP.
2.  **Sequential Transmission Phase**: WDs transmit their offloaded data packets in a TDMA fashion.
3.  **Parallel Execution Phase**: All nodes (AP and WDs) compute their assigned task portions in parallel.

## 2. Project Structure
```
DROO_D2D_PARTIAL_OFFLOADING/
├── config.py             # All system parameters and hyperparameters
├── environment.py        # Generates channel and task states
├── resource_allocator.py # The "Critic": Solves the resource allocation problem
├── agent.py              # The DRL Agent (MemoryDNN): The "Actor"
├── main.py               # Main script to run the simulation
├── utils.py              # Helper functions for plotting and saving results
└── README.md             # This documentation
```
## 3. Setup

### Dependencies
The project requires the following Python libraries. You can install them using pip:

```bash
pip install numpy scipy torch matplotlib pandas tqdm
```

## 4. How to Run
1.  **Configure the Simulation**: Open config.py to adjust any system parameters, such as the number of devices (N_WDs), simulation length (NUM_TIME_FRAMES), or DRL hyperparameters.

2.  **Run the Main Script**: Execute the main simulation file from your terminal.
```bash
python main.py
 ```

3.  **View Results**:

Two plots will be displayed after the simulation: one for the objective function's history and one for the DRL agent's training loss.

A new directory named results/ will be created, containing text files with the raw data for further analysis.

You now have the complete, modular, and well-documented project. You can run `main.py` directly, and it will start the training process. You can then analyze the plots and the saved data in the `results` folder to see if the agent successfully learns to perform intelligent partial offloading.

## 5. System Model
# System Model

We consider a Wireless Powered Mobile-Edge Computing (WPC-MEC) scenario consisting of:
* One Access Point (AP), denoted as node 0.
* $N$ Internet of Things (IoT) Wireless Devices (WDs), denoted by the set $\mathcal{N} = \{1, 2, ..., N\}$.

The AP is responsible for Wireless Power Transfer (WPT) to all WDs and can also act as a computation offloading destination. The WDs employ a **partial offloading** mechanism, allowing them to split their computation tasks and offload portions to the AP, other idle WDs (Device-to-Device, D2D), or execute them locally.

## Channel and Task Model

* **Channel Gain**: The channel gain between any two nodes $i, j \in \{0, 1, ..., N\}$ is denoted by $h_{i,j}$. We assume channels are reciprocal ($h_{i,j} = h_{j,i}$) and $h_{i,i} = 0$. The channel gains form a symmetric matrix $\mathbf{H}$.
* **Task Size**: Each WD $i \in \mathcal{N}$ has a computation task of size $c_i$ bits. The task sizes form a vector $c = [c_1, ..., c_N]$.
* **Offloading Decision**: The partial offloading decision is represented by a matrix $X = \{x_{i,j}\}$, where $x_{i,j}$ is the fraction of task $c_i$ processed by node $j \in \{0, 1, ..., N\}$. We must have $\sum_{j=0}^{N} x_{i,j} = 1$ and $x_{i,j} \ge 0$ for all $i \in \mathcal{N}$. $x_{i,i}$ represents the fraction computed locally by WD $i$.

## Three-Phase Time Structure

The system operates in time frames, each divided into three phases:

### Phase 1: Energy Harvesting

* **Duration**: $a$, where $a \in (0, 1)$ is an optimization variable.
* **Process**: The AP broadcasts RF energy with power $P$.
* **Harvested Energy**: WD $j \in \mathcal{N}$ harvests energy:
    $$E_j^{harv} = \mu P h_{0,j} a S$$
    where $\mu$ is the energy harvesting efficiency, and $S$ is an optional scaling factor for system feasibility.

### Phase 2: Offloading Transmission

* **Total Duration**: $T_{offload}$.
* **Mechanism**: A time-division mechanism where WDs transmit sequentially in order $1, 2, ..., N$. During WD $i$'s turn, it **serially** transmits the task chunks $c_i x_{i,j}$ to all its chosen destination nodes $j \neq i$.
* **Data Rate**: The rate from WD $i$ to node $j$ depends on WD $i$'s transmit power $P_i$ (an optimization variable):
    $$r_{i,j}(P_i) = \frac{B}{v_u} \log_2 \left( 1 + \frac{P_i h_{i,j}}{N_0} \right)$$
    where $B$ is the bandwidth, $v_u$ is overhead, and $N_0$ is noise power.
* **Single WD Transmission Time**: The time taken by WD $i$ to complete all its serial transmissions is the sum of individual transmission times:
    $$t_{i}^{tx}(X, P_i) = \sum_{j \in \{0, ..., N\}, j \neq i, x_{i,j}>0} \left\{ \frac{c_i x_{i,j}}{r_{i,j}(P_i)} \right\}$$
    If WD $i$ only computes locally ($x_{i,j}=0$ for all $j \neq i$), then $t_{i}^{tx} = 0$.
* **Total Transmission Duration**: The duration of this phase is the sum of all WDs' transmission times:
    $$T_{offload}(X, P) = \sum_{i=1}^{N} t_{i}^{tx}(X, P_i)$$
    where $P = [P_1, ..., P_N]$.

### Phase 3: Parallel Execution

* **Duration**: $T_{execute}$.
* **Process**: After all transmissions are complete, all WDs **simultaneously** execute their assigned computation loads.
* **Computation Load**: The total load $L_j$ (in bits) for WD $j$ is its local part plus parts received from others:
    $$L_j(X, c) = c_j x_{j,j} + \sum_{i \in \mathcal{N}, i \neq j} c_i x_{i,j}$$
* **Single WD Computation Time**: The time for WD $j$ to compute its load depends on its CPU frequency $f_j$ (an optimization variable):
    $$t_{j}^{comp}(X, c, f_j) = \frac{L_j(X, c) \times \phi}{f_j}$$
    where $\phi$ is the number of CPU cycles per bit.
* **Total Execution Duration**: The duration of this phase is determined by the slowest WD:
    $$T_{execute}(X, c, f) = \max_{j \in \mathcal{N}} \{ t_{j}^{comp}(X, c, f_j) \}$$
    where $f = [f_1, ..., f_N]$. (AP computation time is neglected).

## Total Latency

The total time latency for one time frame is the sum of the durations of the three phases:
$$T_{total}(X, a, f, P, c) = a + T_{offload}(X, P) + T_{execute}(X, c, f)$$

## Energy Consumption Model

We use a computation energy model defined as $E_{comp} = k f^2 (\text{workload in bits}) \phi$, where $k_j$ is the energy efficiency coefficient for WD $j$.

* **Task $i$'s Total Energy ($E_i^{task}$)**: Used in the objective function. This represents the total energy consumed across the network *for completing task i*.
    $$E_i^{task}(X, f, P, c) = E_{i,loc} + E_{i,tx} + E_{i,d2d\_comp}$$
    * Local Computation Energy: $E_{i,loc} = k_i f_i^2 (c_i x_{i,i}) \phi$
    * Transmission Energy: $E_{i,tx} = P_i \times t_{i}^{tx}(X, P_i)$
    * D2D Computation Energy (energy spent by *other* WDs for task *i*): $E_{i,d2d\_comp} = \sum_{j \in \mathcal{N}, j \neq i} k_j f_j^2 (c_i x_{i,j}) \phi$

* **WD $j$'s Self-Consumed Energy ($E_j^{self}$)**: Used for the energy constraint. This represents the total energy actually drained from WD $j$'s battery.
    $$E_j^{self}(X, f, P, c) = E_{j,comp\_total} + E_{j,tx}$$
    * Total Computation Energy (spent by WD $j$ for *itself and others*): $E_{j,comp\_total} = k_j f_j^2 L_j(X, c) \phi$
    * Transmission Energy (spent by WD $j$ for *its task*): $E_{j,tx} = P_j \times t_{j}^{tx}(X, P_j)$

## Optimization Problem

The goal is to jointly optimize the offloading decisions $X$, energy harvesting time $a$, CPU frequencies $f$, and transmit powers $P$ to minimize a weighted sum of total latency and total task energy, subject to constraints.

$$\min_{X, a, f, P} \quad w_t T_{total}(X, a, f, P, c) + w_E \sum_{i=1}^{N} E_i^{task}(X, f, P, c)$$

**Subject to:**

1.  **Offloading Ratio Sum Constraint**:
    $$\sum_{j=0}^{N} x_{i,j} = 1, \quad \forall i \in \mathcal{N}$$
2.  **Non-negativity Constraint**:
    $$x_{i,j} \ge 0, \quad \forall i \in \mathcal{N}, j \in \{0, ..., N\}$$
3.  **Energy Neutrality Constraint**:
    $$E_j^{self}(X, f, P, c) \le E_j^{harv}(a), \quad \forall j \in \mathcal{N}$$
    (This is typically handled via a penalty term in the objective function).
4.  **Harvesting Time Constraint**:
    $$0 < a < 1$$
5.  **CPU Frequency Constraints**:
    $$f_{min} \le f_j \le f_{max}, \quad \forall j \in \mathcal{N}$$
6.  **Transmit Power Constraints**:
    $$P_{min} \le P_j \le P_{max}, \quad \forall j \in \mathcal{N}$$

Here, $w_t$ and $w_E$ are weighting factors, where $w_E$ might include a scaling factor to balance the numerical scales of time and energy.