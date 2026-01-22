### **TL;DR: REFITT Performance Analysis Strategy**

The goal is to measure how quickly and accurately the model "solves" the supernova's parameters as data accumulates every 5 days.

---

### **1\. Core Metrics (The "What")**

* **Efficiency ($N\_{90}$):** The number of days/iterations until predictions for $M\_{zams}$ or mloss\_rate stay within 10% of their final values.  
* **Stability (Volatility):** Measuring the "jitter" between 5-day runs to identify when the model stops drastically changing its mind.  
* **Fit Accuracy (Residuals):** Calculating the difference between the model's predicted mag\_arr and the actual mag\_arr observed as the transient fades.  
* **Parameter Trajectories:** Tracking if specific values (like Progenitor Mass) tend to be systematically under- or over-estimated in early iterations.

---

### **2\. Extraction Method (The "How")**

By comparing sequential JSON files for the same ztf\_id, we will map the following data structures to our metrics:

| JSON Field | Analysis Method |
| :---- | :---- |
| **parameters** | Extract zams, mloss\_rate, and 56Ni. Plot these values against the **Phase** (days since explosion) for every 5-day run to find the "Convergence Point." |
| **mag\_arr** | Compare the array of predicted magnitudes from an early run against the array in the final "dimmed" run to calculate the **Integrated Error**. |
| **mjd\_arr** | Use these timestamps to align different runs on a master timeline to ensure the 5-day intervals are compared accurately. |
| **Phase** | Use this as the x-axis for all stability plots to see how much "light curve age" is required for a 90% accurate prediction. |

