<div align="center">
<h1> RUMP: Resource Usage Multi-Step Prediciton</h1>
<!-- <--!span><font size="5", > Multi-Step Prediciton of Worker Resource Usage at the Extreme Edge
</font></span> -->
  
  Ruslan Kain, Sara A. Elsayed, Yuanzhu Chen, Hossam Hassanein 
<!-- <a href="https://www.researchgate.net/publication/363157892_Multi-step_Prediction_of_Worker_Resource_Usage_at_the_Extreme_Edge">Ruslan Kain</a> -->
<div><a href="https://www.researchgate.net/publication/363157892_Multi-step_Prediction_of_Worker_Resource_Usage_at_the_Extreme_Edge">[Multi-step Prediction of Worker Resource Usage at the Extreme Edge]</a></div> 

</div>


## Data
May be accessed on [Borealis](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/GOZAJE)
  

### Description

The collection and construction of this dataset were organized by the Queen's Telecommunications Research Lab (TRL) and led by Ruslan Kain, a Ph.D. student at TRL. The dataset includes dynamic resource usage information associated with running edge-native applications on a set of four heterogeneous Single Board Computers.
  
### Experimental Setup

Four Raspberry Pi 4 devices have 2, 2, 4, and 8 GB RAM sizes, and CPU frequencies of 1200, 1500, 1500, and 1800 MHz. This is to establish heterogeneity of the devices used and collected data and to enable data-based applications for Edge Computing Research. The resource usage measurements have a five-second granularity. We managed to collect more than 550 thousand unique data points representing the 768 hours of running applications on Raspberry Pi Devices

<td><img src=figures/RPis.jpg/></td>


### Descriptive Sample

<table>
<tr><th>Dataset </th></tr>
<tr><td>

|       Time      |       CPU Time (s)     |    Memory (%)    |    QOS (sec)    | Resource Usage State     | 
|:----------------:|:-----------------:|:---------------:| :---------------:|  :---------------:| 
|    `Breakfast`    |        0.6       |   33   |    1.2   |   `Idle`   |  
|     `Second Breakfast`    |        12.3       |   44    |    3.5   | `Augmented Reality` |
|     `Lunch`     |        15.6       |    55    |    4.1   | `Crypto Mining` |  
|      `Supper`     |        0.5       |    66    |    1.3   |  `Idle` |  
|     `Dinner` |        4.5       |     66    |    2.7   |   `Streaming` | 
|     `Midnight Snack`    |     9.2    |    11     |    3.4   |   `Gaming`   |  

</td></tr> </table>


